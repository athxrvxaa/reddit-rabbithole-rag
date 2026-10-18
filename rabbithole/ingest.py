"""Stage 1: stream the HF IAmA comments split into SQLite (resumable, memory-light)."""
from __future__ import annotations

import time
import zlib

from .config import Settings
from .db import connect, get_state, set_state
from .text import bare_id

BAD_AUTHORS = {"[deleted]", "[removed]", "AutoModerator", ""}
BAD_BODIES = {"[deleted]", "[removed]", ""}


def _row_to_tuple(row: dict):
    body = (row.get("body") or "").strip()
    author = row.get("author") or ""
    if author in BAD_AUTHORS or body in BAD_BODIES:
        return None
    link = bare_id(str(row.get("link_id") or ""))
    cid = bare_id(str(row.get("id") or ""))
    if not link or not cid:
        return None
    try:
        score = int(row.get("score") or 0)
    except (TypeError, ValueError):
        score = 0
    try:
        created = int(float(row.get("created_utc") or 0))
    except (TypeError, ValueError):
        created = 0
    return (cid, link, str(row.get("parent_id") or ""), author, body, score, created)


def keep_thread(link_id: str, rate: float) -> bool:
    """Deterministic thread-level sampling: the same thread is always kept or always dropped,
    so sampled threads are complete and re-runs/resumes stay consistent."""
    if rate >= 1.0:
        return True
    return (zlib.crc32(link_id.encode("utf-8")) % 100_000) < int(rate * 100_000)


def ingest(
    settings: Settings,
    max_comments: int | None = None,
    batch_size: int = 20_000,
    max_retries: int = 30,
    thread_sample_rate: float = 1.0,
    target_rows: int | None = None,
) -> None:
    """Stream comments into SQLite. Safe to Ctrl-C and re-run: it resumes where it stopped.

    max_comments       stop after SCANNING this many rows of the stream (first-N slice)
    thread_sample_rate keep only this fraction of threads (whole threads, chosen by hash of thread id)
    target_rows        stop once this many rows have been KEPT (safety cap when sampling)
    """
    from datasets import load_dataset  # lazy: only needed for this stage

    if not (0 < thread_sample_rate <= 1):
        raise ValueError("thread_sample_rate must be in (0, 1]")
    con = connect(settings.db_path)
    consumed = int(get_state(con, "ingest_consumed", "0") or 0)
    if get_state(con, "ingest_complete") == "1":
        print("Ingest already complete. Delete the DB to start over.")
        return
    prev_rate = get_state(con, "ingest_sample_rate")
    if consumed and prev_rate is not None and abs(float(prev_rate) - thread_sample_rate) > 1e-9:
        raise SystemExit(
            f"This DB was started with --thread-sample-rate {prev_rate} but you passed {thread_sample_rate}. "
            "Use the same rate to resume, or delete the DB to start over."
        )
    set_state(con, "ingest_sample_rate", repr(thread_sample_rate))
    con.commit()
    kept_total = con.execute("SELECT COUNT(*) FROM comments").fetchone()[0]

    print(
        f"Streaming {settings.hf_dataset} [{settings.hf_split}] (resuming at row {consumed:,}; "
        f"thread sample rate {thread_sample_rate}; {kept_total:,} rows already stored)"
    )
    retries = 0
    t0 = time.time()
    while True:
        try:
            ds = load_dataset(settings.hf_dataset, split=settings.hf_split, streaming=True)
            if consumed:
                ds = ds.skip(consumed)
            buf: list[tuple] = []
            seen = 0
            capped = False
            for row in ds:
                seen += 1
                t = _row_to_tuple(row)
                if t and keep_thread(t[1], thread_sample_rate):
                    buf.append(t)
                if seen >= batch_size:
                    kept_total += len(buf)
                    consumed = _flush(con, buf, consumed, seen)
                    buf, seen = [], 0
                    rate = consumed / max(time.time() - t0, 1)
                    print(f"  scanned {consumed:,} rows ({rate:,.0f}/s) | kept {kept_total:,}")
                    retries = 0
                    if (max_comments and consumed >= max_comments) or (target_rows and kept_total >= target_rows):
                        capped = True
                        break
            if not capped:
                kept_total += len(buf)
                consumed = _flush(con, buf, consumed, seen)
                set_state(con, "ingest_complete", "1")
                con.commit()
            break
        except KeyboardInterrupt:
            print("Interrupted. Progress is saved; re-run to resume.")
            raise
        except Exception as e:  # network hiccup etc.
            retries += 1
            if retries > max_retries:
                raise
            wait = min(120, 5 * retries)
            print(f"  stream error ({type(e).__name__}: {e}); retry {retries}/{max_retries} in {wait}s")
            time.sleep(wait)

    print("Indexing comments by thread...")
    con.execute("CREATE INDEX IF NOT EXISTS idx_comments_link ON comments(link_id)")
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
    t = con.execute("SELECT COUNT(DISTINCT link_id) FROM comments").fetchone()[0]
    print(f"Done: {n:,} comments across {t:,} threads stored in {settings.db_path}")


def _flush(con, buf, consumed, seen) -> int:
    con.executemany("INSERT OR IGNORE INTO comments VALUES (?,?,?,?,?,?,?)", buf)
    consumed += seen
    set_state(con, "ingest_consumed", str(consumed))
    con.commit()
    return consumed
