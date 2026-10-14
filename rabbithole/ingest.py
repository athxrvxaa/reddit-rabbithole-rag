"""Stage 1: stream the HF IAmA comments split into SQLite (resumable, memory-light)."""
from __future__ import annotations

import time

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


def ingest(
    settings: Settings,
    max_comments: int | None = None,
    batch_size: int = 20_000,
    max_retries: int = 30,
) -> None:
    """Stream comments into SQLite. Safe to Ctrl-C and re-run: it resumes where it stopped."""
    from datasets import load_dataset  # lazy: only needed for this stage

    con = connect(settings.db_path)
    consumed = int(get_state(con, "ingest_consumed", "0") or 0)
    if get_state(con, "ingest_complete") == "1" and max_comments is None:
        print("Ingest already complete. Delete data/rabbithole.db to start over.")
        return

    print(f"Streaming {settings.hf_dataset} [{settings.hf_split}] (resuming at row {consumed:,})")
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
                if t:
                    buf.append(t)
                if seen >= batch_size:
                    consumed = _flush(con, buf, consumed, seen)
                    buf, seen = [], 0
                    rate = consumed / max(time.time() - t0, 1)
                    print(f"  {consumed:,} rows consumed ({rate:,.0f}/s)")
                    retries = 0
                    if max_comments and consumed >= max_comments:
                        capped = True
                        break
            if not capped:
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
