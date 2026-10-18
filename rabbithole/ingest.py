"""Stage 1: stream the HF IAmA comments split into SQLite (resumable, memory-light)."""
from __future__ import annotations

import time
import zlib
import json
from pathlib import Path

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
    method: str = "parquet",
) -> None:
    """Stream comments into SQLite. Safe to Ctrl-C and re-run: it resumes where it stopped.

    max_comments       stop after SCANNING this many rows of the stream (first-N slice)
    thread_sample_rate keep only this fraction of threads (whole threads, chosen by hash of thread id)
    target_rows        stop once this many rows have been KEPT (safety cap when sampling)
    """
    from datasets import load_dataset  # lazy: only needed for this stage

    if not (0 < thread_sample_rate <= 1):
        raise ValueError("thread_sample_rate must be in (0, 1]")
    if method not in {"parquet", "stream"}:
        raise ValueError("method must be 'parquet' or 'stream'")
    _quiet_hf_progress()
    if method == "parquet":
        if _ingest_parquet(settings, max_comments, thread_sample_rate, target_rows):
            return
        print("Parquet shards were not found for the configured split; falling back to streaming.")
    con = connect(settings.db_path)
    consumed = int(get_state(con, "ingest_consumed", "0") or 0)
    session_start = consumed
    if get_state(con, "ingest_complete") == "1":
        print("Ingest already complete. Delete the DB to start over.")
        return
    prev_rate = get_state(con, "ingest_sample_rate")
    has_progress = bool(consumed or con.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
                       or get_state(con, "ingest_parquet_completed"))
    if has_progress and prev_rate is not None and abs(float(prev_rate) - thread_sample_rate) > 1e-9:
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
    from .source_info import split_size
    advertised_size = split_size(settings)
    mismatch_warned = False
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
                    rate = _session_rate(consumed, session_start, time.time() - t0)
                    if advertised_size and consumed > advertised_size and not mismatch_warned:
                        print("WARNING: dataset metadata count looks stale")
                        mismatch_warned = True
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
            print_ingest_state(con)
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


def _quiet_hf_progress() -> None:
    try:
        from datasets.utils.logging import disable_progress_bar, set_verbosity_error
        disable_progress_bar()
        set_verbosity_error()
    except (ImportError, AttributeError):
        pass
    try:
        from huggingface_hub.utils import disable_progress_bars, logging as hf_logging
        disable_progress_bars()
        hf_logging.set_verbosity_error()
    except (ImportError, AttributeError):
        pass


def _ingest_parquet(settings: Settings, max_comments: int | None, rate: float, target_rows: int | None) -> bool:
    """Ingest HF parquet shards independently; return False when shard discovery fails."""
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as e:
        raise SystemExit("Parquet ingest requires pyarrow and huggingface_hub; install requirements-pipeline.txt.") from e

    files = HfApi().list_repo_files(settings.hf_dataset, repo_type="dataset")
    split = settings.hf_split.lower()
    shards = sorted(f for f in files if split in Path(f).name.lower() and f.lower().endswith(".parquet"))
    if not shards:
        return False

    con = connect(settings.db_path)
    completed = set(json.loads(get_state(con, "ingest_parquet_completed", "[]") or "[]"))
    previous_rate = get_state(con, "ingest_sample_rate")
    has_progress = bool(completed or con.execute("SELECT COUNT(*) FROM comments").fetchone()[0])
    if has_progress and previous_rate is not None and abs(float(previous_rate) - rate) > 1e-9:
        raise SystemExit(f"This DB was started with --thread-sample-rate {previous_rate} but you passed {rate}. Use the same rate to resume, or delete the DB to start over.")
    set_state(con, "ingest_sample_rate", repr(rate)); con.commit()
    consumed = int(get_state(con, "ingest_parquet_consumed", "0") or 0)
    session_start = consumed
    kept_total = con.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
    if target_rows and kept_total >= target_rows:
        print(f"Target already reached: {kept_total:,} kept rows.")
        return True
    pending = [f for f in shards if f not in completed]
    if not pending:
        set_state(con, "ingest_complete", "1"); con.commit()
        print(f"Ingest already complete: {kept_total:,} comments.")
        return True

    download_dir = settings.data_dir / "ingest_shards"
    download_dir.mkdir(parents=True, exist_ok=True)
    sampled: dict[str, bool] = {}
    from concurrent.futures import ThreadPoolExecutor
    print(f"Reading {len(pending):,}/{len(shards):,} parquet shards from {settings.hf_dataset} [{settings.hf_split}]")
    t0 = time.time()
    from .source_info import split_size
    advertised_size = split_size(settings)
    mismatch_warned = False
    def download(filename):
        return filename, hf_hub_download(settings.hf_dataset, filename, repo_type="dataset", local_dir=str(download_dir))

    # Keep no more than four downloads in flight. Process and remove each local shard before scheduling more.
    with ThreadPoolExecutor(max_workers=4) as pool:
        next_index = 0
        futures = []
        while next_index < min(4, len(pending)):
            futures.append(pool.submit(download, pending[next_index])); next_index += 1
        while futures:
            future = futures.pop(0)
            filename, local_path = future.result()
            try:
                parquet = pq.ParquetFile(local_path)
                capped = False
                shard_rows = int(parquet.metadata.num_rows)
                shard_scanned = 0
                for batch in parquet.iter_batches(columns=["id", "link_id", "parent_id", "author", "body", "score", "created_utc"], batch_size=100_000):
                    buf = []
                    for row in batch.to_pylist():
                        if max_comments and consumed >= max_comments:
                            capped = True; break
                        consumed += 1
                        shard_scanned += 1
                        row_tuple = _row_to_tuple(row)
                        if row_tuple:
                            link = row_tuple[1]
                            if link not in sampled:
                                sampled[link] = keep_thread(link, rate)
                            if sampled[link]:
                                buf.append(row_tuple)
                    con.executemany("INSERT OR IGNORE INTO comments VALUES (?,?,?,?,?,?,?)", buf)
                    kept_total = con.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
                    set_state(con, "ingest_parquet_consumed", str(consumed)); con.commit()
                    elapsed = max(time.time() - t0, 1e-6)
                    speed = _session_rate(consumed, session_start, elapsed)
                    remaining = max(0, shard_rows - shard_scanned)
                    eta = remaining / max(speed, 1)
                    if advertised_size and consumed > advertised_size and not mismatch_warned:
                        print("WARNING: dataset metadata count looks stale")
                        mismatch_warned = True
                    print(f"  scanned {consumed:,} (shard {shard_scanned:,}/{shard_rows:,}; {speed:,.0f} rows/s; shard ETA ~{eta/60:,.1f} min) | kept {kept_total:,}")
                    if target_rows and kept_total >= target_rows:
                        capped = True
                    if capped:
                        break
                if not capped:
                    completed.add(filename)
                    set_state(con, "ingest_parquet_completed", json.dumps(sorted(completed)))
                    con.commit()
                else:
                    break
            finally:
                try:
                    Path(local_path).unlink()
                except FileNotFoundError:
                    pass
            if next_index < len(pending):
                futures.append(pool.submit(download, pending[next_index])); next_index += 1
    con.execute("CREATE INDEX IF NOT EXISTS idx_comments_link ON comments(link_id)")
    all_done = all(filename in completed for filename in shards)
    if all_done:
        set_state(con, "ingest_complete", "1")
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
    threads = con.execute("SELECT COUNT(DISTINCT link_id) FROM comments").fetchone()[0]
    print(f"Done: {n:,} comments across {threads:,} threads stored in {settings.db_path}")
    print_ingest_state(con)
    return True


def _session_rate(consumed: int, session_start: int, elapsed: float) -> float:
    return max(0, consumed - session_start) / max(elapsed, 1)


def print_ingest_state(con) -> None:
    complete = get_state(con, "ingest_complete") == "1"
    scanned = max(int(get_state(con, key, "0") or 0) for key in ("ingest_consumed", "ingest_parquet_consumed"))
    sample_rate = get_state(con, "ingest_sample_rate", "unknown")
    comments = con.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
    print("Ingest state:")
    print(f"  complete: {'yes' if complete else 'no'}")
    print(f"  rows scanned: {scanned:,}")
    print(f"  sample rate: {sample_rate}")
    print(f"  comments stored: {comments:,}")


def _flush(con, buf, consumed, seen) -> int:
    con.executemany("INSERT OR IGNORE INTO comments VALUES (?,?,?,?,?,?,?)", buf)
    consumed += seen
    set_state(con, "ingest_consumed", str(consumed))
    con.commit()
    return consumed
