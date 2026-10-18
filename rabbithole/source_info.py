"""Peek at the HF source so you can pick a sensible sample size before a long ingest."""
from __future__ import annotations

from datetime import datetime, timezone
from itertools import islice

from .config import Settings
from .text import bare_id


def split_size(settings: Settings) -> int | None:
    """Row count of the configured split, if the dataset metadata exposes it."""
    try:
        from datasets import load_dataset_builder

        splits = load_dataset_builder(settings.hf_dataset).info.splits
        if splits and settings.hf_split in splits:
            return int(splits[settings.hf_split].num_examples)
    except Exception:
        return None
    return None


def suggest_rate(split_rows: int, target_rows: int) -> float:
    """Fraction of threads to keep so that ~target_rows comments are stored."""
    return min(1.0, max(1e-5, round(target_rows / split_rows, 5)))


def diagnose_rows(rows: list[dict]) -> dict:
    n = len(rows)
    links = [bare_id(str(r.get("link_id") or "")) for r in rows]
    times = [int(float(r.get("created_utc") or 0)) for r in rows]
    contiguous = sum(a == b for a, b in zip(links, links[1:])) / max(n - 1, 1)
    nondecreasing = sum(b >= a for a, b in zip(times, times[1:])) / max(n - 1, 1)
    if contiguous > 0.5:
        layout = "grouped_by_thread"
        advice = (
            "Rows come grouped by thread. --max-comments N keeps mostly whole threads but only covers the START of the "
            "split. --thread-sample-rate spreads the sample over the whole split (needs a full scan)."
        )
    elif nondecreasing > 0.9:
        layout = "time_ordered"
        advice = (
            "Rows are ordered by time with threads interleaved. --max-comments N would give one time slice full of "
            "half-cut threads. Use --thread-sample-rate (whole threads, whole timeline)."
        )
    else:
        layout = "mixed"
        advice = "Rows are not ordered by thread. Use --thread-sample-rate so sampled threads are complete."
    ts = [x for x in times if x > 0]
    fmt = lambda x: datetime.fromtimestamp(x, tz=timezone.utc).strftime("%Y-%m-%d")  # noqa: E731
    return {
        "rows_checked": n,
        "distinct_threads": len(set(links)),
        "same_thread_as_previous_row": round(contiguous, 3),
        "time_nondecreasing": round(nondecreasing, 3),
        "time_range": (fmt(min(ts)), fmt(max(ts))) if ts else None,
        "avg_body_chars": round(sum(len(r.get("body") or "") for r in rows) / max(n, 1)),
        "layout": layout,
        "advice": advice,
    }


def inspect_source(settings: Settings, n: int = 20_000, target_rows: int = 5_000_000) -> None:
    from datasets import load_dataset

    print(f"Reading the first {n:,} rows of {settings.hf_dataset} [{settings.hf_split}] ...")
    ds = load_dataset(settings.hf_dataset, split=settings.hf_split, streaming=True)
    rows = list(islice(ds, n))
    print("columns:", sorted(rows[0].keys()))
    d = diagnose_rows(rows)
    for k, v in d.items():
        print(f"  {k}: {v}")
    size = split_size(settings)
    if size:
        rate = suggest_rate(size, target_rows)
        print(f"\nsplit rows: {size:,}  ->  for ~{target_rows:,} kept rows use --thread-sample-rate {rate}")
        print(f"full scan at 50k rows/s would take about {size / 50_000 / 60:,.0f} min (the ingest log shows your real speed)")
    else:
        print("\nCould not read the split size from metadata; set SPLIT_ROWS by hand (see the HF dataset page).")
