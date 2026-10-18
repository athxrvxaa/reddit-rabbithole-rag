"""Stage 2: fetch the real submission (title + OP author) for each thread from Arctic Shift.

Knowing the true OP is what fixes v1's noisy pairs (v1 guessed the OP from comment counts).
"""
from __future__ import annotations

import time

import requests

from .config import Settings
from .db import connect
from .text import clean_comment

UA = {"User-Agent": "reddit-rabbithole-rag/2.0 (research project)"}


def _fetch_batch(base: str, ids: list[str], session: requests.Session, retries: int = 5) -> list[dict] | None:
    url = f"{base}/api/posts/ids"
    for attempt in range(retries):
        try:
            r = session.get(url, params={"ids": ",".join(ids)}, headers=UA, timeout=30)
            if r.status_code == 200:
                data = r.json().get("data")
                return data or []
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 3 * (attempt + 1) ** 2))
                continue
            return None
        except (requests.RequestException, ValueError):
            time.sleep(min(60, 3 * (attempt + 1) ** 2))
    return None


def fetch_meta(settings: Settings, min_comments: int | None = None, batch_size: int = 50, sleep: float = 0.3,
               session: requests.Session | None = None) -> None:
    from .extract import ExtractConfig

    if min_comments is None:
        min_comments = ExtractConfig().min_thread_comments
    con = connect(settings.db_path)
    rows = con.execute(
        """
        SELECT link_id FROM comments GROUP BY link_id HAVING COUNT(*) >= ?
        AND link_id NOT IN (SELECT thread_id FROM threads)
        """,
        (min_comments,),
    ).fetchall()
    todo = [r[0] for r in rows]
    print(f"{len(todo):,} threads (>= {min_comments} comments) need metadata")
    if not todo:
        return
    session = session or requests.Session()
    found = missing = 0
    failed_batches: list[list[str]] = []

    def save_batch(batch: list[str], posts: list[dict]) -> tuple[int, int]:
        got: set[str] = set()
        out = []
        for p in posts:
            pid = str(p.get("id") or "")
            if not pid:
                continue
            got.add(pid)
            selftext = p.get("selftext") or ""
            if selftext in ("[removed]", "[deleted]"):
                selftext = ""
            out.append((pid, 1, p.get("author") or "", clean_comment(p.get("title") or "", "marker"),
                        clean_comment(selftext, "marker")[:800], int(p.get("score") or 0),
                        int(p.get("created_utc") or 0), int(p.get("num_comments") or 0)))
        absent = [tid for tid in batch if tid not in got]
        out.extend((tid, 0, "", "", "", 0, 0, 0) for tid in absent)
        con.executemany("INSERT OR REPLACE INTO threads VALUES (?,?,?,?,?,?,?,?)", out)
        con.commit()
        return len(got), len(absent)

    for bi in range(0, len(todo), batch_size):
        batch = todo[bi : bi + batch_size]
        posts = _fetch_batch(settings.arctic_base, batch, session)
        if posts is None:
            failed_batches.append(batch)
        else:
            got, absent = save_batch(batch, posts)
            found += got; missing += absent
        if (bi // batch_size) % 20 == 0:
            print(f"  {bi + len(batch):,}/{len(todo):,} done (found {found:,}, missing {missing:,}, failed batches {len(failed_batches):,})")
        time.sleep(sleep)
    for retry_number in range(1, 4):
        if not failed_batches:
            break
        pending, failed_batches = failed_batches, []
        wait = min(60, 2 ** (retry_number + 1))
        print(f"Retrying {len(pending):,} failed batches (pass {retry_number}/3) after {wait}s")
        time.sleep(wait)
        for batch in pending:
            posts = _fetch_batch(settings.arctic_base, batch, session)
            if posts is None:
                failed_batches.append(batch)
            else:
                got, absent = save_batch(batch, posts)
                found += got; missing += absent
    failed_count = sum(len(batch) for batch in failed_batches)
    print(f"Finished. found={found:,} missing={missing:,} failed={failed_count:,}")
