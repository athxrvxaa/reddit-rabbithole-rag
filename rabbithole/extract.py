"""Stage 3: turn comment trees into clean (question -> OP answer) pairs."""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from itertools import groupby

from .categories import categorize
from .config import Settings
from .db import connect
from .text import (
    bare_id,
    clean_comment,
    content_hash,
    is_question,
    truncate_sentences,
)

_BOT_RX = re.compile(r"(?:Bot|_bot|-bot)$")
_KNOWN_BOTS = {"AutoModerator", "RemindMeBot", "TweetPoster", "imgurtranscriber", "[deleted]", "[removed]"}


@dataclass
class ExtractConfig:
    min_answer_chars: int = 80
    max_answer_chars: int = 4000
    min_question_chars: int = 15
    max_question_chars: int = 1500
    min_answer_score: int = 1
    max_pairs_per_thread: int = 200
    min_thread_comments: int = 10
    require_question: bool = True
    allow_heuristic_op: bool = False  # only used when Arctic Shift has no record of the thread
    followup_min_chars: int = 60  # drop short follow-ups like "why?" whose context is the OP's own comment


def is_bot(author: str) -> bool:
    return author in _KNOWN_BOTS or bool(_BOT_RX.search(author or ""))


def pick_op(comments: list[dict], meta_author: str | None, allow_heuristic: bool) -> tuple[str | None, str]:
    """True OP = submission author. Fallback = most active top-level commenter (flagged)."""
    if meta_author and meta_author not in ("[deleted]", "[removed]"):
        return meta_author, "submission"
    if not allow_heuristic:
        return None, "none"
    c = Counter()
    score = Counter()
    for r in comments:
        if r["parent_id"].startswith("t3_") and not is_bot(r["author"]):
            c[r["author"]] += 1
            score[r["author"]] += r["score"]
    if not c:
        return None, "none"
    best = max(c, key=lambda a: (c[a], score[a]))
    return (best, "heuristic") if c[best] >= 2 else (None, "none")


def make_pair(cfg: ExtractConfig, *, q_raw: str, a_raw: str, q_score: int, a_score: int, followup: bool) -> tuple[str, str] | None:
    """Clean + filter one candidate. Returns (question, answer) or None."""
    if a_score < cfg.min_answer_score:
        return None
    answer = clean_comment(a_raw, quotes="drop")
    if len(answer) < cfg.min_answer_chars:
        answer = clean_comment(a_raw, quotes="marker")
    question = clean_comment(q_raw, quotes="marker")
    if not (cfg.min_answer_chars <= len(answer) <= cfg.max_answer_chars):
        return None
    if not (cfg.min_question_chars <= len(question) <= cfg.max_question_chars):
        return None
    if cfg.require_question and not is_question(question):
        return None
    if followup and len(question) < cfg.followup_min_chars:
        return None
    return question, answer


def extract_thread(thread_id: str, comments: list[dict], meta: dict | None, cfg: ExtractConfig) -> list[dict]:
    op, op_source = pick_op(comments, (meta or {}).get("author"), cfg.allow_heuristic_op)
    if not op:
        return []
    by_id = {c["id"]: c for c in comments}
    title = (meta or {}).get("title") or ""
    context = (meta or {}).get("selftext") or ""
    category = categorize(title, context)
    out: list[dict] = []
    for r in comments:
        if r["author"] != op or r["parent_id"].startswith("t3_"):
            continue
        parent = by_id.get(bare_id(r["parent_id"]))
        if parent is None or parent["author"] == op or is_bot(parent["author"]):
            continue
        gp = by_id.get(bare_id(parent["parent_id"])) if not parent["parent_id"].startswith("t3_") else None
        followup = gp is not None and gp["author"] == op
        qa = make_pair(
            cfg, q_raw=parent["body"], a_raw=r["body"], q_score=parent["score"], a_score=r["score"], followup=followup
        )
        if not qa:
            continue
        q, a = qa
        out.append(
            dict(
                thread_id=thread_id, op_author=op, op_source=op_source, title=title, context=context,
                category=category, question=q, answer=a, question_id=parent["id"], answer_id=r["id"],
                question_score=parent["score"], answer_score=r["score"], created_utc=r["created_utc"],
                source_url=f"https://www.reddit.com/comments/{thread_id}/_/{r['id']}/",
                content_hash=content_hash(q, a),
            )
        )
    out.sort(key=lambda d: (d["answer_score"], len(d["answer"])), reverse=True)
    return out[: cfg.max_pairs_per_thread]


_INSERT = """INSERT OR IGNORE INTO pairs(thread_id, op_author, op_source, title, context, category, question, answer,
 question_id, answer_id, question_score, answer_score, created_utc, source_url, content_hash)
 VALUES (:thread_id,:op_author,:op_source,:title,:context,:category,:question,:answer,:question_id,:answer_id,
 :question_score,:answer_score,:created_utc,:source_url,:content_hash)"""


def run_extract(settings: Settings, cfg: ExtractConfig | None = None) -> None:
    cfg = cfg or ExtractConfig()
    con = connect(settings.db_path)
    con.execute("CREATE INDEX IF NOT EXISTS idx_comments_link ON comments(link_id)")
    meta = {
        r["thread_id"]: dict(r)
        for r in con.execute("SELECT * FROM threads WHERE found=1")
    }
    print(f"{len(meta):,} threads have submission metadata")
    con.execute("DELETE FROM pairs")
    con.execute("DELETE FROM sqlite_sequence WHERE name='pairs'")
    con.commit()

    cur = con.execute(
        "SELECT id, link_id, parent_id, author, body, score, created_utc FROM comments ORDER BY link_id"
    )
    stats = Counter()
    for link_id, rows in groupby(cur, key=lambda r: r["link_id"]):
        comments = [dict(r) for r in rows]
        stats["threads_seen"] += 1
        if len(comments) < cfg.min_thread_comments:
            stats["threads_too_small"] += 1
            continue
        pairs = extract_thread(link_id, comments, meta.get(link_id), cfg)
        if not pairs:
            stats["threads_no_pairs"] += 1
            continue
        stats["threads_with_pairs"] += 1
        con.executemany(_INSERT, pairs)
        if stats["threads_with_pairs"] % 500 == 0:
            con.commit()
            print(f"  {stats['threads_with_pairs']:,} threads -> pairs so far")
    con.commit()
    total = con.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
    print(json.dumps(dict(stats), indent=2))
    print(f"Extracted {total:,} unique Q&A pairs")


def import_v1_jsonl(settings: Settings, path: str, cfg: ExtractConfig | None = None) -> None:
    """Re-clean the v1 corpus (ama_qa_enriched.jsonl) into the v2 `pairs` table.

    Handy for a quick end-to-end smoke test without downloading the big dataset.
    """
    cfg = cfg or ExtractConfig()
    con = connect(settings.db_path)
    con.execute("DELETE FROM pairs")
    con.execute("DELETE FROM sqlite_sequence WHERE name='pairs'")
    kept = total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            total += 1
            qa = make_pair(
                cfg, q_raw=r.get("question", ""), a_raw=r.get("answer", ""),
                q_score=int(r.get("question_score") or 0), a_score=int(r.get("answer_score") or 0), followup=False,
            )
            if not qa:
                continue
            q, a = qa
            title = clean_comment(r.get("ama_title", ""), "marker")
            context = clean_comment(r.get("ama_context", "") or "", "marker")
            tid = bare_id(str(r.get("thread_id", "")))
            row = dict(
                thread_id=tid, op_author=r.get("op_author", ""), op_source="v1", title=title, context=context,
                category=categorize(title, context), question=q, answer=a, question_id=str(r.get("question_id", "")),
                answer_id=str(r.get("answer_id", "")), question_score=int(r.get("question_score") or 0),
                answer_score=int(r.get("answer_score") or 0), created_utc=int(r.get("created_utc") or 0),
                source_url=r.get("source_url") or f"https://www.reddit.com/comments/{tid}/",
                content_hash=content_hash(q, a),
            )
            con.execute(_INSERT, row)
            kept += 1
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
    print(f"v1 import: {total:,} rows read, {kept:,} passed filters, {n:,} unique after dedupe")


def corpus_stats(settings: Settings) -> None:
    con = connect(settings.db_path)
    n = con.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
    print(f"pairs: {n:,}")
    if not n:
        return
    print("threads:", con.execute("SELECT COUNT(DISTINCT thread_id) FROM pairs").fetchone()[0])
    from datetime import datetime, timezone
    years = [datetime.fromtimestamp(int(r[0]), tz=timezone.utc).year for r in con.execute("SELECT created_utc FROM pairs WHERE created_utc > 0")]
    if years:
        lo, hi = min(years), max(years)
        print(f"pair years: {lo}-{hi}")
        print("year histogram:")
        from collections import Counter
        for year, count in sorted(Counter(years).items()):
            print(f"  {year}: {count:,}")
        if hi - lo < 3:
            print("WARNING: this sample spans less than 3 years and is time-biased.")
    print("op_source:", dict(con.execute("SELECT op_source, COUNT(*) FROM pairs GROUP BY 1").fetchall()))
    print("categories:")
    for r in con.execute("SELECT category, COUNT(*) c FROM pairs GROUP BY 1 ORDER BY c DESC"):
        print(f"  {r[0]:22s} {r[1]:>8,}  ({100 * r[1] / n:.1f}%)")
    lens = sorted(r[0] for r in con.execute("SELECT LENGTH(answer) FROM pairs"))
    print("answer chars p10/p50/p90/max:", lens[n // 10], lens[n // 2], lens[9 * n // 10], lens[-1])
    top = con.execute("SELECT title, COUNT(*) c FROM pairs GROUP BY thread_id ORDER BY c DESC LIMIT 3").fetchall()
    print("biggest threads:", [(t[:40], c) for t, c in top])
