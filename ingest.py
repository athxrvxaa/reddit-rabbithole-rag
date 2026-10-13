"""Fetch r/IAmA comments, extract OP replies to commenters, enrich, and clean.

Example: python ingest.py --max-comments 1000000
The source is a large historical stream; raise the cap for broader coverage.
"""
import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests
from datasets import load_dataset

from retrieval import clean_text, load_corpus

BAD = {"[deleted]", "[removed]", "AutoModerator", "reddit"}


def fetch_comments(max_comments, offset=0):
    threads = defaultdict(list)
    total = 0
    stream = load_dataset("HuggingFaceGECLM/REDDIT_comments", split="IAmA", streaming=True)
    if offset:
        stream = stream.skip(offset)
    for row in stream:
        body = clean_text(row.get("body"))
        author = row.get("author") or ""
        if not body or body in BAD or author in BAD:
            continue
        link = str(row.get("link_id", ""))
        if not link:
            continue
        threads[link].append({
            "id": str(row.get("id", "")), "author": author, "body": body,
            "parent_id": str(row.get("parent_id", "")),
            "score": int(row.get("score") or 0),
        })
        total += 1
        if total >= max_comments:
            break
    return threads


def extract_pairs(threads, min_answer_score):
    pairs = []
    for thread, comments in threads.items():
        by_id = {c["id"]: c for c in comments}
        top = [c for c in comments if c["parent_id"] == thread and c["author"] not in BAD]
        counts = Counter(c["author"] for c in top)
        if not counts:
            continue
        op = counts.most_common(1)[0][0]
        # Require repeated top-level participation to reduce false OP assignments.
        if counts[op] < 2:
            continue
        for answer in comments:
            if answer["author"] != op or not answer["parent_id"].startswith("t1_"):
                continue
            parent = by_id.get(answer["parent_id"][3:])
            if not parent or parent["author"] in BAD or parent["author"] == op:
                continue
            question, response = parent["body"], answer["body"]
            if len(question) < 20 or len(question) > 1000 or len(response) < 50:
                continue
            if "?" not in question and not question.lower().startswith(("what", "how", "why", "when", "where", "who", "tell me", "describe", "explain")):
                continue
            if answer["score"] < min_answer_score:
                continue
            pairs.append({
                "thread_id": thread, "question_id": parent["id"], "answer_id": answer["id"],
                "op_author": op, "question": question, "answer": response,
                "question_score": parent["score"], "answer_score": answer["score"],
            })
    return pairs


def enrich(pairs):
    ids = list(dict.fromkeys(p["thread_id"].removeprefix("t3_") for p in pairs))
    metadata = {}
    session = requests.Session()
    for start in range(0, len(ids), 40):
        batch = ids[start:start + 40]
        try:
            response = session.get("https://arctic-shift.photon-reddit.com/api/posts/ids", params={"ids": ",".join(batch)}, timeout=(5, 30))
            response.raise_for_status()
            for post in response.json().get("data", []):
                metadata["t3_" + post["id"]] = post
        except requests.RequestException as exc:
            print(f"Metadata batch {start // 40 + 1} failed: {exc}")
        time.sleep(0.25)
    enriched = []
    for p in pairs:
        post = metadata.get(p["thread_id"])
        if not post or len(post.get("title", "")) < 10:
            continue
        title = clean_text(post.get("title"))
        enriched.append({**p, "ama_title": title, "ama_context": clean_text(post.get("selftext", ""))[:1000],
                         "category": "unspecified", "source_url": f"https://reddit.com/r/IAmA/comments/{p['thread_id'].removeprefix('t3_')}",
                         "answer_length": len(p["answer"])})
    return enriched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-comments", type=int, default=1_000_000)
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip this many records in the source stream to collect a later segment")
    parser.add_argument("--min-answer-score", type=int, default=0)
    parser.add_argument("--output", default="data/ama_qa_enriched.jsonl")
    parser.add_argument("--base-corpus", help="Optional existing JSONL corpus to merge and deduplicate")
    args = parser.parse_args()
    threads = fetch_comments(args.max_comments, args.offset)
    pairs = extract_pairs(threads, args.min_answer_score)
    print(f"Extracted {len(pairs):,} candidate pairs from {len(threads):,} threads")
    rows = enrich(pairs)
    # Merge into a separate temporary file first, then apply app cleanup and dedup.
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".raw.jsonl")
    with temp.open("w", encoding="utf-8") as f:
        if args.base_corpus:
            with Path(args.base_corpus).open(encoding="utf-8") as base:
                for line in base:
                    f.write(line)
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    cleaned = load_corpus(temp)
    candidate = output.with_suffix(output.suffix + ".tmp")
    with candidate.open("w", encoding="utf-8") as f:
        for row in cleaned:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.unlink()
    candidate.replace(output)
    thread_count = len({row["thread_id"] for row in cleaned})
    print(f"Saved {len(cleaned):,} cleaned pairs across {thread_count:,} threads to {output}")


if __name__ == "__main__":
    main()
