"""Evaluate retrieval stages against manually labeled relevant thread IDs.

Judgment JSONL schema: {"question": "...", "relevant_thread_ids": ["t3_..."]}
Only labeled judgments are accepted; this script does not invent relevance labels.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from retrieval import Retriever, load_corpus, tokenize
from rank_bm25 import BM25Okapi


def metrics(ranked_ids, relevant, k=10):
    ranked = ranked_ids[:k]
    hits = [i + 1 for i, item in enumerate(ranked) if item in relevant]
    recall = len(set(ranked) & relevant) / len(relevant) if relevant else 0.0
    reciprocal_rank = 1 / hits[0] if hits else 0.0
    return recall, reciprocal_rank


def unique_threads(rows):
    result, seen = [], set()
    for thread_id in rows:
        if thread_id not in seen:
            result.append(thread_id)
            seen.add(thread_id)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/ama_qa_enriched.jsonl")
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    judgments = [json.loads(line) for line in Path(args.judgments).open(encoding="utf-8") if line.strip()]
    if not judgments:
        raise SystemExit("Judgment file is empty")
    corpus = load_corpus(args.corpus)
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    retriever = Retriever(corpus, model)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    bm25 = BM25Okapi([tokenize(f"{d['question']} {d['answer']}") for d in corpus])
    totals = {name: [0.0, 0.0] for name in ("BM25", "Dense", "Hybrid RRF", "Hybrid + rerank")}
    latency = {name: [] for name in totals}
    for row in judgments:
        query = row["question"]
        relevant = set(map(str, row["relevant_thread_ids"]))
        start = time.perf_counter()
        bm25_order = np.argsort(bm25.get_scores(tokenize(query)))[::-1][:args.k]
        latency["BM25"].append(time.perf_counter() - start)
        start = time.perf_counter()
        query_vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        dense_order = np.argsort(retriever.embeddings @ query_vec)[::-1][:args.k]
        latency["Dense"].append(time.perf_counter() - start)
        start = time.perf_counter()
        hybrid = retriever.search(query, candidate_k=max(args.k, 40))
        latency["Hybrid RRF"].append(time.perf_counter() - start)
        start = time.perf_counter()
        scores = reranker.predict([[query, f"{d['question']} {d['answer']}"] for d in hybrid])
        reranked = [d for _, d in sorted(zip(scores, hybrid), key=lambda pair: pair[0], reverse=True)]
        latency["Hybrid + rerank"].append(time.perf_counter() - start)
        rankings = {
            "BM25": unique_threads([corpus[int(i)]["thread_id"] for i in bm25_order]),
            "Dense": unique_threads([corpus[int(i)]["thread_id"] for i in dense_order]),
            "Hybrid RRF": unique_threads([d["thread_id"] for d in hybrid]),
            "Hybrid + rerank": unique_threads([d["thread_id"] for d in reranked]),
        }
        for name, ids in rankings.items():
            rec, rr = metrics(ids, relevant, args.k)
            totals[name][0] += rec
            totals[name][1] += rr
    print(f"Judged questions: {len(judgments)}; k={args.k}")
    print("Stage                 Recall@k    MRR       Mean latency (s)")
    for name, (recall_sum, rr_sum) in totals.items():
        print(f"{name:21} {recall_sum / len(judgments):.3f}       {rr_sum / len(judgments):.3f}     {np.mean(latency[name]):.3f}")


if __name__ == "__main__":
    main()
