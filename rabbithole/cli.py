"""python -m rabbithole <command>  (see README for the full pipeline)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import get_settings


def _p() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rabbithole", description="Reddit RabbitHole RAG v2")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="stream HF IAmA comments into SQLite (resumable)")
    s.add_argument("--max-comments", type=int, default=None, help="stop after SCANNING N rows (first-N slice)")
    s.add_argument("--thread-sample-rate", type=float, default=1.0,
                   help="keep this fraction of threads, whole threads only (e.g. 0.05 = 5%%)")
    s.add_argument("--target-rows", type=int, default=None, help="stop once N rows have been kept")

    s = sub.add_parser("inspect-source", help="peek at the HF split: layout, size, suggested sample rate")
    s.add_argument("-n", type=int, default=20_000)
    s.add_argument("--target-rows", type=int, default=5_000_000)

    s = sub.add_parser("meta", help="fetch submission author/title per thread (Arctic Shift)")
    s.add_argument("--min-comments", type=int, default=10)

    s = sub.add_parser("extract", help="build clean Q&A pairs from comment trees")
    s.add_argument("--allow-heuristic-op", action="store_true", help="guess OP when metadata is missing (noisier)")
    s.add_argument("--min-answer-score", type=int, default=1)
    s.add_argument("--max-pairs-per-thread", type=int, default=200)

    s = sub.add_parser("import-v1", help="re-clean the v1 jsonl into pairs (quick smoke test, no big download)")
    s.add_argument("path")

    sub.add_parser("stats", help="print corpus statistics")

    s = sub.add_parser("index", help="build FAISS + BM25 + doc store artifacts")
    s.add_argument("--embedder", choices=["st", "hash"], default="st", help="'hash' = fast fake embedder for smoke tests only")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--faiss", choices=["auto", "flat", "hnsw"], default="auto")

    s = sub.add_parser("search", help="retrieval only (no LLM) — good for debugging")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--mode", default="hybrid_rerank", choices=["bm25", "dense", "hybrid", "hybrid_rerank"])
    s.add_argument("--no-rerank", action="store_true")

    s = sub.add_parser("ask", help="full RAG answer in the terminal")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=4)

    s = sub.add_parser("eval-make", help="generate a synthetic retrieval eval set with Gemini")
    s.add_argument("-n", type=int, default=200)
    s.add_argument("--out", default="eval/synthetic_queries.jsonl")

    s = sub.add_parser("eval-retrieval", help="ablation table: bm25 vs dense vs hybrid vs hybrid+rerank")
    s.add_argument("--queries", default="eval/synthetic_queries.jsonl")
    s.add_argument("--no-rerank", action="store_true")

    s = sub.add_parser("calibrate", help="pick the abstain threshold from in-domain vs out-of-domain queries")
    s.add_argument("--seed", default="eval/seed_queries.json")

    s = sub.add_parser("eval-generation", help="citation/abstain/faithfulness checks on seed queries")
    s.add_argument("--seed", default="eval/seed_queries.json")
    s.add_argument("--judge", action="store_true", help="also run an LLM faithfulness judge")
    s.add_argument("--judge-model", default=None, help="use a different model than the generator for judging")

    sub.add_parser("gemini-models", help="list Gemini models your API key can call")

    s = sub.add_parser("upload", help="upload artifacts (or one file) to a HF dataset repo")
    s.add_argument("--repo", default=None)
    s.add_argument("--file", default=None, help="upload a single file instead of the artifacts folder")
    s.add_argument("--in-repo", default=None)
    s.add_argument("--private", action="store_true")

    s = sub.add_parser("run-all", help="meta -> extract -> index (after ingest)")
    s.add_argument("--embedder", choices=["st", "hash"], default="st")
    return p


def main(argv: list[str] | None = None) -> None:
    a = _p().parse_args(argv)
    s = get_settings()
    c = a.cmd

    if c == "ingest":
        from .ingest import ingest
        ingest(s, max_comments=a.max_comments, thread_sample_rate=a.thread_sample_rate, target_rows=a.target_rows)
    elif c == "inspect-source":
        from .source_info import inspect_source
        inspect_source(s, n=a.n, target_rows=a.target_rows)
    elif c == "meta":
        from .meta import fetch_meta
        fetch_meta(s, min_comments=a.min_comments)
    elif c == "extract":
        from .extract import ExtractConfig, run_extract
        run_extract(s, ExtractConfig(allow_heuristic_op=a.allow_heuristic_op, min_answer_score=a.min_answer_score,
                                     max_pairs_per_thread=a.max_pairs_per_thread))
    elif c == "import-v1":
        from .extract import import_v1_jsonl
        import_v1_jsonl(s, a.path)
    elif c == "stats":
        from .extract import corpus_stats
        corpus_stats(s)
    elif c == "index":
        from .embed import make_embedder
        from .indexing import build_index
        build_index(s, make_embedder(a.embedder, s), limit=a.limit, faiss_kind=a.faiss)
    elif c == "search":
        from .pipeline import load_retriever
        r = load_retriever(s, use_reranker=not a.no_rerank and a.mode == "hybrid_rerank")
        res = r.search(a.query, top_k=a.k, mode=a.mode)
        print(f"mode={res.mode} abstain={res.abstain} timings={ {k: round(v) for k, v in res.timings.items()} }")
        for i, h in enumerate(res.hits, 1):
            print(f"\n[{i}] score={h.score:.3f} rerank={h.rerank} dense#{h.dense_rank} bm25#{h.bm25_rank} | {h.doc['category']}")
            print(f"    {h.doc['title'][:90]}\n    Q: {h.doc['question'][:160]}\n    A: {h.doc['answer'][:240]}\n    {h.doc['source_url']}")
    elif c == "ask":
        from .pipeline import build_pipeline
        r = build_pipeline(s).ask(a.query, top_k=a.k)
        print(r.answer, "\n")
        print("citations:", r.citations)
        for i, h in enumerate(r.search.hits, 1):
            print(f"[{i}] {h.doc['title'][:80]}  {h.doc['source_url']}")
    elif c == "eval-make":
        from .eval import make_synthetic_queries
        from .generation import GeminiClient
        make_synthetic_queries(s.artifacts_dir, GeminiClient.from_settings(s), a.n, Path(a.out))
    elif c == "eval-retrieval":
        from .eval import eval_retrieval, load_jsonl, results_markdown
        from .pipeline import load_retriever
        qs = load_jsonl(Path(a.queries))
        res = eval_retrieval(load_retriever(s, use_reranker=not a.no_rerank), qs)
        md = results_markdown(res)
        print(md)
        out = Path("eval/results"); out.mkdir(parents=True, exist_ok=True)
        (out / "retrieval.json").write_text(json.dumps(res, indent=2))
        (out / "retrieval.md").write_text(md + f"\n\n_n = {len(qs)} synthetic queries_\n")
        print(f"\nSaved to {out}/retrieval.md")
    elif c == "calibrate":
        from .eval import calibrate
        from .pipeline import load_retriever
        seed = json.loads(Path(a.seed).read_text())
        out = calibrate(load_retriever(s), seed["in_domain"], seed["out_of_domain"])
        print(json.dumps(out, indent=2))
        print(f"\nSet RR_ABSTAIN_THRESHOLD={out['suggested_threshold']} (small seed set: sanity-check before trusting).")
    elif c == "eval-generation":
        from .eval import eval_generation
        from .generation import GeminiClient
        from .pipeline import build_pipeline
        seed = json.loads(Path(a.seed).read_text())
        judge = None
        if a.judge:
            judge = GeminiClient.from_settings(s)
            if a.judge_model:
                judge = GeminiClient(s.gemini_api_key, a.judge_model, s.gemini_fallbacks)
        out = eval_generation(build_pipeline(s), seed["in_domain"], seed["out_of_domain"], judge)
        rows = out.pop("rows")
        print(json.dumps(out, indent=2))
        Path("eval/results").mkdir(parents=True, exist_ok=True)
        Path("eval/results/generation.json").write_text(json.dumps({**out, "rows": rows}, indent=2))
    elif c == "gemini-models":
        import requests
        r = requests.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                         headers={"x-goog-api-key": s.gemini_api_key}, timeout=30)
        r.raise_for_status()
        for m in r.json().get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []) and "flash" in m["name"]:
                print(m["name"].replace("models/", ""))
    elif c == "upload":
        from .artifacts import upload_file, upload_folder
        repo = a.repo or s.artifacts_repo
        if not repo:
            sys.exit("Pass --repo or set ARTIFACTS_REPO.")
        if a.file:
            upload_file(Path(a.file), repo, a.in_repo or Path(a.file).name, a.private)
        else:
            upload_folder(s.artifacts_dir, repo, a.private)
    elif c == "run-all":
        from .embed import make_embedder
        from .extract import ExtractConfig, run_extract
        from .indexing import build_index
        from .meta import fetch_meta
        fetch_meta(s)
        run_extract(s, ExtractConfig())
        build_index(s, make_embedder(a.embedder, s))


if __name__ == "__main__":
    main()
