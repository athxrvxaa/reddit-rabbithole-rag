"""Evaluation: retrieval ablation (Recall@k / MRR), abstain calibration, generation checks."""
from __future__ import annotations

import json
import random
import re
import sqlite3
import statistics
import time
from pathlib import Path

from .generation import GeminiClient, GeminiQuotaError
from .retrieval import MODES, Retriever


# --------------------------------------------------------------------------- synthetic queries
QGEN_PROMPT = """Below is a Q&A pair from a Reddit AMA. Write ONE natural question that a curious person might type into a search box, which this pair would answer well.
- Do not copy distinctive phrases or names from the pair; paraphrase.
- Do not mention Reddit, AMA, or usernames.
- At most 25 words. Output only the question.

Question: {q}
Answer: {a}"""


def make_synthetic_queries(artifacts_dir: Path, llm: GeminiClient, n: int, out_path: Path, seed: int = 0, sleep: float = 6.5) -> None:
    con = sqlite3.connect(str(Path(artifacts_dir) / "corpus.db"))
    con.row_factory = sqlite3.Row
    ids = [r[0] for r in con.execute("SELECT idx FROM docs WHERE LENGTH(answer) >= 250")]
    rng = random.Random(seed)
    rng.shuffle(ids)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_jsonl(out_path) if out_path.exists() else []
    done_ids = {int(row["idx"]) for row in existing if "idx" in row}
    done = len(existing)
    if len(ids) < 200:
        print(f"Warning: only {len(ids)} eligible source queries exist (recommended: at least 200).")
    quota_stopped = False
    with open(out_path, "a", encoding="utf-8") as f:
        for idx in ids:
            if done >= n:
                break
            if idx in done_ids:
                continue
            r = con.execute("SELECT * FROM docs WHERE idx=?", (idx,)).fetchone()
            try:
                q = llm.generate(QGEN_PROMPT.format(q=r["question"][:400], a=r["answer"][:900]), temperature=0.7).strip()
            except GeminiQuotaError as e:
                print(f"Quota reached; saved progress and stopped: {e}")
                quota_stopped = True
                break
            except Exception as e:  # keep going on transient failures
                print("  skip:", e)
                continue
            q = q.strip().strip('"').splitlines()[0] if q else ""
            if len(q) < 10:
                continue
            f.write(json.dumps({"query": q, "idx": int(idx), "thread_id": r["thread_id"]}) + "\n")
            f.flush()
            done += 1
            done_ids.add(idx)
            if done % 25 == 0:
                print(f"  {done}/{n} saved")
            time.sleep(sleep)
    print(f"{done}/{n} saved" + (" (stopped at quota)" if quota_stopped else ""))
    print(f"Queries saved -> {out_path}")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


# --------------------------------------------------------------------------- retrieval ablation
def eval_retrieval(retriever: Retriever, queries: list[dict], modes: list[str] | None = None, ks=(1, 5, 10, 20)) -> dict:
    modes = [m for m in (modes or list(MODES)) if m != "hybrid_rerank" or retriever.reranker is not None]
    kmax = max(ks)
    results = {}
    for mode in modes:
        hit_pair = {k: 0 for k in ks}
        hit_thread = {k: 0 for k in ks}
        rr = []
        lat = []
        for q in queries:
            res = retriever.search(q["query"], top_k=kmax, mode=mode, per_thread=0)
            lat.append(res.timings["total_ms"])
            pair_rank = thread_rank = None
            for rank, h in enumerate(res.hits, 1):
                if pair_rank is None and h.idx == q["idx"]:
                    pair_rank = rank
                if thread_rank is None and h.doc["thread_id"] == q["thread_id"]:
                    thread_rank = rank
            rr.append(1.0 / pair_rank if pair_rank else 0.0)
            for k in ks:
                hit_pair[k] += bool(pair_rank and pair_rank <= k)
                hit_thread[k] += bool(thread_rank and thread_rank <= k)
        n = len(queries)
        results[mode] = {
            "n": n,
            **{f"recall@{k}": hit_pair[k] / n for k in ks},
            **{f"thread_recall@{k}": hit_thread[k] / n for k in ks},
            f"mrr@{kmax}": statistics.mean(rr),
            "latency_ms_mean": statistics.mean(lat),
            "latency_ms_p95": sorted(lat)[int(0.95 * (len(lat) - 1))],
        }
    return results


def results_markdown(results: dict, ks=(1, 5, 10, 20)) -> str:
    kmax = max(ks)
    head = "| Mode | " + " | ".join(f"R@{k}" for k in ks) + f" | MRR@{kmax} | Thread R@5 | Latency ms (mean / p95) |"
    sep = "|" + "---|" * (len(ks) + 4)
    rows = [head, sep]
    for mode, m in results.items():
        cells = " | ".join(f"{m[f'recall@{k}']:.3f}" for k in ks)
        rows.append(
            f"| {mode} | {cells} | {m[f'mrr@{kmax}']:.3f} | {m.get('thread_recall@5', 0):.3f} | "
            f"{m['latency_ms_mean']:.0f} / {m['latency_ms_p95']:.0f} |"
        )
    return "\n".join(rows)


# --------------------------------------------------------------------------- abstain calibration
def calibrate(retriever: Retriever, in_domain: list[str], ood: list[str]) -> dict:
    if retriever.reranker is None:
        raise SystemExit("Calibration needs the cross-encoder reranker.")

    def top_scores(qs):
        out = []
        for q in qs:
            res = retriever.search(q, top_k=1, mode="hybrid_rerank", per_thread=0)
            out.append(res.hits[0].rerank if res.hits and res.hits[0].rerank is not None else -99.0)
        return out

    a, b = top_scores(in_domain), top_scores(ood)
    thresholds = sorted(set(a + b))
    table = [{"threshold": t, "in_domain_pass_rate": sum(x >= t for x in a) / len(a),
              "ood_abstain_rate": sum(x < t for x in b) / len(b)} for t in thresholds]
    eligible = [row for row in table if row["in_domain_pass_rate"] >= .95]
    suggested = max(eligible, key=lambda row: row["threshold"]) if eligible else None
    pct = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]  # noqa: E731
    return {
        "in_domain_top_score": {"p10": pct(a, .1), "median": pct(a, .5), "p90": pct(a, .9)},
        "ood_top_score": {"p10": pct(b, .1), "median": pct(b, .5), "p90": pct(b, .9)},
        "reranker": getattr(retriever.reranker, "model_name", retriever.settings.reranker_model),
        "threshold_table": table,
        "suggested_threshold": suggested["threshold"] if suggested else None,
        "suggested_ood_abstain_rate": suggested["ood_abstain_rate"] if suggested else None,
        "n_in": len(a), "n_ood": len(b),
    }


# --------------------------------------------------------------------------- generation checks
JUDGE_PROMPT = """You are a strict fact-checker. Given SOURCES and an ANSWER, count the factual claims in the ANSWER and how many are directly supported by the SOURCES.
Reply with JSON only: {{"total": <int>, "supported": <int>}}

SOURCES:
{sources}

ANSWER:
{answer}"""


def judge_faithfulness(judge: GeminiClient, answer: str, hits) -> float | None:
    src = "\n\n".join(f"[{i}] {h.doc['question'][:300]}\n{h.doc['answer'][:1200]}" for i, h in enumerate(hits, 1))
    try:
        txt = judge.generate(JUDGE_PROMPT.format(sources=src, answer=answer), temperature=0.0)
        m = re.search(r"\{.*?\}", txt, re.S)
        d = json.loads(m.group(0))
        return d["supported"] / d["total"] if d["total"] else None
    except Exception:
        return None


def eval_generation(pipeline, in_domain: list[str], ood: list[str], judge: GeminiClient | None = None, sleep: float = 0.5, limit: int | None = None) -> dict:
    quota_error = None
    def run(qs):
        nonlocal quota_error
        rows = []
        if quota_error:
            return rows
        for q in qs[:limit] if limit is not None else qs:
            try:
                r = pipeline.ask(q)
            except GeminiQuotaError as e:
                quota_error = str(e)
                break
            f = judge_faithfulness(judge, r.answer, r.search.hits) if judge and not r.citations.abstained else None
            rows.append(
                {"query": q, "abstained": r.citations.abstained, "invalid_cites": len(r.citations.invalid),
                 "uncited_rate": r.citations.uncited_rate, "truncated": r.truncated, "faithfulness": f,
                 "latency_ms": r.timings["retrieve_ms"] + r.timings["generate_ms"]}
            )
            time.sleep(sleep)
        if quota_error:
            return rows
        return rows

    a, b = run(in_domain), run(ood)
    mean = lambda xs: (statistics.mean(xs) if xs else None)  # noqa: E731
    answered = [r for r in a if not r["abstained"]]
    return {
        "in_domain_abstain_rate": mean([r["abstained"] for r in a]),
        "ood_abstain_rate": mean([r["abstained"] for r in b]),
        "answers_with_invalid_citation": mean([r["invalid_cites"] > 0 for r in answered]),
        "mean_uncited_sentence_rate": mean([r["uncited_rate"] for r in answered]),
        "truncation_rate": mean([r["truncated"] for r in a]),
        "mean_faithfulness_judge": mean([r["faithfulness"] for r in answered if r["faithfulness"] is not None]),
        "mean_latency_ms": mean([r["latency_ms"] for r in a]),
        "completed": len(a) + len(b), "quota_error": quota_error,
        "rows": {"in_domain": a, "ood": b},
    }
