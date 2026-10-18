"""Hybrid retrieval: dense (FAISS) + BM25 -> RRF fusion -> cross-encoder rerank -> diversity -> abstain."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Settings, get_settings
from .embed import Embedder
from .store import CorpusStore
from .text import truncate_sentences

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    idx: int
    doc: dict
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank: float | None = None


@dataclass
class SearchResult:
    query: str
    mode: str
    hits: list[Hit]
    abstain: bool = False
    confidence: str = "none"
    timings: dict = field(default_factory=dict)


def rrf_fuse(rankings: list[list[int]], weights: list[float] | None = None, k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion. Scale-free, so BM25 and cosine scores never need normalizing."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + w / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def rerank_text(doc: dict) -> str:
    return (
        f"{doc['title']}\n{truncate_sentences(doc['question'], 400)}\n{truncate_sentences(doc['answer'], 1200)}"
    )


class Retriever:
    def __init__(self, artifacts_dir: str | Path, embedder: Embedder, reranker=None, settings: Settings | None = None):
        import bm25s
        import faiss

        from .indexing import make_stemmer

        self.settings = settings or get_settings()
        d = Path(artifacts_dir)
        self.manifest = json.loads((d / "manifest.json").read_text())
        self.embedder = embedder
        self.reranker = reranker
        self.index = faiss.read_index(str(d / "faiss.index"))
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = 128
        self.bm25 = bm25s.BM25.load(str(d / "bm25"), load_corpus=False, mmap=True)
        self._bm25s = bm25s
        self.stemmer = make_stemmer()
        self.store = CorpusStore(d / "corpus.db")
        self.cat_ids = np.load(d / "cat_ids.npy")
        self.thread_codes = np.load(d / "thread_codes.npy")
        self.categories: list[str] = self.manifest["categories"]
        self.cat_to_id = {c: i for i, c in enumerate(self.categories)}
        self.n = int(self.manifest["n_docs"])

    # ---- candidate generators -------------------------------------------------
    def _dense(self, query: str, k: int) -> list[tuple[int, float]]:
        v = self.embedder.encode_query(query).astype("float32")[None, :]
        scores, ids = self.index.search(v, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def _bm25(self, query: str, k: int) -> list[tuple[int, float]]:
        try:
            toks = self._bm25s.tokenize(query, stopwords="en", stemmer=self.stemmer, show_progress=False)
            res, sc = self.bm25.retrieve(toks, k=k, show_progress=False)
        except Exception:  # empty / all-stopword / all-OOV query
            return []
        return [(int(i), float(s)) for i, s in zip(res[0], sc[0]) if s > 0]

    # ---- main entry -----------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid_rerank",
        category: str | None = None,
        per_thread: int | None = None,
    ) -> SearchResult:
        assert mode in MODES, f"mode must be one of {MODES}"
        s = self.settings
        t_all = time.perf_counter()
        timings: dict[str, float] = {}
        cid = self.cat_to_id.get(category) if category and category != "all" else None
        k = min(s.pool * (5 if cid is not None else 1), self.n)

        dense: list[tuple[int, float]] = []
        bm: list[tuple[int, float]] = []
        t = time.perf_counter()
        if mode in ("dense", "hybrid", "hybrid_rerank"):
            dense = self._dense(query, k)
        timings["dense_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        if mode in ("bm25", "hybrid", "hybrid_rerank"):
            bm = self._bm25(query, k)
        timings["bm25_ms"] = (time.perf_counter() - t) * 1000

        dense_rank = {i: r + 1 for r, (i, _) in enumerate(dense)}
        bm_rank = {i: r + 1 for r, (i, _) in enumerate(bm)}
        if mode == "bm25":
            ranking = bm
        elif mode == "dense":
            ranking = dense
        else:
            ranking = rrf_fuse([[i for i, _ in dense], [i for i, _ in bm]],
                               weights=[s.rrf_dense_weight, s.rrf_bm25_weight], k=s.rrf_k)
        if cid is not None:
            ranking = [(i, sc) for i, sc in ranking if self.cat_ids[i] == cid]

        rerank_scores: dict[int, float] = {}
        mode_used = mode
        if mode == "hybrid_rerank":
            if self.reranker is None:
                mode_used = "hybrid"
            else:
                t = time.perf_counter()
                cands = ranking[: s.rerank_candidates]
                docs = self.store.get_many([i for i, _ in cands])
                texts = [rerank_text(docs[i]) for i, _ in cands]
                scores = self.reranker.score(query, texts)
                rerank_scores = {i: float(sc) for (i, _), sc in zip(cands, scores)}
                ranking = sorted(((i, rerank_scores[i]) for i, _ in cands), key=lambda x: x[1], reverse=True)
                timings["rerank_ms"] = (time.perf_counter() - t) * 1000

        # diversity: cap hits per AMA thread
        cap = s.per_thread if per_thread is None else per_thread
        picked: list[tuple[int, float]] = []
        per: dict[int, int] = {}
        for i, sc in ranking:
            if cap and cap > 0:
                code = int(self.thread_codes[i])
                if per.get(code, 0) >= cap:
                    continue
                per[code] = per.get(code, 0) + 1
            picked.append((i, sc))
            if len(picked) >= top_k:
                break

        docs = self.store.get_many([i for i, _ in picked])
        hits = [
            Hit(
                idx=i, doc=docs[i], score=sc, dense_rank=dense_rank.get(i), bm25_rank=bm_rank.get(i),
                rerank=rerank_scores.get(i),
            )
            for i, sc in picked
            if i in docs
        ]
        abstain = not hits
        confidence = "none" if not hits else "high"
        if mode_used == "hybrid_rerank" and hits and hits[0].rerank is not None:
            score = hits[0].rerank
            if s.abstain_threshold is not None and score < s.abstain_threshold:
                abstain, confidence = True, "none"
            elif s.lowconf_threshold is not None and score < s.lowconf_threshold:
                confidence = "low"
        timings["total_ms"] = (time.perf_counter() - t_all) * 1000
        return SearchResult(query=query, mode=mode_used, hits=hits, abstain=abstain, confidence=confidence, timings=timings)
