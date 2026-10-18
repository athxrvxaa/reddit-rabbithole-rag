"""RagPipeline: retrieve -> (abstain?) -> grounded generation -> citation check."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .config import Settings, get_settings
from .generation import (
    ABSTAIN_TEXT, SYSTEM_PROMPT, CitationReport, GeminiClient, build_prompt, check_citations,
)
from .retrieval import Retriever, SearchResult


@dataclass
class AskResult:
    query: str
    answer: str
    search: SearchResult
    citations: CitationReport | None = None
    truncated: bool = False
    timings: dict = field(default_factory=dict)


class RagPipeline:
    def __init__(self, retriever: Retriever, llm: GeminiClient | None):
        self.retriever = retriever
        self.llm = llm

    def retrieve(self, query: str, top_k: int = 4, category: str | None = None) -> SearchResult:
        return self.retriever.search(query, top_k=top_k, mode="hybrid_rerank", category=category)

    def stream_answer(self, query: str, search: SearchResult) -> Iterator[str]:
        """Yields the cumulative answer text."""
        if search.abstain or not search.hits:
            yield ABSTAIN_TEXT
            return
        if self.llm is None:
            raise RuntimeError("No LLM configured (set GEMINI_API_KEY).")
        prompt = build_prompt(query, search.hits)
        acc = ""
        for chunk in self.llm.stream(prompt, system=SYSTEM_PROMPT):
            acc += chunk
            yield acc

    def ask(self, query: str, top_k: int = 4, category: str | None = None) -> AskResult:
        t0 = time.perf_counter()
        search = self.retrieve(query, top_k, category)
        t1 = time.perf_counter()
        if search.abstain or not search.hits:
            answer, truncated = ABSTAIN_TEXT, False
        else:
            if self.llm is None:
                raise RuntimeError("No LLM configured (set GEMINI_API_KEY).")
            answer = self.llm.generate(build_prompt(query, search.hits), system=SYSTEM_PROMPT)
            truncated = self.llm.last_finish_reason == "MAX_TOKENS"
        t2 = time.perf_counter()
        rep = check_citations(answer, len(search.hits))
        return AskResult(
            query, answer, search, rep, truncated,
            {"retrieve_ms": (t1 - t0) * 1000, "generate_ms": (t2 - t1) * 1000, **search.timings},
        )


def load_retriever(settings: Settings | None = None, artifacts_dir: str | Path | None = None, embedder=None, reranker=None, use_reranker: bool = True) -> Retriever:
    """Load artifacts + models. Downloads artifacts from the HF dataset repo if missing locally."""
    import json
    import os

    from .artifacts import ensure_artifacts
    from .embed import CrossEncoderReranker, HashEmbedder, STEmbedder

    s = settings or get_settings()
    d = Path(artifacts_dir) if artifacts_dir else s.artifacts_dir
    ensure_artifacts(d, s.artifacts_repo)
    manifest = json.loads((d / "manifest.json").read_text())
    if embedder is None:
        if manifest["embed_model"] == "hash-test":
            embedder = HashEmbedder(manifest["dim"])
        else:
            embedder = STEmbedder(manifest["embed_model"], manifest.get("query_prefix", ""), s.embed_max_seq_length)
    if os.environ.get("RR_USE_RERANKER", "1") == "0":
        use_reranker = False  # e.g. very small hosts: hybrid only, no abstain gate
    if reranker is None and use_reranker:
        reranker = CrossEncoderReranker(s.reranker_model)
    return Retriever(d, embedder, reranker, s)


def build_pipeline(settings: Settings | None = None, **kw) -> RagPipeline:
    s = settings or get_settings()
    retriever = load_retriever(s, **kw)
    llm = GeminiClient.from_settings(s) if s.gemini_api_key else None
    return RagPipeline(retriever, llm)
