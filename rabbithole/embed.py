"""Embedding + reranking backends. Heavy imports are lazy so tests/CLI start fast."""
from __future__ import annotations

import re
import zlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def encode_docs(self, texts: list[str], batch_size: int = 128, show_progress: bool = False) -> np.ndarray: ...
    def encode_query(self, text: str) -> np.ndarray: ...


class STEmbedder:
    """sentence-transformers embedder (default: BAAI/bge-small-en-v1.5)."""

    def __init__(self, model_name: str, query_prefix: str = "", max_seq_length: int = 320, device: str | None = None, fp16: bool = True):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_seq_length
        if fp16 and str(self.model.device).startswith("cuda"):
            self.model.half()
        self.query_prefix = query_prefix
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.model_name = model_name

    def encode_docs(self, texts, batch_size=128, show_progress=False):
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=show_progress,
        ).astype("float32")

    def encode_query(self, text):
        v = self.model.encode([self.query_prefix + text], normalize_embeddings=True, convert_to_numpy=True)[0]
        return v.astype("float32")


class HashEmbedder:
    """Dependency-free bag-of-words hashing embedder. TESTING / SMOKE RUNS ONLY (weak semantics)."""

    model_name = "hash-test"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(tok.encode()) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_docs(self, texts, batch_size=128, show_progress=False):
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), "float32")

    def encode_query(self, text):
        return self._vec(text)


class CrossEncoderReranker:
    def __init__(self, model_name: str, max_length: int = 512, batch_size: int = 16):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=max_length)
        self.batch_size = batch_size
        self.model_name = model_name

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, "float32")
        return np.asarray(
            self.model.predict([[query, t] for t in texts], batch_size=self.batch_size, show_progress_bar=False),
            dtype="float32",
        )


def make_embedder(kind: str, settings) -> Embedder:
    if kind == "hash":
        return HashEmbedder()
    return STEmbedder(
        settings.embed_model, settings.query_prefix, settings.embed_max_seq_length
    )
