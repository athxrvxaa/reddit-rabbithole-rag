"""Central settings. Everything can be overridden with environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Tiny .env loader (no extra dependency). Existing env vars win."""
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float | None) -> float | None:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    if v.lower() in {"none", "off", "disable"}:
        return None
    return float(v)


@dataclass
class Settings:
    # paths
    data_dir: Path = field(default_factory=lambda: Path(_env("RR_DATA_DIR", str(ROOT / "data"))))
    artifacts_dir: Path = field(default_factory=lambda: Path(_env("RR_ARTIFACTS_DIR", str(ROOT / "artifacts"))))

    # source data
    hf_dataset: str = field(default_factory=lambda: _env("RR_HF_DATASET", "HuggingFaceGECLM/REDDIT_comments"))
    hf_split: str = field(default_factory=lambda: _env("RR_HF_SPLIT", "IAmA"))
    arctic_base: str = "https://arctic-shift.photon-reddit.com"

    # models
    embed_model: str = field(default_factory=lambda: _env("RR_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    query_prefix: str = field(
        default_factory=lambda: _env(
            "RR_QUERY_PREFIX", "Represent this sentence for searching relevant passages: "
        )
    )
    embed_max_seq_length: int = field(default_factory=lambda: int(_env("RR_EMBED_MAX_LEN", "320")))
    embed_batch_size: int = field(default_factory=lambda: int(_env("RR_EMBED_BATCH", "128")))
    reranker_model: str = field(
        default_factory=lambda: _env("RR_RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )

    # retrieval
    pool: int = field(default_factory=lambda: int(_env("RR_POOL", "100")))  # per-retriever candidates
    rerank_candidates: int = field(default_factory=lambda: int(_env("RR_RERANK_N", "30")))
    per_thread: int = field(default_factory=lambda: int(_env("RR_PER_THREAD", "2")))  # max hits per AMA
    rrf_k: int = 60
    rrf_dense_weight: float = field(default_factory=lambda: float(_env("RR_RRF_DENSE_WEIGHT", "1.0")))
    rrf_bm25_weight: float = field(default_factory=lambda: float(_env("RR_RRF_BM25_WEIGHT", "1.0")))
    # Reranker scores are model-specific; calibrate both tiers for each reranker.
    abstain_threshold: float | None = field(default_factory=lambda: _env_float("RR_ABSTAIN_THRESHOLD", -5.0))
    lowconf_threshold: float | None = field(default_factory=lambda: _env_float("RR_LOWCONF_THRESHOLD", -1.0))

    # generation
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-flash-latest"))
    gemini_fallbacks: list[str] = field(
        default_factory=lambda: [m.strip() for m in _env("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash,gemini-flash-lite-latest").split(",") if m.strip()]
    )
    max_output_tokens: int = field(default_factory=lambda: int(_env("RR_MAX_OUT", "2048")))

    # artifacts hosting
    artifacts_repo: str = field(default_factory=lambda: _env("ARTIFACTS_REPO", ""))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "rabbithole.db"


def get_settings() -> Settings:
    return Settings()
