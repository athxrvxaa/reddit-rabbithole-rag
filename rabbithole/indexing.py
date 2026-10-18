"""Stage 4: build search artifacts (dense FAISS + BM25 + doc store) from the `pairs` table."""
from __future__ import annotations

import json
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .categories import CATEGORIES
from .config import Settings
from .db import connect
from .embed import Embedder
from .text import truncate_sentences

DOC_COLUMNS = (
    "idx INTEGER PRIMARY KEY, thread_id TEXT, title TEXT, context TEXT, category TEXT, question TEXT, answer TEXT, "
    "question_id TEXT, answer_id TEXT, answer_score INTEGER, source_url TEXT, op_author TEXT"
)


def embed_text(title: str, question: str, answer: str) -> str:
    """What the dense model sees. Question is capped so long quoted questions can't push the answer out."""
    return f"{truncate_sentences(title, 150)}\nQ: {truncate_sentences(question, 400)}\nA: {answer}"


def bm25_text(title: str, question: str, answer: str) -> str:
    return f"{title}\n{question}\n{answer}"


def build_faiss(emb: np.ndarray, kind: str = "auto"):
    import faiss

    n, d = emb.shape
    if kind == "auto":
        kind = "flat" if n <= 200_000 else "hnsw"
    if kind == "flat":
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 100
    step = 50_000
    for i in range(0, n, step):
        index.add(np.ascontiguousarray(emb[i : i + step], dtype="float32"))
    return index, kind


def make_stemmer():
    import Stemmer

    return Stemmer.Stemmer("english")


def build_index(settings: Settings, embedder: Embedder, limit: int | None = None, faiss_kind: str = "auto") -> None:
    import bm25s
    import faiss

    out = settings.artifacts_dir
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    src = connect(settings.db_path)
    q = "SELECT * FROM pairs ORDER BY pair_id" + (f" LIMIT {int(limit)}" if limit else "")
    n = src.execute(f"SELECT COUNT(*) FROM ({q})").fetchone()[0]
    if n == 0:
        raise SystemExit("No pairs found. Run extract (or import-v1) first.")
    print(f"Building index for {n:,} documents (dim={embedder.dim})")
    year_values = [int(r[0]) for r in src.execute(
        f"SELECT DISTINCT strftime('%Y', created_utc, 'unixepoch') FROM ({q}) WHERE created_utc > 0"
    ) if r[0]]

    # 1) doc store + embeddings (batched, memory-mapped)
    store = sqlite3.connect(str(out / "corpus.db"))
    store.execute(f"CREATE TABLE docs({DOC_COLUMNS})")
    emb = np.lib.format.open_memmap(str(out / "embeddings.npy"), mode="w+", dtype="float16", shape=(n, embedder.dim))
    cat_to_id = {c: i for i, c in enumerate(CATEGORIES)}
    cat_ids = np.zeros(n, dtype="uint8")
    thread_codes = np.zeros(n, dtype="int32")
    thread_map: dict[str, int] = {}
    bm25_docs: list[str] = []

    B = 2048
    t0 = time.time()
    cur = src.execute(q)
    i = 0
    while True:
        rows = cur.fetchmany(B)
        if not rows:
            break
        texts = [embed_text(r["title"] or "", r["question"], r["answer"]) for r in rows]
        emb[i : i + len(rows)] = embedder.encode_docs(texts, batch_size=settings.embed_batch_size).astype("float16")
        docs = []
        for j, r in enumerate(rows):
            idx = i + j
            cat_ids[idx] = cat_to_id.get(r["category"], cat_to_id["other"])
            thread_codes[idx] = thread_map.setdefault(r["thread_id"], len(thread_map))
            bm25_docs.append(bm25_text(r["title"] or "", r["question"], r["answer"]))
            docs.append(
                (idx, r["thread_id"], r["title"], r["context"], r["category"], r["question"], r["answer"],
                 r["question_id"], r["answer_id"], r["answer_score"], r["source_url"], r["op_author"])
            )
        store.executemany("INSERT INTO docs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", docs)
        i += len(rows)
        if (i // B) % 10 == 0 or i == n:
            rate = i / max(time.time() - t0, 1e-6)
            print(f"  embedded {i:,}/{n:,} ({rate:,.0f} docs/s)")
    store.execute("CREATE INDEX idx_docs_thread ON docs(thread_id)")
    store.commit()
    store.close()
    emb.flush()
    np.save(out / "cat_ids.npy", cat_ids)
    np.save(out / "thread_codes.npy", thread_codes)

    # 2) FAISS
    print("Building FAISS index...")
    index, kind = build_faiss(np.load(out / "embeddings.npy", mmap_mode="r"), faiss_kind)
    faiss.write_index(index, str(out / "faiss.index"))

    # 3) BM25 (bm25s: sparse, fast, saved without pickle)
    print("Building BM25 index...")
    stemmer = make_stemmer()
    tokens = bm25s.tokenize(bm25_docs, stopwords="en", stemmer=stemmer, show_progress=False)
    bm = bm25s.BM25()
    bm.index(tokens, show_progress=False)
    bm.save(str(out / "bm25"))

    manifest = {
        "version": 2,
        "n_docs": n,
        "dim": embedder.dim,
        "embed_model": getattr(embedder, "model_name", "unknown"),
        "query_prefix": settings.query_prefix if getattr(embedder, "model_name", "") != "hash-test" else "",
        "faiss_kind": kind,
        "categories": CATEGORIES,
        "n_threads": len(thread_map),
        "min_year": min(year_values) if year_values else None,
        "max_year": max(year_values) if year_values else None,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    size = sum(f.stat().st_size for f in Path(out).rglob("*") if f.is_file()) / 1e6
    print(f"Done in {time.time() - t0:.0f}s. Artifacts: {out} ({size:,.0f} MB)")
