"""Corpus cleanup and persistent, pickle-free BM25/FAISS retrieval indexes."""
import hashlib
import html
import json
import math
import os
import re
import sqlite3
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem.snowball import EnglishStemmer

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
STEMMER = EnglishStemmer()
CACHE_SCHEMA = 1
BM25_K1 = 1.5
BM25_B = 0.75


def clean_text(value):
    return re.sub(r"\s+", " ", html.unescape(str(value or ""))).strip()


def load_corpus(path):
    seen = set()
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row["question"] = clean_text(row.get("question"))
            row["answer"] = clean_text(row.get("answer"))
            row["ama_title"] = clean_text(row.get("ama_title")) or "Untitled AMA"
            if len(row["question"]) < 20 or len(row["question"]) > 1000 or len(row["answer"]) < 50:
                continue
            if int(row.get("answer_score", 0) or 0) < 0:
                continue
            key = (row["question"].casefold(), row["answer"].casefold())
            if key in seen:
                continue
            seen.add(key)
            row["thread_id"] = str(row.get("thread_id", ""))
            row["source_url"] = clean_text(row.get("source_url"))
            rows.append(row)
    if not rows:
        raise ValueError(f"No usable Q&A rows found in {path}")
    return rows


def tokenize(text):
    return [STEMMER.stem(t) for t in TOKEN_RE.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


def corpus_fingerprint(corpus):
    digest = hashlib.sha256()
    for row in corpus:
        # Only fields used by retrieval and source alignment affect the indexes.
        digest.update(json.dumps(
            [row.get("thread_id", ""), row["question"], row["answer"]],
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_bm25(path, tokenized_docs):
    """Persist BM25 postings and IDF values in portable SQLite, never pickle."""
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.executescript("""
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value REAL NOT NULL);
            CREATE TABLE doc_lengths (doc_id INTEGER PRIMARY KEY, length INTEGER NOT NULL);
            CREATE TABLE terms (term TEXT PRIMARY KEY, idf REAL NOT NULL, doc_freq INTEGER NOT NULL);
            CREATE TABLE postings (term TEXT NOT NULL, doc_id INTEGER NOT NULL, tf INTEGER NOT NULL,
                                   PRIMARY KEY(term, doc_id));
            CREATE INDEX postings_doc_idx ON postings(doc_id);
        """)
        count = len(tokenized_docs)
        lengths = [len(doc) for doc in tokenized_docs]
        average_length = sum(lengths) / count if count else 0.0
        conn.executemany("INSERT INTO metadata VALUES (?, ?)", [
            ("doc_count", float(count)), ("average_length", average_length),
            ("k1", BM25_K1), ("b", BM25_B),
        ])
        conn.executemany("INSERT INTO doc_lengths VALUES (?, ?)", enumerate(lengths))
        term_docs = {}
        for doc_id, tokens in enumerate(tokenized_docs):
            for term, tf in Counter(tokens).items():
                term_docs.setdefault(term, []).append((doc_id, tf))
        rows = []
        for term, postings in term_docs.items():
            df = len(postings)
            # Positive Robertson-Sparck Jones IDF avoids negative scores for common terms.
            idf = math.log1p((count - df + 0.5) / (df + 0.5))
            conn.execute("INSERT INTO terms VALUES (?, ?, ?)", (term, idf, df))
            rows.extend((term, doc_id, tf) for doc_id, tf in postings)
            if len(rows) >= 50000:
                conn.executemany("INSERT INTO postings VALUES (?, ?, ?)", rows)
                rows.clear()
        if rows:
            conn.executemany("INSERT INTO postings VALUES (?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()


class Retriever:
    def __init__(self, corpus, encoder, cache_dir=".cache/rag", model_name="", model_revision="main"):
        self.corpus = corpus
        self.encoder = encoder
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name or encoder.__class__.__name__
        self.model_revision = model_revision
        self.texts = [f"Question: {d['question']} Answer: {d['answer']}" for d in corpus]
        self.tokens = [tokenize(text) for text in self.texts]
        self._cache_identity = {
            "schema": CACHE_SCHEMA,
            "corpus_sha256": corpus_fingerprint(corpus),
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "count": len(corpus),
            "dimension": int(encoder.get_sentence_embedding_dimension()),
        }
        self.manifest_path = self.cache_dir / "manifest.json"
        self.embeddings_path = self.cache_dir / "embeddings.npy"
        self.faiss_path = self.cache_dir / "faiss.index"
        self.bm25_path = self.cache_dir / "bm25.sqlite3"
        loaded = False
        if self._cache_is_valid():
            try:
                self._load_cache()
                loaded = True
                print(f"Loaded cached FAISS, embeddings, and BM25 index from {self.cache_dir}")
            except (OSError, ValueError, sqlite3.Error, RuntimeError) as exc:
                print(f"Index cache is invalid ({exc}); rebuilding it")
        if not loaded:
            print("Building retrieval indexes (cache absent or corpus/model changed)...")
            self._build_cache()
            self._load_cache()

    def _cache_is_valid(self):
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return (all(manifest.get(k) == v for k, v in self._cache_identity.items())
                    and all(p.is_file() and p.stat().st_size > 0 for p in
                            (self.embeddings_path, self.faiss_path, self.bm25_path)))
        except (OSError, ValueError, TypeError):
            return False

    def _build_cache(self):
        embeddings = self.encoder.encode(
            self.texts, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=True,
        ).astype("float32")
        if embeddings.shape != (len(self.corpus), self._cache_identity["dimension"]):
            raise ValueError(f"Unexpected embedding matrix shape: {embeddings.shape}")

        faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss_index.add(np.ascontiguousarray(embeddings))
        temp_embeddings = self.embeddings_path.with_suffix(".npy.tmp")
        temp_faiss = self.faiss_path.with_suffix(".index.tmp")
        temp_bm25 = self.bm25_path.with_suffix(".sqlite3.tmp")
        temp_manifest = self.manifest_path.with_suffix(".json.tmp")
        try:
            with temp_embeddings.open("wb") as file:
                np.save(file, embeddings, allow_pickle=False)
            faiss.write_index(faiss_index, str(temp_faiss))
            _write_bm25(temp_bm25, self.tokens)
            temp_manifest.write_text(json.dumps(self._cache_identity, indent=2), encoding="utf-8")
            os.replace(temp_embeddings, self.embeddings_path)
            os.replace(temp_faiss, self.faiss_path)
            os.replace(temp_bm25, self.bm25_path)
            # Manifest is replaced last, so partial builds are never marked valid.
            os.replace(temp_manifest, self.manifest_path)
        finally:
            for temp in (temp_embeddings, temp_faiss, temp_bm25, temp_manifest):
                if temp.exists():
                    temp.unlink()

    def _load_cache(self):
        self.embeddings = np.load(self.embeddings_path, allow_pickle=False)
        self.index = faiss.read_index(str(self.faiss_path))
        if self.embeddings.shape != (len(self.corpus), self._cache_identity["dimension"]):
            raise ValueError("Cached embedding shape does not match the current corpus/model")
        if self.index.ntotal != len(self.corpus):
            raise ValueError("Cached FAISS index size does not match the current corpus")
        with sqlite3.connect(self.bm25_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            meta = dict(db.execute("SELECT key, value FROM metadata"))
            meta["doc_lengths"] = dict(db.execute("SELECT doc_id, length FROM doc_lengths"))
        if int(meta.get("doc_count", -1)) != len(self.corpus):
            raise ValueError("Cached BM25 index size does not match the current corpus")
        self._bm25_meta = meta

    def _bm25_scores(self, query_tokens):
        scores = np.zeros(len(self.corpus), dtype="float32")
        if not query_tokens:
            return scores
        avgdl = self._bm25_meta["average_length"] or 1.0
        k1, b = self._bm25_meta["k1"], self._bm25_meta["b"]
        with sqlite3.connect(self.bm25_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            for term in set(query_tokens):
                term_row = db.execute("SELECT idf FROM terms WHERE term = ?", (term,)).fetchone()
                if not term_row:
                    continue
                idf = term_row[0]
                postings = db.execute("SELECT doc_id, tf FROM postings WHERE term = ?", (term,))
                for doc_id, tf in postings:
                    length = self._bm25_meta["doc_lengths"][doc_id]
                    denom = tf + k1 * (1 - b + b * length / avgdl)
                    scores[doc_id] += idf * (tf * (k1 + 1) / denom)
        return scores

    def search(self, query, candidate_k=40):
        k = min(candidate_k, len(self.corpus))
        q_tokens = tokenize(query)
        lexical = self._bm25_scores(q_tokens)
        q_vec = self.encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].astype("float32")
        dense_scores, dense_indices = self.index.search(q_vec.reshape(1, -1), len(self.corpus))
        dense = np.empty(len(self.corpus), dtype="float32")
        dense[dense_indices[0]] = dense_scores[0]
        # Reciprocal Rank Fusion makes the two rank lists comparable.
        lex_order = np.argsort(lexical)[::-1]
        dense_order = np.argsort(dense)[::-1]
        fused = np.zeros(len(self.corpus), dtype="float32")
        for rank, idx in enumerate(lex_order, 1):
            fused[idx] += 1.0 / (60 + rank)
        for rank, idx in enumerate(dense_order, 1):
            fused[idx] += 1.0 / (60 + rank)
        indices = np.argsort(fused)[::-1][:k]
        result = []
        for idx in indices:
            item = dict(self.corpus[int(idx)])
            item["retrieval_score"] = float(fused[idx])
            result.append(item)
        return result
