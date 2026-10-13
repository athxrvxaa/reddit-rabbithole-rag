import tempfile
import unittest
from pathlib import Path

import numpy as np

from retrieval import Retriever


class TinyEncoder:
    """Deterministic stand-in for SentenceTransformer in cache tests."""

    def __init__(self, dimension=3):
        self.dimension = dimension
        self.bulk_encode_calls = 0

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True,
               show_progress_bar=False):
        if len(texts) > 1:
            self.bulk_encode_calls += 1
        vectors = []
        for text in texts:
            lower = text.lower()
            vector = np.array([
                1.0 if any(word in lower for word in ("prison", "cell")) else 0.0,
                1.0 if any(word in lower for word in ("career", "job")) else 0.0,
                1.0,
            ], dtype=np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vector)
                if norm:
                    vector /= norm
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)


def sample_corpus():
    return [
        {"thread_id": "t3_a", "question": "What is prison life like?",
         "answer": "Daily life in a prison cell includes routines and meals.",
         "ama_title": "Prison AMA", "source_url": "https://example.test/a"},
        {"thread_id": "t3_b", "question": "What is a difficult career like?",
         "answer": "My job involved long hours and difficult decisions.",
         "ama_title": "Career AMA", "source_url": "https://example.test/b"},
    ]


class PersistentRetrieverTests(unittest.TestCase):
    def test_build_then_reuse_cache_without_reembedding_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "index cache"
            corpus = sample_corpus()
            first_encoder = TinyEncoder()
            first = Retriever(corpus, first_encoder, cache, "test-model", "rev-1")
            self.assertEqual(first_encoder.bulk_encode_calls, 1)
            self.assertTrue((cache / "embeddings.npy").is_file())
            self.assertTrue((cache / "faiss.index").is_file())
            self.assertTrue((cache / "bm25.sqlite3").is_file())
            self.assertFalse(list(cache.glob("*.pkl")))
            first_result = first.search("prison routine", candidate_k=2)

            second_encoder = TinyEncoder()
            second = Retriever(corpus, second_encoder, cache, "test-model", "rev-1")
            self.assertEqual(second_encoder.bulk_encode_calls, 0)
            second_result = second.search("prison routine", candidate_k=2)
            self.assertEqual([d["thread_id"] for d in first_result],
                             [d["thread_id"] for d in second_result])

    def test_dataset_change_rebuilds_indexes(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = sample_corpus()
            first_encoder = TinyEncoder()
            Retriever(corpus, first_encoder, tmp, "test-model", "rev-1")
            corpus[0]["answer"] += " Updated response."
            changed_encoder = TinyEncoder()
            Retriever(corpus, changed_encoder, tmp, "test-model", "rev-1")
            self.assertEqual(changed_encoder.bulk_encode_calls, 1)

    def test_model_identity_change_rebuilds_indexes(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = sample_corpus()
            Retriever(corpus, TinyEncoder(), tmp, "model-a", "rev-1")
            changed_encoder = TinyEncoder()
            Retriever(corpus, changed_encoder, tmp, "model-b", "rev-1")
            self.assertEqual(changed_encoder.bulk_encode_calls, 1)


if __name__ == "__main__":
    unittest.main()
