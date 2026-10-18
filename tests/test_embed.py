import sys
import types

from rabbithole.embed import STEmbedder


class FakeModel:
    device = "cpu"
    def __init__(self, dimension=None):
        self.dimension = dimension
        self.new_calls = self.old_calls = 0
    def get_embedding_dimension(self):
        self.new_calls += 1
        return self.dimension
    def get_sentence_embedding_dimension(self):
        self.old_calls += 1
        return 17


def test_embedding_dimension_prefers_current_method(monkeypatch):
    model = FakeModel(19)
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=lambda *a, **k: model))
    emb = STEmbedder("fake", fp16=False)
    assert emb.dim == 19 and model.new_calls == 1 and model.old_calls == 0


def test_embedding_dimension_falls_back_when_current_method_missing(monkeypatch):
    model = FakeModel(None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=lambda *a, **k: model))
    emb = STEmbedder("fake", fp16=False)
    assert emb.dim == 17 and model.new_calls == 1 and model.old_calls == 1
