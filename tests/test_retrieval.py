import numpy as np
import pytest

from rabbithole.config import Settings
from rabbithole.db import connect
from rabbithole.embed import HashEmbedder
from rabbithole.indexing import build_index
from rabbithole.retrieval import Retriever, rrf_fuse

DOCS = [
    # (thread, title, category, question, answer)
    ("t1", "I am a prison guard. AMA", "crime_law", "What do inmates do all day?", "Inmates wake early, eat breakfast, work in the laundry or kitchen, and spend free time in the yard playing cards and basketball."),
    ("t1", "I am a prison guard. AMA", "crime_law", "Is it dangerous inside prison?", "Mostly boring, but fights happen. Guards rely on routine and communication to keep the cell block calm every shift."),
    ("t1", "I am a prison guard. AMA", "crime_law", "What about food in the prison?", "Meals are bland and repetitive: oatmeal, sandwiches and mystery casserole served on trays at fixed times."),
    ("t2", "I am a flight attendant. AMA", "jobs_trades", "What is the worst part of flying for a living?", "Jet lag, unruly passengers and living out of a suitcase. The schedule changes constantly and sleep suffers."),
    ("t3", "I am a nurse. AMA", "health_medicine", "How hard are night shifts for nurses?", "Night shifts wreck your sleep. Coffee helps, but patients still need care at four in the morning every single night."),
    ("t4", "I am a chef. AMA", "jobs_trades", "What do kitchens never tell customers?", "Kitchens are hot and loud, and the special is often what needs using up before it spoils in the fridge."),
]


class FakeReranker:
    def score(self, query, texts):
        q = set(query.lower().split())
        return np.array([len(q & set(t.lower().split())) - 2.0 for t in texts], dtype="float32")


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    root = tmp_path_factory.mktemp("rr")
    s = Settings(data_dir=root / "data", artifacts_dir=root / "art", per_thread=2, abstain_threshold=0.0)
    con = connect(s.db_path)
    for i, (t, title, cat, q, a) in enumerate(DOCS):
        con.execute(
            "INSERT INTO pairs(thread_id,op_author,op_source,title,context,category,question,answer,question_id,answer_id,"
            "question_score,answer_score,created_utc,source_url,content_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (t, "op", "submission", title, "", cat, q, a, f"q{i}", f"a{i}", 3, 5, 0, f"https://r/{t}/{i}", f"h{i}"),
        )
    con.commit()
    emb = HashEmbedder(128)
    build_index(s, emb)
    return Retriever(s.artifacts_dir, emb, FakeReranker(), s)


def test_rrf_prefers_agreement():
    out = rrf_fuse([[1, 2, 3], [3, 4, 1]])
    assert [i for i, _ in out][:2] == [1, 3]


def test_modes_find_relevant(retriever):
    for mode in ("bm25", "dense", "hybrid", "hybrid_rerank"):
        res = retriever.search("what do inmates do all day in prison", top_k=3, mode=mode, per_thread=0)
        assert res.hits and res.hits[0].doc["thread_id"] == "t1", mode


def test_per_thread_cap(retriever):
    res = retriever.search("prison inmates guards food fights", top_k=5, mode="hybrid", per_thread=1)
    threads = [h.doc["thread_id"] for h in res.hits]
    assert len(threads) == len(set(threads))


def test_category_filter(retriever):
    res = retriever.search("sleep and schedules", top_k=5, mode="hybrid", category="health_medicine", per_thread=0)
    assert res.hits and all(h.doc["category"] == "health_medicine" for h in res.hits)


def test_abstain_on_irrelevant(retriever):
    res = retriever.search("quantum chromodynamics lagrangian", top_k=3, mode="hybrid_rerank")
    assert res.abstain is True


def test_not_abstain_on_relevant(retriever):
    res = retriever.search("what do inmates do all day in prison", top_k=3, mode="hybrid_rerank")
    assert res.abstain is False


def test_two_tier_confidence_gate(retriever):
    # Query has a known score from the fake reranker; move tier boundaries around it.
    old_hard, old_low = retriever.settings.abstain_threshold, retriever.settings.lowconf_threshold
    res = retriever.search("what do inmates do all day in prison", top_k=1, mode="hybrid_rerank", per_thread=0)
    score = res.hits[0].rerank
    retriever.settings.abstain_threshold = score - 2
    retriever.settings.lowconf_threshold = score + 1
    low = retriever.search("what do inmates do all day in prison", top_k=1, mode="hybrid_rerank", per_thread=0)
    assert low.confidence == "low" and not low.abstain
    retriever.settings.abstain_threshold = score + 1
    hard = retriever.search("what do inmates do all day in prison", top_k=1, mode="hybrid_rerank", per_thread=0)
    assert hard.confidence == "none" and hard.abstain
    retriever.settings.abstain_threshold = score - 2
    retriever.settings.lowconf_threshold = score - 3
    high = retriever.search("what do inmates do all day in prison", top_k=1, mode="hybrid_rerank", per_thread=0)
    assert high.confidence == "high" and not high.abstain
    retriever.settings.abstain_threshold, retriever.settings.lowconf_threshold = old_hard, old_low


def test_empty_and_stopword_query(retriever):
    assert retriever.search("the of and", top_k=3, mode="bm25").hits == []
