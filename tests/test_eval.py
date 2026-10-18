import sqlite3

import pytest

from rabbithole.eval import _bootstrap_mean_interval, current_index_queries, intervals_overlap, wilson_interval


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(5, 10)
    assert low < .5 < high
    assert wilson_interval(0, 10)[0] == 0


def test_bootstrap_is_reproducible():
    assert _bootstrap_mean_interval([0, .5, 1], seed=4) == _bootstrap_mean_interval([0, .5, 1], seed=4)


def test_detect_overlapping_mode_intervals():
    assert intervals_overlap({"a": {"r_ci95": [0, .6]}, "b": {"r_ci95": [.4, 1]}}, "r")
    assert not intervals_overlap({"a": {"r_ci95": [0, .2]}, "b": {"r_ci95": [.8, 1]}}, "r")


def artifact_db(tmp_path):
    path = tmp_path / "corpus.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE docs(idx INTEGER PRIMARY KEY, thread_id TEXT, answer_id TEXT)")
    con.executemany("INSERT INTO docs VALUES (?, ?, ?)", [(0, "t1", "a1"), (1, "t2", "a2"), (2, "t3", "a3")])
    con.commit(); con.close()
    return tmp_path


def test_eval_queries_resolve_stable_answer_ids(tmp_path):
    current, dropped = current_index_queries(artifact_db(tmp_path), [
        {"query": "q1", "idx": 90, "answer_id": "a2", "thread_id": "t2"},
        {"query": "q2", "idx": 0, "answer_id": "a3", "thread_id": "t3"},
    ])
    assert dropped == 0
    assert [q["idx"] for q in current] == [1, 2]


def test_eval_queries_drop_partially_stale_and_validate_legacy_ids(tmp_path):
    current, dropped = current_index_queries(artifact_db(tmp_path), [
        {"query": "fresh", "idx": 1, "thread_id": "t2"},
        {"query": "stale", "idx": 9, "thread_id": "old"},
    ])
    assert dropped == 1 and current[0]["idx"] == 1


def test_eval_queries_reject_fully_stale_set_with_regeneration_guidance(tmp_path):
    with pytest.raises(ValueError, match="regenerate queries with eval-make"):
        current_index_queries(artifact_db(tmp_path), [
            {"query": "old1", "idx": 9, "thread_id": "old"},
            {"query": "old2", "idx": 8, "thread_id": "old"},
        ])


def test_eval_make_writes_answer_id_and_thread_id(tmp_path):
    import json
    from rabbithole.eval import make_synthetic_queries
    con = sqlite3.connect(tmp_path / "corpus.db")
    con.execute("CREATE TABLE docs(idx INTEGER PRIMARY KEY, question TEXT, answer TEXT, answer_id TEXT, thread_id TEXT)")
    con.execute("INSERT INTO docs VALUES (7, 'What happened?', ?, 'answer-7', 'thread-7')", ("A" * 300,))
    con.commit(); con.close()
    class FakeLLM:
        def generate(self, prompt, temperature):
            return "What was happening in the story?"
    out = tmp_path / "queries.jsonl"
    make_synthetic_queries(tmp_path, FakeLLM(), 1, out, sleep=0)
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["answer_id"] == "answer-7" and row["thread_id"] == "thread-7"
