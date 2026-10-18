import sqlite3

from rabbithole.config import Settings
from rabbithole.extract import ExtractConfig, _artifact_year_stats, extract_thread, is_bot, pick_op

LONG = "This is a sufficiently long answer from the host that explains things in real detail for the reader. " * 2


def c(id, parent, author, body, score=5):
    return dict(id=id, parent_id=parent, author=author, body=body, score=score, created_utc=1)


def thread():
    return [
        c("a1", "t3_T", "host", "Thanks for all the questions everyone, this is fun."),
        c("q1", "t3_T", "alice", "What is the hardest part of your job?"),
        c("r1", "t1_q1", "host", LONG),
        c("q2", "t3_T", "bob", "Do you ever get scared at work?"),
        c("r2", "t1_q2", "host", LONG + " second", score=0),  # filtered: score
        c("q3", "t3_T", "AutoModerator", "What is the rule here?"),
        c("r3", "t1_q3", "host", LONG),  # filtered: bot parent
        c("h1", "t3_T", "host", "I can answer more."),
        c("r4", "t1_h1", "host", LONG),  # filtered: parent is OP
        c("q5", "t1_r1", "carol", "why?"),
        c("r5", "t1_q5", "host", LONG),  # filtered: too short / follow-up
        c("q6", "t1_r1", "dave", "Could you describe a typical day in much more detail please?"),
        c("r6", "t1_q6", "host", "&gt; typical day\n\n" + LONG + " day", score=9),  # kept, quote dropped
    ]


def test_extract_with_true_op():
    pairs = extract_thread("T", thread(), {"author": "host", "title": "I am a nurse. AMA"}, ExtractConfig())
    ids = {p["answer_id"] for p in pairs}
    assert ids == {"r1", "r6"}
    assert all(p["op_source"] == "submission" for p in pairs)
    p6 = next(p for p in pairs if p["answer_id"] == "r6")
    assert "typical day" not in p6["answer"].split(".")[0].lower() or ">" not in p6["answer"]
    assert pairs[0]["answer_score"] >= pairs[1]["answer_score"]  # sorted by score
    assert pairs[0]["category"] == "health_medicine"
    assert pairs[0]["source_url"].startswith("https://www.reddit.com/comments/T/_/")


def test_artifact_stats_handles_old_docs_without_dates(tmp_path, capsys):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    con = sqlite3.connect(artifacts / "corpus.db")
    con.execute("CREATE TABLE docs(idx INTEGER PRIMARY KEY, thread_id TEXT)")
    con.commit(); con.close()
    _artifact_year_stats(Settings(data_dir=tmp_path / "missing", artifacts_dir=artifacts))
    assert "no created_utc column" in capsys.readouterr().out


def test_no_meta_no_heuristic_means_no_pairs():
    assert extract_thread("T", thread(), None, ExtractConfig()) == []


def test_heuristic_flagged():
    cs = thread() + [c("x", "t3_T", "host", "second top-level answer")]
    op, src = pick_op(cs, None, allow_heuristic=True)
    assert (op, src) == ("host", "heuristic")


def test_bots():
    assert is_bot("AutoModerator") and is_bot("RemindMeBot") and is_bot("foo_bot")
    assert not is_bot("Abbot") and not is_bot("robotfan")


def test_per_thread_cap():
    cfg = ExtractConfig(max_pairs_per_thread=1)
    pairs = extract_thread("T", thread(), {"author": "host", "title": "x"}, cfg)
    assert len(pairs) == 1 and pairs[0]["answer_id"] == "r6"  # highest score
