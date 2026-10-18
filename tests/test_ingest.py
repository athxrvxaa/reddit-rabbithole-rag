import sys
import types
import shutil

import pytest

from rabbithole.config import Settings
from rabbithole.db import connect, get_state
from rabbithole.ingest import ingest, keep_thread
from rabbithole.source_info import diagnose_rows, suggest_rate


def make_rows(n_threads=400, per_thread=10, interleave=False):
    rows = []
    for t in range(n_threads):
        for c in range(per_thread):
            rows.append(dict(id=f"c{t}_{c}", link_id=f"t3_th{t}", parent_id=f"t3_th{t}", author=f"u{c}",
                             body=f"comment {t} {c}", score=1, created_utc=1_000_000 + (c * n_threads + t if interleave else t * per_thread + c)))
    if interleave:
        rows.sort(key=lambda r: r["created_utc"])
    return rows


class FakeDS:
    def __init__(self, rows, fail_after=None):
        self.rows, self.fail_after = rows, fail_after

    def skip(self, n):
        return FakeDS(self.rows[n:], self.fail_after)

    def __iter__(self):
        for i, r in enumerate(self.rows):
            if self.fail_after is not None and i >= self.fail_after:
                raise ConnectionError("boom")
            yield r


@pytest.fixture
def fake_datasets(monkeypatch):
    state = {"rows": make_rows(), "fail_after": None}

    def load_dataset(name, split=None, streaming=False):
        fa, state["fail_after"] = state["fail_after"], None  # fail only once
        return FakeDS(state["rows"], fa)

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    monkeypatch.setattr("time.sleep", lambda s: None)
    return state


def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "art")


def count(s):
    return connect(s.db_path).execute("SELECT COUNT(*) FROM comments").fetchone()[0]


def test_keep_thread_is_deterministic_and_close_to_rate():
    ids = [f"th{i}" for i in range(20_000)]
    kept = [i for i in ids if keep_thread(i, 0.1)]
    assert kept == [i for i in ids if keep_thread(i, 0.1)]
    assert 0.08 < len(kept) / len(ids) < 0.12
    assert keep_thread("x", 1.0)


def test_full_ingest(tmp_path, fake_datasets):
    s = settings(tmp_path)
    ingest(s, batch_size=500, method="stream")
    assert count(s) == 4000
    assert get_state(connect(s.db_path), "ingest_complete") == "1"


def test_thread_sampling_keeps_whole_threads(tmp_path, fake_datasets):
    s = settings(tmp_path)
    ingest(s, batch_size=500, thread_sample_rate=0.25, method="stream")
    con = connect(s.db_path)
    per = [r[0] for r in con.execute("SELECT COUNT(*) FROM comments GROUP BY link_id")]
    assert per and all(c == 10 for c in per)  # no half-cut threads
    assert 0.15 < len(per) / 400 < 0.35


def test_max_comments_caps_scan(tmp_path, fake_datasets):
    s = settings(tmp_path)
    ingest(s, batch_size=500, max_comments=1000, method="stream")
    assert count(s) == 1000
    assert get_state(connect(s.db_path), "ingest_complete") is None


def test_target_rows_caps_kept(tmp_path, fake_datasets):
    s = settings(tmp_path)
    ingest(s, batch_size=500, target_rows=1000, method="stream")
    assert 1000 <= count(s) < 4000


def test_resume_after_network_error_has_no_duplicates(tmp_path, fake_datasets):
    fake_datasets["fail_after"] = 1700  # dies mid-stream once, then succeeds
    s = settings(tmp_path)
    ingest(s, batch_size=500, method="stream")
    assert count(s) == 4000


def test_resume_after_cap_continues(tmp_path, fake_datasets):
    s = settings(tmp_path)
    ingest(s, batch_size=500, max_comments=1000, method="stream")
    ingest(s, batch_size=500, method="stream")
    assert count(s) == 4000


def test_rate_mismatch_on_resume_is_refused(tmp_path, fake_datasets):
    s = settings(tmp_path)
    ingest(s, batch_size=500, thread_sample_rate=0.5, max_comments=1000, method="stream")
    with pytest.raises(SystemExit):
        ingest(s, batch_size=500, thread_sample_rate=0.1, method="stream")


def test_diagnose_layouts():
    assert diagnose_rows(make_rows())["layout"] == "grouped_by_thread"
    assert diagnose_rows(make_rows(interleave=True))["layout"] == "time_ordered"
    import random
    rows = make_rows(); random.Random(0).shuffle(rows)
    assert diagnose_rows(rows)["layout"] == "mixed"


def test_suggest_rate():
    assert suggest_rate(592_000_000, 5_000_000) == pytest.approx(0.00845, abs=1e-5)
    assert suggest_rate(1_000_000, 5_000_000) == 1.0


def parquet_hf(monkeypatch, tmp_path, shard_paths, fail_once=None):
    pytest.importorskip("pyarrow")
    import sys
    calls = []
    failed = set()
    class Api:
        def list_repo_files(self, repo_id, repo_type=None):
            return list(shard_paths)
    def download(repo_id, filename, repo_type=None, local_dir=None):
        calls.append(filename)
        if filename == fail_once and filename not in failed:
            failed.add(filename)
            raise ConnectionError("temporary download failure")
        dest = tmp_path / "download" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(shard_paths[filename], dest)
        return str(dest)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=Api, hf_hub_download=download))
    return calls


def write_parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_parquet_filtering_and_sampling_match_streaming(tmp_path, monkeypatch):
    import datetime
    rows = make_rows(n_threads=30, per_thread=4)
    rows += [dict(id="deleted", link_id="t3_x", parent_id="t3_x", author="[deleted]", body="gone", score=0, created_utc=1),
             dict(id="removed", link_id="t3_y", parent_id="t3_y", author="user", body="[removed]", score=0, created_utc=1)]
    shard = tmp_path / "source.parquet"
    write_parquet(shard, rows)
    files = {"data/IAmA-000.parquet": shard}
    parquet_hf(monkeypatch, tmp_path, files)
    ps = settings(tmp_path / "parquet")
    ingest(ps, thread_sample_rate=.5)
    assert get_state(connect(ps.db_path), "ingest_complete") == "1"

    ss = settings(tmp_path / "stream")
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=lambda *a, **k: FakeDS(rows)))
    ingest(ss, thread_sample_rate=.5, method="stream", batch_size=50)
    pids = [r[0] for r in connect(ps.db_path).execute("SELECT id FROM comments ORDER BY id")]
    sids = [r[0] for r in connect(ss.db_path).execute("SELECT id FROM comments ORDER BY id")]
    assert pids == sids
    assert "deleted" not in pids and "removed" not in pids


def test_parquet_shard_resume_and_rate_mismatch(tmp_path, monkeypatch):
    rows = make_rows(n_threads=2, per_thread=3)
    p0, p1 = tmp_path / "s0.parquet", tmp_path / "s1.parquet"
    write_parquet(p0, rows[:3]); write_parquet(p1, rows[3:])
    files = {"IAmA-000.parquet": p0, "IAmA-001.parquet": p1}
    calls = parquet_hf(monkeypatch, tmp_path, files, fail_once="IAmA-001.parquet")
    s = settings(tmp_path)
    with pytest.raises(ConnectionError):
        ingest(s, thread_sample_rate=.5)
    state_con = connect(s.db_path)
    assert "IAmA-000.parquet" in get_state(state_con, "ingest_parquet_completed")
    with pytest.raises(SystemExit):
        ingest(s, thread_sample_rate=.8)
    ingest(s, thread_sample_rate=.5)
    assert calls.count("IAmA-000.parquet") == 1
    assert count(s) == sum(1 for row in rows if __import__("rabbithole.ingest", fromlist=["_row_to_tuple"])._row_to_tuple(row) and
                           __import__("rabbithole.ingest", fromlist=["keep_thread"]).keep_thread(row["link_id"].replace("t3_", ""), .5))
