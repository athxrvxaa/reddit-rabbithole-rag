from rabbithole.config import Settings
from rabbithole.db import connect
from rabbithole.meta import fetch_meta


class Response:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload or {}

    def json(self):
        return self._payload


class Session:
    def __init__(self):
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= 5:
            return Response(503)
        return Response(200, {"data": [{"id": "thread", "author": "host", "title": "Ask me anything",
                                          "selftext": "", "score": 1, "created_utc": 123, "num_comments": 20}]})


def test_failed_metadata_batch_is_retried_and_saved(tmp_path, monkeypatch):
    monkeypatch.setattr("rabbithole.meta.time.sleep", lambda _: None)
    settings = Settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "art")
    con = connect(settings.db_path)
    con.executemany("INSERT INTO comments VALUES (?,?,?,?,?,?,?)",
                    [(f"c{i}", "thread", "thread", "user", "body", 1, 123) for i in range(10)])
    con.commit()
    session = Session()
    fetch_meta(settings, session=session, sleep=0)
    assert session.calls == 6
    row = connect(settings.db_path).execute("SELECT author, title, found FROM threads WHERE thread_id='thread'").fetchone()
    assert tuple(row) == ("host", "Ask me anything", 1)
