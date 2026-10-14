"""Read-only access to the document store (artifacts/corpus.db)."""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path


class CorpusStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._lock = threading.Lock()

    def get_many(self, idxs: list[int]) -> dict[int, dict]:
        if not idxs:
            return {}
        ph = ",".join("?" * len(idxs))
        with self._lock:
            rows = self._con.execute(f"SELECT * FROM docs WHERE idx IN ({ph})", [int(i) for i in idxs]).fetchall()
        return {r["idx"]: dict(r) for r in rows}

    def get(self, idx: int) -> dict | None:
        return self.get_many([idx]).get(int(idx))

    def count(self) -> int:
        with self._lock:
            return self._con.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
