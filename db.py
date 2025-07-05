# db.py
"""
SQLite wrapper that stores fp32 vectors as BLOBs
and is SAFE to use from multiple threads.

Later, when you switch to `sqlite-vec`, only the table
creation SQL and the SELECT in `search()` need to change.
"""
from pathlib import Path
import sqlite3
import numpy as np
from embeddings import embed

DEFAULT_DB = "search.db"


def _vec_to_blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def _blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def init_db(path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    # ⚠️  key line below
    con = sqlite3.connect(path, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS docs(
            id      INTEGER PRIMARY KEY,
            path    TEXT UNIQUE,
            content TEXT,
            emb     BLOB
        )
        """
    )
    return con


def add_document(con: sqlite3.Connection, path: str, text: str) -> None:
    vec = embed(text)
    con.execute(
        "INSERT OR REPLACE INTO docs(path, content, emb) VALUES (?, ?, ?)",
        (path, text, _vec_to_blob(vec)),
    )
    con.commit()


def search(con: sqlite3.Connection, query: str, top_k: int = 10):
    qv = embed(query)
    rows = con.execute("SELECT path, content, emb FROM docs").fetchall()
    scored: list[tuple[float, str, str]] = []
    for path, content, blob in rows:
        dv = _blob_to_vec(blob)
        score = float(np.dot(qv, dv))  # vectors already unit-length
        scored.append((score, path, content))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[:top_k]
