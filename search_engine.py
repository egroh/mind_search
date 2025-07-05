"""One‑stop helper for ingesting folders and running queries."""

from pathlib import Path
import os
from db import init_db, add_document, search


class SearchEngine:
    def __init__(self, db_path: str = "search.db") -> None:
        self.con = init_db(db_path)

    # ----- Corpus ingestion -------------------------------------------------
    def ingest_folder(
        self, folder: str | os.PathLike, exts: tuple[str, ...] = (".txt", ".md")
    ) -> None:
        folder = Path(folder)
        for fp in folder.rglob("*"):
            if fp.suffix.lower() not in exts:
                continue
            try:
                txt = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                txt = fp.name  # fallback: just file name
            add_document(self.con, str(fp), txt)

    # ----- Query ------------------------------------------------------------
    def query(self, text: str, k: int = 10):
        return search(self.con, text, k)
