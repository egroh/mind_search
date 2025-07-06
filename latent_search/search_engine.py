# search_engine.py
from pathlib import Path
from latent_search.indexer import FileIndexer

class SearchEngine:
    """
    Thin wrapper used by GUI.
    Creates a singleton FileIndexer that is safe to call from threads.
    """
    _instance = None

    def __new__(cls, folder: str | Path = "demo"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init(Path(folder))
        return cls._instance

    def _init(self, folder: Path):
        folder = folder if folder.exists() else Path(".")
        self.indexer = FileIndexer(folder)

    # --------------- API for GUI --------------------------------------------
    def search(self, text: str, k: int = 10):
        return self.indexer.search(text, k)
