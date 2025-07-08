# indexer.py
"""
Maintains an in-memory embedding matrix + filenames + raw text.
"""

from pathlib import Path
import json, threading
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PySide6.QtCore import Signal, QObject

from latent_search.embedder import embed_text, embed_image, embed_audio
from latent_search.text_extract_pdf import extract_texts_from_dir

EMB_DIR      = Path("embeddings")
EMB_NPY_PATH = EMB_DIR / "matrix.npy"
EMB_META     = EMB_DIR / "files.json"

class IndexBuilt(QObject):
    ready = Signal()

class FileIndexer(FileSystemEventHandler):
    """Thread-safe index based on parallel lists + a dict of raw text."""
    def __init__(self, folder: Path):
        super().__init__()
        self.folder = folder
        self.paths  : list[str]          = []
        self.matrix : np.ndarray         = np.empty((0, 384), dtype=np.float32)
        self.texts  : dict[str, str]     = {}    # ← NEW: path → raw content
        self.lock   = threading.Lock()
        EMB_DIR.mkdir(exist_ok=True)
        self._load_or_build()
        self._start_watcher()
        self.signal = IndexBuilt()

    # --------------- public API --------------------------------------------
    def search(self, query: str, k: int = 10):
        q = embed_text(query)
        with self.lock:
            if self.matrix.size == 0:
                return []
            sims = self.matrix @ q
            best = np.argsort(-sims)[:k]
            # ← return triples (score, path, content)
            results = []
            for i in best:
                p = self.paths[i]
                txt = self.texts.get(p, "")  # empty for images/audio
                results.append((float(sims[i]), p, txt))
            return results

    # --------------- build/rebuild -----------------------------------------
    def _load_or_build(self):
        try:
            if EMB_NPY_PATH.exists() and EMB_META.exists():
                self.matrix = np.load(EMB_NPY_PATH)
                self.paths  = json.loads(EMB_META.read_text())
                # LOAD raw texts too
                for p in self.paths:
                    self.texts[p] = Path(p).read_text(errors="ignore")
                if self.matrix.shape[0] == len(self.paths):
                    print("[Indexer] loaded saved index")
                    return
        except Exception:
            print("[Indexer] saved index corrupted, rebuilding...")
        self._rebuild()

    def _rebuild(self):
        print("[Indexer] building index…")
        texts = {}
        # — plain-text files
        for p in self.folder.rglob("*"):
            if p.suffix.lower() in {".txt", ".md"}:
                txt = p.read_text(errors="ignore")
                texts[p.as_posix()] = txt
        # — PDFs via helper
        pdfs = extract_texts_from_dir(self.folder.as_posix())
        for name, txt in pdfs.items():
            full = (self.folder / name).as_posix()
            texts[full] = txt

        vecs, paths = [], []
        for path, txt in texts.items():
            vecs.append(embed_text(txt))
            paths.append(path)

        with self.lock:
            if vecs:
                self.matrix = np.vstack(vecs)
                self.paths  = paths
                self.texts  = texts.copy()        # ← store raw texts
                np.save(EMB_NPY_PATH, self.matrix)
                EMB_META.write_text(json.dumps(self.paths))
        print(f"[Indexer] built {len(self.paths)} docs")
        self.signal.ready.emit()

    # --------------- live updates via watchdog ----------------------------
    def _start_watcher(self):
        obs = Observer()
        obs.schedule(self, self.folder.as_posix(), recursive=True)
        obs.daemon = True
        obs.start()

    def on_created(self, event):
        self._index_file(Path(event.src_path))

    def on_modified(self, event):
        self._index_file(Path(event.src_path))

    def on_deleted(self, event):
        p = Path(event.src_path).as_posix()
        with self.lock:
            if p in self.paths:
                i = self.paths.index(p)
                self.paths.pop(i)
                self.matrix = np.delete(self.matrix, i, axis=0)
                self.texts.pop(p, None)
                np.save(EMB_NPY_PATH, self.matrix)
                EMB_META.write_text(json.dumps(self.paths))

    def _index_file(self, fp: Path):
        vec, txt = self._embed_file(fp)
        if vec is None:
            return  # unsupported or empty

        p = fp.as_posix()
        with self.lock:
            if p in self.paths:
                # update existing
                idx = self.paths.index(p)
                self.matrix[idx] = vec
                if txt:
                    self.texts[p] = txt
                print(f"[Indexer] updated {p}")
            else:
                # append new
                self.paths.append(p)
                self.matrix = np.vstack([self.matrix, vec])
                if txt:
                    self.texts[p] = txt
                print(f"[Indexer] added   {p}")

            # persist changes
            np.save(EMB_NPY_PATH, self.matrix)
            EMB_META.write_text(json.dumps(self.paths))

