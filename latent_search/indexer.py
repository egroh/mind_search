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
EMBED_DIM = 512

class IndexBuilt(QObject):
    ready = Signal()

class FileIndexer(FileSystemEventHandler):
    """Thread-safe index based on parallel lists + a dict of raw text."""
    def __init__(self, folder: Path):
        super().__init__()
        self.folder = folder
        self.paths  : list[str]          = []
        self.matrix : np.ndarray         = np.empty((0, EMBED_DIM), dtype=np.float32)
        self.texts  : dict[str, str]     = {}    # ← NEW: path → raw content
        self.lock   = threading.Lock()
        EMB_DIR.mkdir(exist_ok=True)
        self._load_or_build()
        self._start_watcher()
#        self.signal = IndexBuilt()

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
        """
        Load an existing index (if any), then update it to reflect
        added/removed files under self.folder. Reuses embeddings
        when possible, embeds only new files.
        """
        print("[Indexer] loading or updating index…")

        # 1) Gather all supported files on disk
        exts = {".txt", ".md", ".pdf",
                ".png", ".jpg", ".jpeg", ".webp",
                ".wav", ".mp3", ".flac"}
        current = [str(p) for p in self.folder.rglob("*") if p.suffix.lower() in exts]
        current.sort()

        # 2) Load persisted index if it matches shape; build vec_map
        vec_map = {}
        if EMB_NPY_PATH.exists() and EMB_META.exists():
            try:
                old_mat  = np.load(EMB_NPY_PATH)
                old_paths= json.loads(EMB_META.read_text())
                if old_mat.shape == (len(old_paths), EMBED_DIM):
                    vec_map = {path: old_mat[i] for i, path in enumerate(old_paths)}
                else:
                    print(f"[Indexer] persisted index shape {old_mat.shape} != ({len(old_paths)},{EMBED_DIM}), rebuilding entries")
            except Exception as e:
                print("[Indexer] failed loading persisted index:", e)

        # 3) Reconstruct index, reusing old embeddings where possible
        new_paths, new_vecs, new_texts = [], [], {}
        for p in current:
            if p in vec_map:
                vec = vec_map[p]
                # reload raw text for text/pdf
                if Path(p).suffix.lower() in {".txt", ".md", ".pdf"}:
                    try:
                        new_texts[p] = Path(p).read_text(errors="ignore")
                    except:
                        new_texts[p] = ""
            else:
                vec, txt = self._embed_file(Path(p))
                if vec is None:
                    continue
                if txt:
                    new_texts[p] = txt
            new_paths.append(p)
            new_vecs.append(vec)

        # 4) Swap into memory & persist
        with self.lock:
            self.paths  = new_paths
            self.matrix = (np.vstack(new_vecs)
                           if new_vecs
                           else np.empty((0, EMBED_DIM), dtype=np.float32))
            self.texts  = new_texts
            np.save(EMB_NPY_PATH, self.matrix)
            EMB_META.write_text(json.dumps(self.paths))

        print(f"[Indexer] index updated: {len(self.paths)} files")


    def _rebuild(self):
        print("[Indexer] building index…")
        vecs, paths, texts = [], [], {}

        # scan everything under self.folder
        for fp in self.folder.rglob("*"):
            vec, txt = self._embed_file(fp)
            if vec is None:
                continue

            p = fp.as_posix()
            paths.append(p)
            vecs.append(vec)
            if txt:
                texts[p] = txt

        # atomically swap in new index
        with self.lock:
            if vecs:
                self.matrix = np.vstack(vecs)
            else:
                # no matches → empty matrix
                self.matrix = np.empty((0, EMBED_DIM), dtype=np.float32)
            self.paths = paths
            self.texts = texts
            # persist to disk
            np.save(EMB_NPY_PATH, self.matrix)
            EMB_META.write_text(json.dumps(self.paths))

        print(f"[Indexer] built {len(self.paths)} docs")


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


    def _embed_file(self, fp: Path):
        """
        Given a single file path, return (vec: np.ndarray, txt: str|None).
        If the file isn't supported or is empty, returns (None, None).
        """
        ext = fp.suffix.lower()
        txt = None

        # TEXT / MD
        if ext in {".txt", ".md"}:
            txt = fp.read_text(errors="ignore")
            if not txt.strip():
                return None, None
            vec = embed_text(txt)

        # PDF
        elif ext == ".pdf":
            texts = extract_texts_from_dir(fp.parent.as_posix())
            txt = texts.get(fp.name, "")
            if not txt.strip():
                return None, None
            vec = embed_text(txt)

        # IMAGES
        elif ext in {".png", ".jpg", ".jpeg", ".webp"}:
            vec = embed_image(fp.as_posix())

        # AUDIO
        elif ext in {".wav", ".mp3", ".flac"}:
            vec = embed_audio(fp.as_posix())

        else:
            return None, None

        return vec, txt