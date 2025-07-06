"""
embedder.py
-----------
Text → 384-D unit vector using ONNX Runtime.
Falls back gracefully if QNNExecutionProvider (Snapdragon NPU) is absent.

Dependencies:
    pip install onnxruntime==1.18.0 transformers==4.* numpy
"""
from pathlib import Path
import os, threading
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

BASE_DIR   = Path(__file__).resolve().parent
MODEL_DIR   = BASE_DIR / Path("models/all-MiniLM-L6-v2-onnx")
MODEL_PATH  = MODEL_DIR / "model.onnx"
MAX_LEN     = 256
STRIDE      = 128
BATCH_SIZE  = 16

if not MODEL_DIR.is_dir():
    # You can raise, log, or exit here
    raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

# ---------- model & tokenizer are global singletons -------------------------
_tokenizer   = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
_session     = None
_session_lck = threading.Lock()


def _get_session() -> ort.InferenceSession:
    global _session
    with _session_lck:
        if _session is not None:
            return _session
        providers = [
            ("QNNExecutionProvider", {"backend_path": "QnnHtp.dll"}),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        tried, ok = [], None
        for p in providers:
            try:
                _session = ort.InferenceSession(
                    MODEL_PATH.as_posix(),
                    sess_options=ort.SessionOptions(),
                    providers=[p] if isinstance(p, str) else [p[0]],
                    provider_options=[{}] if isinstance(p, str) else [p[1]],
                )
                ok = p[0] if isinstance(p, tuple) else p
                break
            except Exception:
                tried.append(p[0] if isinstance(p, tuple) else p)
        if _session is None:
            raise RuntimeError(f"ONNX providers failed: {tried}")
        print(f"[Embedder] using {ok}")
        return _session


# ---------- helper ----------------------------------------------------------
def _chunk(text: str) -> list[list[int]]:
    toks = _tokenizer.encode(text, add_special_tokens=True)
    out  = []
    for i in range(0, len(toks), STRIDE):
        win = toks[i : i + MAX_LEN]
        win += [_tokenizer.pad_token_id] * (MAX_LEN - len(win))
        out.append(win)
        if i + MAX_LEN >= len(toks):
            break
    return out or [[_tokenizer.pad_token_id] * MAX_LEN]


# ---------- public API ------------------------------------------------------
def embed_text(text: str) -> np.ndarray:
    sess   = _get_session()
    ids_nm_tt = [i.name for i in sess.get_inputs()]   # usually id,mask,token-type
    chunks = _chunk(text)
    embs   = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = np.array(chunks[i : i + BATCH_SIZE], dtype=np.int64)
        mask  = (batch != _tokenizer.pad_token_id).astype(np.int64)
        tt    = np.zeros_like(batch, dtype=np.int64)
        # model’s 2nd output is pooled sentence embedding (MiniLM export)
        out = sess.run(None, {ids_nm_tt[0]: batch,
                              ids_nm_tt[1]: mask,
                              ids_nm_tt[2]: tt})[1]
        embs.append(out)
    mat = np.vstack(embs)                       # (n_chunks, 384)
    mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    v = mat.mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-12)
