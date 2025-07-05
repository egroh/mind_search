"""Light‑weight, deterministic text → vector function.
Replace this later with your ONNX/QNN model; keep the API.
"""
import hashlib
import numpy as np

DIM = 384  # keep in sync with real embedding size later


def embed(text: str) -> np.ndarray:
    """Very simple hashed bag‑of‑words → unit‑length vector."""
    vec = np.zeros(DIM, dtype=np.float32)
    for token in text.lower().split():
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[h % DIM] += 1.0
    # L2‑normalise so dot‑product ≈ cosine
    norm = np.linalg.norm(vec)
    if norm:
        vec /= norm
    return vec