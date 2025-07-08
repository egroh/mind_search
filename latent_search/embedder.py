# ── stdlib / third-party ──────────────────────────────────────────────
from pathlib import Path
import os, numpy as np, onnxruntime as ort
from transformers import CLIPTokenizer
from PIL import Image
# import soundfile as sf
import librosa

# ── MODEL LOCATIONS ───────────────────────────────────────────────────
FULL_BASE = Path(__file__).parent / "models" / "clip" / "clip_full_int8_qdq.onnx"
FULL_CTX  = FULL_BASE.with_name(FULL_BASE.stem + "_ctx.onnx")
QNN_DLL   = "QnnHtp.dll"

def _load_full_session() -> ort.InferenceSession:
    so = ort.SessionOptions()
    if FULL_CTX.exists():
        model_path   = FULL_CTX.as_posix()
        providers    = ["QNNExecutionProvider"]
        prov_options = [{"backend_path": QNN_DLL}]
    else:
        model_path   = FULL_BASE.as_posix()
        providers    = ["QNNExecutionProvider", "CPUExecutionProvider"]
        prov_options = [{"backend_path": QNN_DLL}, {}]
        so.add_session_config_entry("ep.context_enable",     "1")
        so.add_session_config_entry("ep.context_embed_mode", "1")
        so.add_session_config_entry("ep.context_file_path",  FULL_CTX.as_posix())
    return ort.InferenceSession(
        model_path,
        sess_options    = so,
        providers       = providers,
        provider_options= prov_options
    )

# one singleton for the full CLIP
_SESSION = _load_full_session()

print(f"🖼️ Image Session Inputs : {[(inp.name, inp.shape, inp.type) for inp in _SESSION.get_inputs()]}")
print(f"🏷️ Image Session Outputs: {[(out.name, out.shape, out.type) for out in _SESSION.get_outputs()]}")

# ── TOKENIZER (unchanged) ────────────────────────────────────────────
TOKENIZER = CLIPTokenizer.from_pretrained(
    "openai/clip-vit-base-patch32",
    cache_dir=str(Path(__file__).parent.parent / "models"),
    local_files_only=False
)

# inputs = [(inp.name, inp.shape, inp.type) for inp in _IMG_SESS.get_inputs()]
# outputs = [(out.name, out.shape, out.type) for out in _IMG_SESS.get_outputs()]
# print(f"🖼️ Image Session Inputs : {inputs}")
# print(f"🏷️ Image Session Outputs: {outputs}")

# ── L2 helper ─────────────────────────────────────────────────────────
# ── L2 helper ─────────────────────────────────────────────────────────
def _l2(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)

# ── TEXT EMBEDDING ────────────────────────────────────────────────────
def embed_text(text: str) -> np.ndarray:
    """
    Feeds only the text inputs into the full CLIP ONNX and grabs the
    512-D text embedding (now at outputs[2]).
    """
    toks = TOKENIZER(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=77,
    )
    input_ids      = toks["input_ids"].astype(np.int64)
    attention_mask = toks["attention_mask"].astype(np.int64)
    # dummy image to satisfy the ONNX inputs
    dummy_img = np.zeros((1, 3, 224, 224), dtype=np.float32)

    # run everything and pick output index 2 (the pooled text_embeds)
    outputs = _SESSION.run(
        None,
        {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "pixel_values":   dummy_img,
        },
    )
    emb512 = outputs[2][0]   # outputs[2] has shape (1,512)

    return _l2(emb512)


# ── IMAGE EMBEDDING ───────────────────────────────────────────────────
def embed_image(img) -> np.ndarray:
    """
    Feeds only the image input into the full CLIP ONNX and grabs the
    512-D image embedding (now at outputs[3]).
    """
    if isinstance(img, (str, Path)):
        img = Image.open(img)
    arr = np.asarray(
        img.convert("RGB").resize((224, 224), Image.BICUBIC),
        dtype=np.float32
    ) / 255.0
    arr = ((arr.transpose(2, 0, 1) - 0.5) / 0.5)[None]  # (1,3,224,224)

    # dummy text inputs
    dummy_ids  = np.zeros((1, 77), dtype=np.int64)
    dummy_mask = np.zeros((1, 77), dtype=np.int64)

    outputs = _SESSION.run(
        None,
        {
            "pixel_values":   arr,
            "input_ids":      dummy_ids,
            "attention_mask": dummy_mask,
        },
    )
    emb512 = outputs[3][0]   # outputs[3] has shape (1,512)

    return _l2(emb512)


# ── AUDIO EMBEDDING ───────────────────────────────────────────────────
def embed_audio(wav_path: str, *, sr=16_000, sec=5) -> np.ndarray:
    # y, _ = sf.read(wav_path) Fix ARM64 compability for soundfile
    y = None
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = librosa.util.fix_length(y, sec * sr)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=1024, hop_length=320, n_mels=128, fmax=8000
    )
    db   = librosa.power_to_db(mel, ref=np.max)
    norm = (db - db.min()) / (db.max() - db.min() + 1e-9)
    rgb  = np.stack([norm * 255] * 3, -1).astype(np.uint8)
    img  = Image.fromarray(rgb).resize((224, 224), Image.BICUBIC)
    return embed_image(img)
