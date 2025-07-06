import os
import numpy as np
import json
import onnxruntime as ort
from transformers import AutoTokenizer
from text_extract_pdf import extract_texts_from_dir
from tqdm import tqdm

# Configuration
PDF_DIR         = 'Pdf'
OUTPUT_DIR      = 'embeddings'
TOKENIZER_DIR   = 'all-MiniLM-L6-v2-onnx'
ONNX_MODEL_PATH = 'all-MiniLM-L6-v2-onnx/model.onnx'

BATCH_SIZE  = 8
MAX_LENGTH  = 256
STRIDE      = 128
MAX_FILES   = 100

# Ensure output directory exists and set transformers offline
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Load tokenizer and initialize session
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
session = None

def create_session(model_path=ONNX_MODEL_PATH):
    global session
    if session is None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads    = os.cpu_count()
        opts.inter_op_num_threads    = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            opts.append_execution_provider('NNAPI', {'use_arena': True})
        except Exception:
            pass
        session = ort.InferenceSession(model_path, sess_options=opts)
    return session


def chunk_text(text: str) -> list[list[int]]:
    """
    Tokenize and split text into overlapping chunks of fixed length.
    """
    tokens = tokenizer.encode(text, add_special_tokens=True)
    chunks = []
    for i in range(0, len(tokens), STRIDE):
        window = tokens[i:i + MAX_LENGTH]
        if len(window) < MAX_LENGTH:
            window += [tokenizer.pad_token_id] * (MAX_LENGTH - len(window))
        chunks.append(window)
        if i + MAX_LENGTH >= len(tokens):
            break
    return chunks


def embed_chunks(chunks: list[list[int]]) -> np.ndarray:
    sess = create_session()
    name_ids, name_mask, name_tt = [x.name for x in sess.get_inputs()]
    embs = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = np.array(chunks[i:i+BATCH_SIZE], dtype=np.int64)
        mask  = (batch != tokenizer.pad_token_id).astype(np.int64)
        tt    = np.zeros_like(batch)
        # --------------- take the SECOND output -----------------
        out = sess.run(None, {name_ids: batch, name_mask: mask, name_tt: tt})[1]
        # --------------------------------------------------------
        embs.append(out)              # (batch, 384)
    return np.vstack(embs)            # (n_chunks, 384)


if __name__ == '__main__':
    # 1) Extract up to MAX_FILES texts
    texts = extract_texts_from_dir(PDF_DIR)
    texts = dict(list(texts.items())[:MAX_FILES])

    filenames  = []
    embeddings = []

    # 2) Embed each document
    for fname, text in tqdm(texts.items(), desc='Embedding PDFs', unit='file'):
        chunks = chunk_text(text)
        if not chunks:
            continue
        chunk_embs = embed_chunks(chunks)
        # Normalize and mean-pool
        chunk_embs /= (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-12)
        doc_emb = chunk_embs.mean(axis=0)
        doc_emb /= (np.linalg.norm(doc_emb) + 1e-12)
        filenames.append(fname)
        embeddings.append(doc_emb)

    # 3) Save matrix and metadata
    emb_matrix = np.vstack(embeddings)
    np.save(os.path.join(OUTPUT_DIR, 'all_embeddings.npy'), emb_matrix)
    with open(os.path.join(OUTPUT_DIR, 'filenames.json'), 'w') as f:
        json.dump(filenames, f)

    print(f"Saved {len(filenames)} embeddings (dim={emb_matrix.shape[1]}) to '{OUTPUT_DIR}'")

