import os, sys, json, numpy as np
from text_embedding_pipeline import chunk_text, embed_chunks, create_session

OUTPUT_DIR = "embeddings"

# ------------------------------------------------------------------
# 1) Load index & filenames
# ------------------------------------------------------------------
emb_path = os.path.join(OUTPUT_DIR, "all_embeddings.npy")
file_path = os.path.join(OUTPUT_DIR, "filenames.json")

if not os.path.isfile(emb_path):
    sys.exit(f"❌  {emb_path} not found. Run text_embedding_pipeline.py first.")
if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
    sys.exit(f"❌  {file_path} is missing or empty. Re-run embedding script.")

emb_matrix = np.load(emb_path)
filenames  = json.load(open(file_path))
if len(filenames) != emb_matrix.shape[0]:
    sys.exit("❌  filenames.json length doesn’t match embeddings rows.")

print(f"Loaded {len(filenames)} embeddings  • dim={emb_matrix.shape[1]}")

# ------------------------------------------------------------------
# 2) Query embedding
# ------------------------------------------------------------------
def embed_query(text: str) -> np.ndarray:
    chunks = chunk_text(text)
    q_chunks = embed_chunks(chunks)          # uses output[1] → (n, 384)
    q_chunks /= (np.linalg.norm(q_chunks, axis=1, keepdims=True) + 1e-12)
    q_vec = q_chunks.mean(axis=0)
    return q_vec / (np.linalg.norm(q_vec) + 1e-12)

# ------------------------------------------------------------------
# 3) Search
# ------------------------------------------------------------------
def search(query: str, top_k: int = 5):
    q_vec = embed_query(query)
    if q_vec.shape[0] != emb_matrix.shape[1]:
        raise RuntimeError(
            f"Dimension mismatch: query {q_vec.shape[0]} vs index {emb_matrix.shape[1]}"
        )
    sims = emb_matrix @ q_vec           # cosine (all vectors unit-norm)
    best = np.argsort(-sims)[:top_k]
    return [(filenames[i], float(sims[i])) for i in best]

# ------------------------------------------------------------------
# 4) CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python search_pipeline_txt.py "query text" [top_k]')
        sys.exit(0)
    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    for fname, score in search(query, k):
        print(f"{score: .4f}  {fname}")

# python search_pipeline_txt.py "Daily FloodScan WorldPop Somalia raster methodology" 5
# python search_pipeline_txt.py "NIST vulnerability threat event adverse impact humanitarian"