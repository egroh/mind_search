# Mind Search - Raise Hackathon - Qualcomm Track

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.11%2B-orange)](#)
![Arch: ARM64](https://img.shields.io/badge/Arch-ARM64-brightgreen)  
![Accel: NPU](https://img.shields.io/badge/Accel-NPU-blue)

---

A semantic file search tool that finds your files, even when you can’t remember their names. It works by understanding the context and content in your own vague recollections.

**Optimized for** ARM64 on Snapdragon X Elite with NPU acceleration.


![Title GIF](demo.gif)

---

## ![SECTION](https://img.shields.io/badge/SECTION-Installation-blue)  

```bash
# Clone the repo
git clone https://github.com/egroh/hackathon_raise.git mind-search
cd mind-search

# (Optional) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Index Your Files

```bash
# Index a folder of mixed files (pdf, images, audio)
python mind_search/index.py \
  --data-dir ./my_documents \
  --index-file ./index.pkl \
  --model openai-embedding
```

This will:

1. Traverse `./my_documents` for supported file types.
2. Generate embeddings for each file.
3. Store vectors and metadata in `index.pkl`.

### 2. Query Your Files

```bash
python mind_search/query.py \
  --index-file ./index.pkl \
  --query "that contract I reviewed last month about AI ethics" \
  --top-k 5
```

Output:

```
1. /my_documents/contracts/ai_ethics_contract.pdf (score: 0.92)
2. /my_documents/notes/ethics_meeting_recording.mp3 (score: 0.87)
3. /my_documents/images/ai_conference_slide.png (score: 0.84)
…  
```

---

## Architecture

```plain
+------------------+      +-----------------------+      +----------------+
| 1. Data Ingest   | ---> | 2. Preprocessing      | ---> | 3. Modality-    |
|  - PDFs          |      |  - Text extraction    |      |    Specific     |
|  - Images        |      |  - Audio transcription|      |    Encoders     |
|  - Audio Files   |      |                       |      |  (ONNX-         |
+------------------+      +-----------------------+      |   Quantized     |
                                                         |   for NPU accel)|
                                                         +-----------------+
                                                              v
+---------------------+      +-----------------------+      +----------------+
| 4. Joint Embedding  | ---> | 5. Vector Indexing    | ---> | 6. Query Encode |
|    Projector        |      |  - FAISS / Annoy      |      |  & Similarity   |
|  - Align modality   |      |  - HNSW / Milvus      |      |  - Cosine sim   |
|    vectors in shared|      |                       |      +----------------+
|    latent space     |
+---------------------+
```

1. **Data Ingest:** Load PDFs, images, and audio files.
2. **Preprocessing:** Extract raw text, convert speech to text, and standardize images.
3. **Modality-Specific Encoders:** Generate embeddings using ONNX-quantized models optimized for Snapdragon X Elite NPU.
4. **Joint Embedding Projector:** Align modality embeddings into a unified latent space.
5. **Vector Indexing:** Store vectors in FAISS (flat/IVF), Annoy, or HNSW for rapid nearest-neighbor retrieval.
6. **Query Encode & Similarity:** Embed user queries into the same space and compute cosine similarity to retrieve top-K files.


---

## Contributors

* [Eddie Groh](https://github.com/egroh)
* [Vijay Venkatesh M](https://github.com/vijaysr4)

---

*Made with ❤️ at Hackathon Raise*

