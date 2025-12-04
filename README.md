# Mind Search 🧠

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Arch: ARM64](https://img.shields.io/badge/Arch-ARM64-brightgreen)](https://en.wikipedia.org/wiki/AArch64)
[![Accel: NPU](https://img.shields.io/badge/Accel-NPU-blueviolet)](https://www.qualcomm.com/products/mobile/snapdragon/pcs)

**Mind Search** is a semantic file search tool that finds your files based on *meaning*, not just filenames. It helps you locate documents, images, and audio recordings using vague recollections and natural language queries.

![Mind Search Demo](assets/demo.gif)

## Features

- **🔍 Semantic Search**: Find files by describing their content (e.g., "that contract about AI ethics").
- **⚡ NPU Accelerated**: Optimized for Snapdragon X Elite NPUs using ONNX-quantized models.
- **🖼️ Multi-Modal**: Supports PDFs, images, and audio files.
- **🔒 Local & Private**: All processing happens locally on your device.

> [!NOTE]
> **Hackathon Project**: This project was built for the **RAISE Hackathon (Qualcomm Track)** to demonstrate local semantic search optimized for Snapdragon X Elite NPUs.

## Architecture

Mind Search uses a pipeline to process different file types into a shared latent space.

```mermaid
graph LR
    subgraph Ingest
    A[PDFs]
    B[Images]
    C[Audio]
    end
    
    subgraph Processing
    D[Text Extraction]
    E[Transcription]
    F[Image Norm]
    end
    
    subgraph Embedding
    G[ONNX Encoder]
    H[Latent Projector]
    end
    
    subgraph Search
    I[Vector Index]
    J[Query Engine]
    end

    A --> D
    B --> F
    C --> E
    
    D --> G
    E --> G
    F --> G
    
    G --> H
    H --> I
    I <--> J
```

1.  **Ingest**: Load supported file types.
2.  **Processing**: Extract text, transcribe audio, standardize images.
3.  **Embedding**: Generate vectors using NPU-optimized models.
4.  **Search**: Index vectors for fast nearest-neighbor retrieval.

## Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/egroh/mind_search.git mind-search
    cd mind-search
    ```

2.  **Install dependencies**
    ```bash
    # Recommended: Use a virtual environment
    python3 -m venv .venv
    source .venv/bin/activate
    
    pip install -r requirements.txt
    ```

## Usage

### 1. Index Your Files

```bash
python mind_search/index.py \
  --data-dir ./my_documents \
  --index-file ./index.pkl \
  --model openai-embedding
```

### 2. Search

```bash
python mind_search/query.py \
  --index-file ./index.pkl \
  --query "that contract I reviewed last month about AI ethics" \
  --top-k 5
```

## Contributors

*   [Eddie Groh](https://github.com/egroh)
*   [Vijay Venkatesh M](https://github.com/vijaysr4)

---

*Made with ❤️ at the RAISE Hackathon*
