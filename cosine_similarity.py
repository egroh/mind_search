#!/usr/bin/env python3

import sys
from pathlib import Path

import fitz
import numpy as np
from light_embed import TextEmbedding
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()
DEMO_FOLDER = SCRIPT_DIR / "sample_docs"


def read_pdf_pymupdf(file_path: str) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    text = []
    with fitz.open(file_path) as pdf_doc:
        for page in pdf_doc:
            text.append(page.get_text())
    return "\n".join(text)


def read_pdfs_in_directory(directory: str, limit: int = None) -> dict:
    """
    Recursively read all PDF files in a directory and extract their text.

    Args:
        directory (str): Path to the target directory.
        limit (int, optional): Max number of PDFs to read.

    Returns:
        dict: A dictionary mapping file paths to their extracted text.
    """
    dir_path = Path(directory).resolve()
    pdf_files = list(dir_path.rglob("*.pdf"))
    if limit:
        pdf_files = pdf_files[:limit]

    pdf_texts = {}
    for pdf_file in tqdm(pdf_files, desc="Reading PDFs", unit="file"):
        try:
            text = read_pdf_pymupdf(str(pdf_file))
            pdf_texts[str(pdf_file)] = text
        except Exception as e:
            print(f"Failed to read {pdf_file}: {e}")

    return pdf_texts


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        raise ValueError(
            "One of the vectors is zero-length; cannot compute cosine similarity."
        )

    return np.dot(v1, v2) / (norm_v1 * norm_v2)


def main(user_query: str, num_docs: int = 10):
    # ------------------------------------------------------------
    # 1) Initialize embedding model
    # ------------------------------------------------------------
    model = TextEmbedding("onnx-models/all-MiniLM-L6-v2-onnx")

    # ------------------------------------------------------------
    # 2) Read PDFs
    # ------------------------------------------------------------
    pdf_texts = read_pdfs_in_directory(DEMO_FOLDER, limit=num_docs)

    if not pdf_texts:
        print("No PDFs found.")
        sys.exit(1)

    # ------------------------------------------------------------
    # 3) Encode PDFs
    # ------------------------------------------------------------
    print("Encoding PDFs...")
    pdf_embeddings = model.encode(list(pdf_texts.values()))

    # ------------------------------------------------------------
    # 4) Encode user query
    # ------------------------------------------------------------
    user_embedding = model.encode(user_query)

    # ------------------------------------------------------------
    # 5) Compute similarities
    # ------------------------------------------------------------
    print("Computing similarities...")
    similarities = [
        (file_path, cosine_similarity(user_embedding, pdf_embedding))
        for file_path, pdf_embedding in zip(pdf_texts.keys(), pdf_embeddings)
    ]

    # ------------------------------------------------------------
    # 6) Sort and display results
    # ------------------------------------------------------------
    ranked = sorted(similarities, key=lambda x: x[1], reverse=True)

    print("\nRanked PDF files by similarity:\n")
    for file_path, score in ranked:
        print(f"{score:.4f} - {file_path}")


if __name__ == "__main__":
    user_query = (
        "This guidance note by the Centre for Humanitarian Data outlines the principles, "
        "challenges, and recommendations for ensuring safe, ethical, and effective management "
        "of humanitarian data, emphasizing the need for clear codes of conduct, staff capacity "
        "to handle ethical dilemmas, and regular ethical audits to address concerns such as "
        "fairness, bias, transparency, privacy, and accountability in data-driven humanitarian action."
    )
    num_docs = 10
    main(user_query, num_docs)
