"""
Extract raw text from PDFs using pdfminer.six (pure-Python; works on Win ARM).
Install with:
    pip install pdfminer.six
"""

from pathlib import Path
from pdfminer.high_level import extract_text

# Maximum number of PDFs to process in one go
MAX_FILES = 50

def extract_texts_from_dir(directory: str) -> dict[str, str]:
    """
    Walk `directory`, find up to MAX_FILES .pdf files,
    and return a dict mapping filename → full extracted text.
    """
    texts: dict[str, str] = {}
    pdf_paths = list(Path(directory).rglob("*.pdf"))[:MAX_FILES]

    for p in pdf_paths:
        try:
            # pdfminer handles password-free PDFs; slow but reliable
            text = extract_text(p)
            if text:
                texts[p.name] = text
        except Exception as e:
            print(f"[extract_texts_from_dir] failed on {p}: {e}")

    return texts
