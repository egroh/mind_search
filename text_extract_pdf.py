import os
import fitz  # PyMuPDF
from tqdm import tqdm

# Maximum number of PDFs to process
MAX_FILES = 50

# Function to extract text from PDFs in a directory (up to MAX_FILES)
def extract_texts_from_dir(directory: str) -> dict[str, str]:
    texts = {}
    # Gather all PDF paths
    pdf_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_paths.append(os.path.join(root, file))

    # Limit to the first MAX_FILES PDFs
    pdf_paths = pdf_paths[:MAX_FILES]

    # Iterate with progress bar
    for path in tqdm(pdf_paths, desc="Reading PDFs", unit="file", leave=True):
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        texts[os.path.basename(path)] = text
    return texts
#
# if __name__ == '__main__':
#     # Directory containing PDF files; specify your path here
#     PDF_DIR = 'Pdf'
#
#     # Extract texts (up to MAX_FILES) with progress bar
#     texts = extract_texts_from_dir(PDF_DIR)
#     if not texts:
#         print(f"No PDF files found in directory: {PDF_DIR}")
#         exit(0)
#
#     # Print the first file's text
#     first_fname = next(iter(texts))
#     print(f"--- Text for {first_fname} ---\n")
#     print(texts[first_fname])
