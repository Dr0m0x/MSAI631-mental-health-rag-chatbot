"""
ingest.py — Build the knowledge index (run once, and again whenever you
add or change documents in the docs/ folder).

Reads every .pdf and .txt file in docs/, splits them into overlapping chunks,
embeds each chunk, and saves:
    index.faiss   (the vector index)
    chunks.pkl    (the raw text for each vector)
"""

import os
import glob
import pickle
from pathlib import Path

import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
BASE_DIR = Path(__file__).resolve().parent
DOCS_FOLDER = BASE_DIR / "docs"
CHUNK_SIZE = 180      # words per chunk
CHUNK_OVERLAP = 40    # words shared between neighbouring chunks


def load_texts(folder=DOCS_FOLDER):
    texts = []
    folder = Path(folder)
    for path in glob.glob(os.path.join(str(folder), "*.pdf")):
        reader = PdfReader(path)
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                texts.append(t)
    for path in glob.glob(os.path.join(str(folder), "*.txt")):
        with open(path, encoding="utf-8") as f:
            texts.append(f.read())
    return texts


def chunk(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if overlap >= size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
    words = text.split()
    out = []
    step = size - overlap
    for i in range(0, len(words), step):
        piece = " ".join(words[i:i + size])
        if piece.strip():
            out.append(piece)
    return out


def main():
    DOCS_FOLDER.mkdir(parents=True, exist_ok=True)
    docs = load_texts()
    if not docs:
        raise SystemExit(
            f"No documents found in {DOCS_FOLDER}. Add at least one .pdf or .txt file."
        )

    chunks = []
    for d in docs:
        chunks.extend(chunk(d))
    print(f"Loaded {len(docs)} document(s) -> {len(chunks)} chunks")

    embedder = SentenceTransformer(EMBED_MODEL)
    emb = embedder.encode(chunks, normalize_embeddings=True, show_progress_bar=True)
    emb = np.array(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])  # cosine similarity (normalized vectors)
    index.add(emb)

    index_path = BASE_DIR / "index.faiss"
    chunks_path = BASE_DIR / "chunks.pkl"
    faiss.write_index(index, str(index_path))
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved {index_path} and {chunks_path}")


if __name__ == "__main__":
    main()
