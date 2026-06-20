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

import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
DOCS_FOLDER = "docs"
CHUNK_SIZE = 180      # words per chunk
CHUNK_OVERLAP = 40    # words shared between neighbouring chunks


def load_texts(folder=DOCS_FOLDER):
    texts = []
    for path in glob.glob(os.path.join(folder, "*.pdf")):
        reader = PdfReader(path)
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                texts.append(t)
    for path in glob.glob(os.path.join(folder, "*.txt")):
        with open(path, encoding="utf-8") as f:
            texts.append(f.read())
    return texts


def chunk(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    out = []
    step = size - overlap
    for i in range(0, len(words), step):
        piece = " ".join(words[i:i + size])
        if piece.strip():
            out.append(piece)
    return out


def main():
    docs = load_texts()
    if not docs:
        raise SystemExit(
            "No documents found in docs/. Add at least one .pdf or .txt file."
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

    faiss.write_index(index, "index.faiss")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print("Saved index.faiss and chunks.pkl")


if __name__ == "__main__":
    main()
