"""
rag.py — Core Retrieval-Augmented Generation logic.

Loads the FAISS index, retrieves the most relevant document chunks for a
question, and generates a grounded answer. The generator is chosen automatically
based on which API key is available:

    1. GEMINI_API_KEY  -> Google Gemini 2.5 Flash   (recommended)
    2. GROQ_API_KEY    -> Groq Llama 3.3 70B         (fast fallback)
    3. neither set     -> local Flan-T5             (fully offline)
"""

import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # read the .env file into the environment
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------- configuration ----------
EMBED_MODEL = "all-MiniLM-L6-v2"
BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "index.faiss"
CHUNKS_PATH = BASE_DIR / "chunks.pkl"
TOP_K = 4

DISCLAIMER = (
    "\n\n---\n"
    "ℹ️ This is general information, not medical advice or a diagnosis. "
    "If you are struggling or in crisis, please contact a local crisis line or "
    "emergency services, or speak with a qualified professional."
)

PROMPT_TEMPLATE = """You are a supportive mental-health *information* assistant.
Answer the user's question using ONLY the context below.
If the context does not contain the answer, say you don't have that information
and gently suggest speaking with a qualified professional.
Be warm, clear, and concise. Do not give medical diagnoses or treatment plans.

Context:
{context}

Question: {question}
Answer:"""

# ---------- lazy-loaded resources ----------
_embedder = None
_index = None
_chunks = None


def load():
    """Load the embedder, FAISS index, and chunks once and cache them."""
    global _embedder, _index, _chunks
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    if _index is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"Missing FAISS index: {INDEX_PATH}. Run ingest.py to build it."
            )
        _index = faiss.read_index(str(INDEX_PATH))
    if _chunks is None:
        if not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                f"Missing chunk store: {CHUNKS_PATH}. Run ingest.py to build it."
            )
        with open(CHUNKS_PATH, "rb") as f:
            _chunks = pickle.load(f)
    return _embedder, _index, _chunks


def retrieve(query, k=TOP_K):
    """Return the k most relevant text chunks for a query."""
    embedder, index, chunks = load()
    q = embedder.encode([query], normalize_embeddings=True)
    q = np.array(q, dtype="float32")
    _scores, idx = index.search(q, k)
    return [chunks[i] for i in idx[0] if i != -1]


def build_prompt(question, passages):
    context = "\n\n".join(passages) if passages else "(no relevant context found)"
    return PROMPT_TEMPLATE.format(context=context, question=question)


# ---------- generation providers ----------
def _gen_gemini(prompt):
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    # model = genai.GenerativeModel("gemini-2.5-flash")
    model = genai.GenerativeModel("gemini-3-flash-preview")
    resp = model.generate_content(prompt)
    text = getattr(resp, "text", None)
    if text:
        return text.strip()
    raise RuntimeError("Gemini returned an empty response.")


def _gen_groq(prompt):
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


_flan = None


def _gen_flan(prompt):
    global _flan
    if _flan is None:
        from transformers import pipeline
        _flan = pipeline("text2text-generation", model="google/flan-t5-base")
    return _flan(prompt, max_new_tokens=256)[0]["generated_text"].strip()


def generate(question, k=TOP_K):
    """Retrieve context and produce a grounded, disclaimer-appended answer."""
    passages = retrieve(question, k)
    prompt = build_prompt(question, passages)

    answer = None
    errors = []

    if os.environ.get("GEMINI_API_KEY"):
        try:
            answer = _gen_gemini(prompt)
        except Exception as exc:
            errors.append(f"gemini failed: {exc}")

    if answer is None and os.environ.get("GROQ_API_KEY"):
        try:
            answer = _gen_groq(prompt)
        except Exception as exc:
            errors.append(f"groq failed: {exc}")

    if answer is None:
        try:
            answer = _gen_flan(prompt)
        except Exception as exc:
            errors.append(f"flan failed: {exc}")
            raise RuntimeError("; ".join(errors)) from exc

    return answer + DISCLAIMER


if __name__ == "__main__":
    # quick manual test
    print(generate("How can I manage everyday stress?"))
