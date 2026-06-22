# 💚 Mental Health Support Chatbot (RAG)

A Retrieval-Augmented Generation (RAG) chatbot that answers general mental-health
and wellbeing questions using a small collection of trusted documents. Answers are
grounded in retrieved passages to reduce hallucination, and every reply includes a
disclaimer pointing toward professional help.

> ⚠️ For information and education only. Not a diagnostic or treatment tool.

## How it works
1. Documents in `docs/` are split into chunks and embedded with
   `sentence-transformers` (all-MiniLM-L6-v2).
2. The chunks are stored in a **FAISS** vector index (`ingest.py`).
3. At question time, the most relevant chunks are retrieved and passed to a
   language model (Gemini → Groq → offline Flan-T5 fallback) which answers using
   only that context (`rag.py`).
4. A **Gradio** chat interface serves it (`app.py`).

## Project structure
```
.
├── app.py             # Gradio chat UI
├── rag.py             # retrieval + grounded generation
├── ingest.py          # builds index.faiss and chunks.pkl from docs/
├── requirements.txt
├── .env.example       # template for your API key
└── docs/              # your trusted source documents (.pdf / .txt)
```

## Setup
```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your free API key
cp .env.example .env             # then edit .env and paste your Gemini key
```
Get a free Gemini key at https://aistudio.google.com (no credit card).

## Build the index
Put your documents in `docs/`, then run:
```bash
python ingest.py
```
This creates `index.faiss` and `chunks.pkl`.

## Run the app
```bash
# load the key from .env into your shell, then start the app
export $(grep -v '^#' .env | xargs)   # Windows PowerShell: see note below
python app.py
```
Open the printed local URL (usually http://127.0.0.1:7860).

> Windows PowerShell key load:
> `Get-Content .env | Where-Object {$_ -notmatch '^#'} | ForEach-Object { $p=$_.Split('='); [Environment]::SetEnvironmentVariable($p[0],$p[1]) }`

## Deploy (optional, free)
1. Create a free Space at https://huggingface.co/spaces (SDK: **Gradio**).
2. Upload `app.py`, `rag.py`, `requirements.txt`, `index.faiss`, `chunks.pkl`.
3. In **Settings → Variables and secrets**, add `GEMINI_API_KEY`.
4. The Space builds automatically and gives you a public demo URL.

## ~Sources and Reused Components

Gradio framework

FAISS vector database

Sentence Transformers

Groq API / Llama model

Hugging Face Spaces deployment

WHO and CBT source materials

## ~Challenges Encountered

Team GitHub access and permissions

Dependency management and package compatibility

Retrieval quality tuning

## ~Deployment configuration

Knowledge base preparation and chunking

Known Limitations

Educational use only

Not intended for crisis intervention

Limited by source document coverage

Response quality depends on retrieval accuracy

## ~Execution Status

The application executes successfully locally and on Hugging Face Spaces.

Public deployment available at: [link]

## Responsible AI
The bot is scoped to general psychoeducation, answers only from retrieved
sources, declines out-of-scope questions, and signposts professional and crisis
support rather than attempting to handle emergencies.

## AI disclosure
Portions of this project were developed with AI assistance and reviewed before use.

## Public deployment available at: https://huggingface.co/spaces/ggebreselassie29144/mental-health-chatbot
