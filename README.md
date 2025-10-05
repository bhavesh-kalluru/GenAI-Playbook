# RAG — Deep Guide & Demo (OpenAI-only, Streamlit)

This project teaches **Retrieval-Augmented Generation (RAG)** step-by-step and lets you query your own documents.
It uses **OpenAI embeddings** + **ChromaDB** and avoids local ML dependencies (no torch/sentence-transformers/tiktoken).
**Image generation is removed.**

## Folder layout
```
rag-deep-openai-only/
├─ docs/              # put your PDFs / .txt / .md here
├─ chroma/            # local vector DB
├─ .env.example       # copy to .env and add your key
├─ requirements.txt
├─ rag_core.py        # ingestion, chunking, embeddings (OpenAI), retrieval, prompting
└─ app.py             # Streamlit app (Deep Guide • Demo • Diagnostics)
```

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt

cp .env .env
# paste your OPENAI_API_KEY; optionally adjust OPENAI_MODEL / OPENAI_EMBED_MODEL

# add a few .md/.txt/.pdf files to ./docs

streamlit run app.py
```

## How it works (concise)
1. **Ingest**: reads files in `./docs` (PDF/MD/TXT).
2. **Chunk**: character-based (~1500 chars, 200 overlap) to avoid tiktoken.
3. **Embed**: calls OpenAI embeddings (`text-embedding-3-small`) to get vectors.
4. **Store**: saves vectors + metadata (file path, chunk id) in ChromaDB.
5. **Retrieve**: embeds user query; finds top-K nearest chunks.
6. **Generate**: passes chunks to a chat model and asks for a citation-backed answer.

## Tuning hints
- **Chunk size/overlap**: start ~1500/200; shorten if answers are too broad, lengthen if context gets cut.
- **Top-K**: start 5; increase for coverage, decrease for precision.
- **Prompt**: keep temperature low (0–0.2) and demand sources.
- **Filtering**: add metadata filters (e.g., restrict by filename/section) for multi-domain corpora.

## Troubleshooting
- **No index / No results**: add docs to `./docs` and click **Rebuild Index**.
- **Behind a proxy**: set `OPENAI_PROXY` or `HTTPS_PROXY`/`HTTP_PROXY` in your environment.
- **Rate limits**: slow down rebuilds or reduce batch size in `rag_core.py`.
