import os
import glob
import uuid
from typing import List, Tuple
from pypdf import PdfReader
from dotenv import load_dotenv
import chromadb
from openai import OpenAI

load_dotenv()

DATA_DIR = os.getenv("RAG_DATA_DIR", "docs")
PERSIST_DIR = os.getenv("RAG_PERSIST_DIR", "chroma")
COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "docs")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# ------------------
# I/O
# ------------------
def read_txt_md(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for i, page in enumerate(reader.pages):
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts)

def load_documents() -> List[dict]:
    docs = []
    for path in glob.glob(os.path.join(DATA_DIR, "**/*"), recursive=True):
        if os.path.isdir(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext in [".txt", ".md"]:
            content = read_txt_md(path)
        elif ext in [".pdf"]:
            content = read_pdf(path)
        else:
            continue
        if content.strip():
            docs.append({"path": path, "text": content})
    return docs

# ------------------
# Chunking (character-based to avoid tiktoken)
# ------------------
def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> List[str]:
    chunks = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + max_chars, n)
        cut = text.rfind(" ", start, end)
        if cut == -1 or cut < start + int(max_chars * 0.6):
            cut = end
        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)
        if cut == n:
            break
        start = max(0, cut - overlap)
    return chunks

# ------------------
# OpenAI client (proxy-safe)
# ------------------
def _make_openai_client() -> OpenAI:
    import httpx
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    proxy = os.getenv("OPENAI_PROXY") or os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    kwargs = {"api_key": api_key}
    if proxy:
        kwargs["http_client"] = httpx.Client(proxies=proxy, timeout=60.0)
    return OpenAI(**kwargs)

def embed_batch(texts: List[str]) -> List[List[float]]:
    client = _make_openai_client()
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

# ------------------
# Vector DB
# ------------------
client = chromadb.PersistentClient(path=PERSIST_DIR)
try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception:
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

def rebuild_index(progress_cb=None) -> Tuple[chromadb.api.models.Collection.Collection, int, int]:
    global collection
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    docs = load_documents()
    if not docs:
        if progress_cb: progress_cb("No documents found in ./docs. Add PDFs, .txt, or .md and rebuild.")
        return collection, 0, 0

    ids, texts, metadatas = [], [], []
    total_chunks = 0
    for d in docs:
        chunks = chunk_text(d["text"])
        for i, ch in enumerate(chunks):
            ids.append(str(uuid.uuid4()))
            texts.append(ch)
            metadatas.append({"source": d["path"], "chunk_id": i})
        total_chunks += len(chunks)

    BATCH = 64
    for i in range(0, len(texts), BATCH):
        batch_texts = texts[i:i+BATCH]
        batch_ids = ids[i:i+BATCH]
        batch_meta = metadatas[i:i+BATCH]
        embs = embed_batch(batch_texts)
        collection.add(documents=batch_texts, ids=batch_ids, embeddings=embs, metadatas=batch_meta)
        if progress_cb:
            progress_cb(f"Indexed {min(i+BATCH, len(texts))}/{len(texts)} chunks...")

    if progress_cb:
        progress_cb(f"Ingested {total_chunks} chunks from {len(docs)} files.")
    return collection, total_chunks, len(docs)

def ensure_index():
    global collection
    try:
        c = collection.count()
        if c == 0:
            rebuild_index()
    except Exception:
        rebuild_index()
    return collection

def retrieve(query: str, k: int = 5):
    q_emb = embed_batch([query])[0]
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    return list(zip(docs, metas, dists))

def make_prompt(question: str, contexts) -> str:
    joined = "\n\n---\n\n".join(
        [f"[Source: {c['source']} | Chunk: {c['chunk_id']}]\n{txt}" for txt, c, _ in contexts]
    )
    user = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say you don't know.

Question: {question}

Context:
{joined}

Then list the sources you used (file paths) at the end under "Sources".
"""
    return user
