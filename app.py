import os
import platform
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from rag_core import ensure_index, rebuild_index, retrieve, make_prompt

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

st.set_page_config(page_title="RAG — Deep Guide & Demo (OpenAI-only)", page_icon="📚")

st.title("📚 RAG — Deep Guide & Demo (OpenAI-only)")
st.write("A step-by-step Retrieval-Augmented Generation (RAG) guide with an interactive demo. No local ML deps, no image generation.")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    if st.button("🔁 Rebuild Index"):
        with st.status("Rebuilding index...", expanded=True) as status:
            def cb(msg):
                st.write(msg)
            try:
                rebuild_index(progress_cb=cb)
                status.update(label="Index rebuilt!", state="complete")
            except Exception as e:
                status.update(label="Index rebuild failed", state="error")
                st.exception(e)
    k = st.slider("Top-K chunks", min_value=3, max_value=12, value=5, step=1)
    st.caption("Increase if answers miss context; decrease if answers seem noisy.")

# Client helper (proxy-safe)
def _make_openai_client():
    import httpx
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Create `.env` from `.env`.")
    proxy = os.getenv("OPENAI_PROXY") or os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    kwargs = {"api_key": api_key}
    if proxy:
        kwargs["http_client"] = httpx.Client(proxies=proxy, timeout=60.0)
    return OpenAI(**kwargs)

def call_llm(system_prompt: str, user_prompt: str) -> str:
    client = _make_openai_client()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )
    return resp.choices[0].message.content

tab_deep, tab_demo, tab_diag = st.tabs(["Deep Guide (Step-by-Step)", "Demo: Ask your docs", "Setup & Diagnostics"])

# ------------------
# Deep Guide
# ------------------
with tab_deep:
    st.header("1) Problem RAG solves")
    st.markdown("""
LLMs don't know your private data and can hallucinate. **RAG** fixes this by retrieving
relevant **chunks** from your knowledge base and **grounding** the LLM's answer in them.
    """)

    st.header("2) Ingestion & Chunking")
    st.markdown("""
- **What:** Read files (`.pdf`, `.md`, `.txt`) from `./docs`.
- **Why chunk:** Retrieval works on chunk-level granularity. Good defaults:
  - `max_chars ≈ 700–1500`, `overlap ≈ 10–20%`.
  - Split on whitespace near the limit to avoid cutting sentences.
- **Trade-offs:**
  - Too small → many irrelevant hits; too big → misses specific facts.
  - Overlap smooths boundaries but increases index size.
    """)

    st.header("3) Embeddings & Vector Store")
    st.markdown("""
- **Embedding model:** This project uses **OpenAI embeddings** (`text-embedding-3-small`) to avoid local installs.
- **Vector DB:** **ChromaDB** (local, simple). Stores vectors + metadata (file path, chunk id).
- **Similarity:** cosine distance (Chroma default; we store normalized vectors on provider side).
    """)

    st.header("4) Retrieval")
    st.markdown("""
- Convert the **query** to an embedding.
- Fetch **top-K** most similar chunks (tune `K` in the sidebar).
- Optional: re-rank/filters by metadata (page, section) — omitted here for simplicity.
    """)

    st.header("5) Prompting & Generation")
    st.markdown("""
- **Stuffing pattern:** Concatenate top chunks and instruct the LLM to **only** use the provided context.
- Ask for **sources** (file paths) in the answer for transparency.
- Keep `temperature` low for deterministic, reference-style responses.
    """)

    st.header("6) Evaluation & Hardening")
    st.markdown("""
- **Faithfulness** (citations actually support the claims)
- **Answer Correctness** (accurate & complete)
- **Coverage** (did we retrieve the right chunks?)
- **Latency & Cost** (chunking, `K`, and model choice)
- **Safety & Compliance** (filter PII, redact secrets, validate links)
    """)

    st.header("7) Operations")
    st.markdown("""
- **Cold start:** index empty → run **Rebuild Index**.
- **Updates:** watch folders and re-chunk incrementally (future work).
- **Caching:** memoize embeddings for repeated content.
- **Observability:** log retrieval hits & distances for debugging.
    """)

# ------------------
# Demo
# ------------------
with tab_demo:
    st.header("Ask your documents")
    st.write("Put `.pdf`, `.md`, or `.txt` files in **`./docs`**, then rebuild the index from the sidebar.")
    try:
        ensure_index()
    except Exception as e:
        st.error("Index not ready.")
        st.exception(e)

    query = st.text_input("Your question")
    if st.button("Go", type="primary") and query:
        try:
            with st.spinner("Retrieving..."):
                hits = retrieve(query, k=k)
            if not hits:
                st.warning("No relevant chunks found. Try rebuilding the index or adding more docs.")
            else:
                sys_prompt = "You are a helpful assistant. Prefer concise, citation-backed answers."
                user_prompt = make_prompt(query, hits)
                with st.spinner("Generating answer..."):
                    answer = call_llm(sys_prompt, user_prompt)
                st.subheader("Answer")
                st.write(answer)

                st.subheader("Top Matches")
                for (_, meta, dist) in hits:
                    st.write(f"- **{meta['source']}** (relevance: {1 - dist:.3f})")
        except Exception as e:
            st.error("Query failed.")
            st.exception(e)

# ------------------
# Diagnostics
# ------------------
with tab_diag:
    st.header("Environment & Checks")
    st.write(f"- **OS**: {platform.platform()}")
    st.write(f"- **Python**: {platform.python_version()}")
    st.write(f"- **OPENAI_API_KEY set**: {'✅' if bool(OPENAI_API_KEY) else '❌'}")

    if st.button("Test chat call"):
        try:
            out = call_llm("You are a concise assistant.", "Reply with 'OK' if you can read this.")
            st.success(f"LLM reply: {out}")
        except Exception as e:
            st.error("Chat test failed.")
            st.exception(e)

    if st.button("Test index & retrieve"):
        try:
            ensure_index()
            hits = retrieve("What is the PTO policy?", k=3)
            if hits:
                st.success("Retrieval works. Top hit sources:")
                for _, meta, dist in hits:
                    st.write(f"- {meta['source']} (relevance: {1 - dist:.3f})")
            else:
                st.warning("No hits. Add files to ./docs and rebuild the index.")
        except Exception as e:
            st.error("Index/retrieval test failed.")
            st.exception(e)
