Hi, I’m **Bhavesh** — a GenAI practitioner with **5 years of experience** building ML/AI features end-to-end. I care about **reliable retrieval**, **measurable quality**, **safety**, and **shipping to production**.

---

## What I do
- **RAG systems**: ingestion → chunking → embeddings → vector store → retrieval → grounded answers with citations
- **Quality & safety**: eval harnesses (faithfulness, answer quality), prompt tests, PII redaction, guardrails
- **LLM apps**: chat UX, tools/agents, function calling, cost/latency tuning
- **Ops**: tracing, observability, A/B tests, rollout playbooks

---

## Tech I use
- **Python**, FastAPI, Streamlit
- **OpenAI API** (chat + embeddings), function calling
- Vector stores: **Chroma**, Pinecone
- Pipelines: LangChain / LlamaIndex (when helpful), or lightweight DIY
- Infra: Docker, CI/CD, basic K8s & cloud (AWS/GCP/Azure)
- Evals: Ragas/DeepEval patterns, offline A/B, golden sets

---

## Portfolio highlights (links)
- **RAG Mini (OpenAI-only)** — minimal, production-minded RAG reference  
  `./projects/rag-mini/` (ingestion, retrieval, citations, Streamlit demo)
- **LLM Evaluation Harness** — golden sets, faithfulness checks, cost/latency tracking  
  `./projects/eval-harness/`
- **Agentic Workflows** — tool use (search/db), safety filters, retry/backoff  
  `./projects/agentic-tools/`

> Tip: Replace the paths above with your public repos / demos (or keep this structure and push later).

---

## Starter structure
genai-playbook-bhavesh/
├─ projects/
│ ├─ rag-mini/ # end-to-end RAG reference (OpenAI embeddings + Chroma)
│ ├─ eval-harness/ # reusable evaluation scripts and golden sets
│ └─ agentic-tools/ # examples of function/tool calling patterns
├─ notebooks/ # quick experiments, ablations
├─ docs/ # design notes, diagrams, writeups
└─ README.md

markdown
Copy code

---

## “How I approach RAG” (condensed)
1. **Ingest & Normalize**: PDFs/MD/TXT; track metadata (file, section, page).
2. **Chunking**: size ~700–1500 chars, overlap 10–20%; avoid cutting mid-sentence.
3. **Embeddings**: pick a stable model; monitor drift if you swap.
4. **Store**: vector DB with metadata for filtering.
5. **Retrieval**: top-K + (optional) re-ranking; minimize prompt bloat.
6. **Generation**: “use-context-only” instruction; include **citations**.
7. **Evaluate**: faithfulness, answer quality, coverage, latency, cost.
8. **Operate**: tracing, redaction, safety filters, caching, rollback plan.

---

## Contact & Availability
- **Location**: USA — open to **full-time** roles
- **Email**: _your.email@domain.com_  
- **LinkedIn**: _https://www.linkedin.com/in/your-handle_  
- **Calendly** (optional): _https://calendly.com/your-handle/intro_

> I’m ready to help teams ship practical GenAI features with measurable quality.

---
