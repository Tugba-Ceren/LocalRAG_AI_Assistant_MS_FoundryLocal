# LocalRAG_AI_ Assistant_MS_FoundryLocal
Microsoft Turkiye Summer Internship 2026 AI Project
# SpotterAI Local RAG Assistant

An offline, privacy-focused Retrieval-Augmented Generation (RAG) system built using Python, SQLite vector storage, and Microsoft Foundry Local SDK models (`qwen3-embedding-0.6b` and `qwen2.5-0.5b`).

## 📌 Project Overview & Purpose
SpotterAI Assistant provides fast, local Q&A over hardware documentation, SDK specs, and security event logs without sending sensitive data to external cloud APIs.

### Key Capabilities
- **Local Vector Search:** SQLite storage with Cosine Similarity search over embedded document chunks.
- **Strict Grounding Guardrails:** Programmatic similarity thresholding (`top_score < 0.48`) to prevent out-of-scope hallucinations and enforce strict fallback messaging.
- **Low-Latency Inference:** Fully local processing optimized for lightweight 0.5B parameters local language models.

---

## 🛠️ Architecture & Design Decisions
1. **Embedding & Retrieval:** Ingests technical documentation using `qwen3-embedding-0.6b` into an SQLite vector store.
2. **Deterministic Fallback Interception:** Bypasses LLM generation when vector match confidence is low, providing sub-0.1s out-of-scope response times.
3. **Generative Synthesis:** Synthesizes single-chunk context via `qwen2.5-0.5b` chat completions constrained to 1–2 sentences for quick CPU inference.

---

## 🚀 Environment Setup & Installation

### Prerequisites
- Python 3.10+
- Microsoft Foundry Local Manager SDK installed and configured

### 1. Clone & Set Up Virtual Environment
```bash
git clone <your-repository-url>
cd AI_Assistant_MS
python -m venv venv
venv\Scripts\activate