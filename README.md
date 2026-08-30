# LocalRAG_AI_ Assistant_MS_FoundryLocal
Microsoft Turkiye Summer Internship 2026 AI Project
# SpotterAI Local RAG Assistant

An offline, Retrieval-Augmented Generation (RAG) system built using Python, SQLite vector storage, and Microsoft Foundry Local SDK models (`qwen3-embedding-0.6b` and `qwen2.5-0.5b`).

## 📌 Project Overview & Purpose
SpotterAI Assistant provides fast, local Q&A over software documentation, SDK specs, and security event logs without sending sensitive data to external cloud APIs.

### Key Capabilities
- **Local Vector Search:** SQLite storage with Cosine Similarity search over embedded document chunks.
- **Strict Grounding Guardrails:** Programmatic similarity thresholding (`top_score < 0.48`) to prevent out-of-scope hallucinations and enforce strict fallback messaging.
- **Low-Latency Inference:** Fully local processing optimized for lightweight 0.5B parameters local language models.



## 🛠️ Architecture & Design Decisions
1. **Embedding & Retrieval:** Ingests technical documentation using `qwen3-embedding-0.6b` into an SQLite vector store.
2. **Deterministic Fallback Interception:** Bypasses LLM generation when vector match confidence is low, providing sub-0.1s out-of-scope response times.
3. **Generative Synthesis:** Synthesizes single-chunk context via `qwen2.5-0.5b` chat completions constrained to 1–2 sentences for quick CPU inference.


### Core Execution Flow
1. **Document Ingestion:** Local technical documentation is chunked, embedded using `qwen3-embedding-0.6b`, and indexed in an SQLite vector table alongside source document metadata.
2. **Dense Vector Search:** User queries are converted into vectors and matched against document vectors using dot-product cosine similarity.
3. **Deterministic Guardrail Filtering:** If the highest similarity score falls below `0.48`, the system intercepts the execution path directly in Python, bypassing the LLM to return an instant fallback message.
4. **Grounded Synthesis:** For high-confidence matches, the retrieved chunk context is injected into a strict prompt template evaluated by `qwen2.5-0.5b`.



## 💡 Key Design Decisions & Limitations

### Design Decisions
* **Single-Chunk Context Window (`top_k=2`):** Restricting retrieval to a single top-ranking document chunk cuts input prefill processing overhead by ~50%, preventing CPU hardware slowdowns.
* **Code-Level Fallback Interception:** Ultra-small language models (0.5B parameters) frequently ignore soft prompt fallback rules when encountering partially related words. Intercepting queries in Python before invoking the model eliminates out-of-scope hallucinations (e.g., generating fake camera prices) and drops exit latency to **<0.05 seconds**.
* **Strict Token Generation Constraints:** System prompts strictly limit output length to 1–2 short sentences, reducing CPU generation latency from 18+ seconds down to **1.5–3 seconds**.

### System Limitations
* **CPU Latency Bounds:** Multi-sentence generation on CPU hardware averages ~7–12 seconds per query. 
* **Strict Threshold Cutoffs:** Setting a strict cosine similarity cutoff (`0.48`) prioritizes hallucination prevention over handling broadly or vaguely phrased queries.

---

## 💻 Environment Setup Guide

### Prerequisites
* **Operating System:** Windows 10/11, macOS, or Linux
* **Python Version:** Python 3.10 or higher
* **Runtime:** Microsoft Foundry Local Manager SDK configured on your workstation

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone [https://github.com/your-username/AI_Assistant_MS.git](https://github.com/your-username/AI_Assistant_MS.git)
cd AI_Assistant_MS

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

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

### Source 
https://learn.microsoft.com/en-us/azure/foundry-local/tutorials/tutorial-build-rag-app?tabs=windows