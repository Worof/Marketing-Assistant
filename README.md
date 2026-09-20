# Marketing Knowledge Assistant

A local, API-key-free RAG chatbot for marketing data science — answers questions
about customer segmentation (RFM), churn risk, and campaign strategy using a
local LLM (Qwen2.5-1.5B-Instruct), retrieval over a PDF strategy doc and a CSV
customer dataset, and full conversation memory.

## Features
- Retrieval-Augmented Generation over PDF + CSV sources
- Local LLM inference — no API keys required
- Buffer + rolling-summary conversation memory, with follow-up question rewriting
- Anti-hallucination: precomputed dataset facts, exact customer-ID lookup, similarity thresholding
- Streamlit chat UI with live token streaming and per-answer source citations
- File upload to swap in your own PDF/CSV knowledge base

## Run locally
\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Run in Colab (GPU)
1. Upload `app.py`, `marketing_strategy.pdf`, `customer_segments.csv` to your Colab session.
2. Install dependencies: `!pip install -r requirements.txt pyngrok`
3. Launch: `!streamlit run app.py --server.port 8501 &`
4. Tunnel with `pyngrok` or Colab's `serve_kernel_port_as_window(8501)`.

## Data
- `customer_segments.csv` — RFM (Recency/Frequency/Monetary) segmentation computed
  from the UCI Online Retail dataset (541,909 real transactions).
- `marketing_strategy.pdf` — retention and campaign strategy playbook.

## Architecture
Retriever: FAISS + `all-MiniLM-L6-v2` embeddings.
Generator: `Qwen/Qwen2.5-1.5B-Instruct`, run locally via `transformers`.
Memory: bounded conversation buffer (recent turns verbatim) + LLM-generated
rolling summary of older turns.
