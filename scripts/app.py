import os
import re
import threading
import tempfile

import pandas as pd
import streamlit as st
import torch

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

# ============================================================
# CONFIG
# ============================================================
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_PDF_PATH = "marketing_strategy.pdf"
DEFAULT_CSV_PATH = "customer_segments.csv"

DISTANCE_THRESHOLD = 1.3
DEFAULT_MAX_NEW_TOKENS = 160
PRONOUN_PAT = re.compile(r"\b(they|them|their|it|that|those|these|this|he|she)\b", re.I)
AGG_PAT = re.compile(r"\b(how many|count|total|average|mean|median|most|highest|lowest|percent)\b", re.I)

st.set_page_config(page_title="Marketing Knowledge Assistant", page_icon="📊", layout="wide")


# ============================================================
# CACHED RESOURCES
# ============================================================
@st.cache_resource(show_spinner=False)
def load_embedder():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})


@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    return model, tokenizer


# ============================================================
# DATA LOADING
# ============================================================
def load_pdf_docs_from_path(path):
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


def load_pdf_docs_from_upload(uploaded_pdf):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        tmp_path = tmp.name
    try:
        return load_pdf_docs_from_path(tmp_path)
    finally:
        os.unlink(tmp_path)


def load_csv_docs_from_df(df):
    docs = []
    for _, row in df.iterrows():
        text = ", ".join(f"{col}={row[col]}" for col in df.columns)
        docs.append(Document(page_content=text, metadata={"source": "csv"}))
    return docs


def build_data_facts(df):
    """Precompute real aggregate statistics so the model never guesses counts/averages."""
    lines = [f"TOTAL CUSTOMERS IN DATASET: {len(df)}"]

    if "Segment" in df.columns:
        lines.append("CUSTOMER COUNT BY SEGMENT:")
        for seg, count in df["Segment"].value_counts().items():
            pct = 100 * count / len(df)
            lines.append(f"  - {seg}: {count} customers ({pct:.1f}%)")

    if "ChurnRisk" in df.columns:
        lines.append("CUSTOMER COUNT BY CHURN RISK:")
        for risk, count in df["ChurnRisk"].value_counts().items():
            pct = 100 * count / len(df)
            lines.append(f"  - {risk} risk: {count} customers ({pct:.1f}%)")

    if "Monetary" in df.columns:
        lines.append(
            f"MONETARY VALUE: total={df['Monetary'].sum():.2f}, "
            f"mean={df['Monetary'].mean():.2f}, median={df['Monetary'].median():.2f}, "
            f"min={df['Monetary'].min():.2f}, max={df['Monetary'].max():.2f}"
        )
        if "Segment" in df.columns:
            lines.append("AVERAGE MONETARY VALUE BY SEGMENT:")
            avg = df.groupby("Segment")["Monetary"].mean().sort_values(ascending=False)
            for seg, val in avg.items():
                lines.append(f"  - {seg}: {val:.2f}")

    for col in ("Recency", "Frequency"):
        if col in df.columns:
            lines.append(
                f"{col.upper()}: mean={df[col].mean():.2f}, median={df[col].median():.2f}, "
                f"min={df[col].min()}, max={df[col].max()}"
            )

    if "Country" in df.columns:
        top = df["Country"].value_counts().head(5)
        lines.append("TOP 5 COUNTRIES BY CUSTOMER COUNT:")
        for country, count in top.items():
            lines.append(f"  - {country}: {count}")

    facts_text = "\n".join(lines)
    facts_docs = [
        Document(page_content=chunk, metadata={"source": "computed_facts"})
        for chunk in facts_text.split("\n\n")
    ]
    return facts_text, facts_docs


def build_index(pdf_docs, csv_docs, facts_docs, embedder):
    all_docs = pdf_docs + csv_docs + facts_docs
    if not all_docs:
        return None
    return FAISS.from_documents(all_docs, embedder)


def exact_customer_lookup(question, df):
    """Pull an exact row straight from the DataFrame instead of relying on similarity search."""
    if df is None or "CustomerID" not in df.columns:
        return None
    ids = re.findall(r"\b\d{4,8}\b", str(question))
    if not ids:
        return None
    blocks = []
    for raw_id in ids:
        try:
            cid = int(raw_id)
        except ValueError:
            continue
        match = df[df["CustomerID"] == cid]
        if not match.empty:
            row = match.iloc[0]
            blocks.append("EXACT DATABASE RECORD: " + ", ".join(f"{c}={row[c]}" for c in df.columns))
        else:
            blocks.append(f"EXACT DATABASE RECORD: no customer with ID {cid} exists in the dataset.")
    return "\n".join(blocks) if blocks else None


# ============================================================
# GENERATION
# ============================================================
def stream_generate(model, tokenizer, system_prompt, user_prompt, placeholder, max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        repetition_penalty=1.05,
        use_cache=True,
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id,
    )

    error_holder = {}

    def run_generation():
        try:
            with torch.no_grad():
                model.generate(**generation_kwargs)
        except Exception as e:
            error_holder["error"] = e
            streamer.end()

    thread = threading.Thread(target=run_generation)
    thread.start()

    partial_text = ""
    for token_chunk in streamer:
        partial_text += token_chunk
        placeholder.markdown(partial_text + "▌")
    thread.join()

    if "error" in error_holder:
        msg = f"⚠️ Generation failed: {error_holder['error']}"
        placeholder.markdown(msg)
        return msg

    placeholder.markdown(partial_text)
    return partial_text.strip()


def generate_plain(model, tokenizer, system_prompt, user_prompt, max_new_tokens=60):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return text.strip()


# ============================================================
# MEMORY — bounded buffer + rolling summary, skipped when not needed
# ============================================================
SUMMARY_SYSTEM_PROMPT = """You compress conversation history. Given an existing summary and
new exchanges, produce an updated summary in under 80 words. Keep concrete details:
customer IDs discussed, segments, decisions. No preamble."""

CONDENSE_SYSTEM_PROMPT = """Rewrite the user's latest question into a standalone search query
that makes sense without the conversation. Resolve pronouns like "they", "them", "that customer"
using the history. Output ONLY the rewritten query, nothing else."""


def format_chat_history(messages, max_turns):
    trimmed = messages[-(max_turns * 2):]
    lines = [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in trimmed]
    return "\n".join(lines)


def update_rolling_summary(model, tokenizer, old_summary, dropped_messages):
    if not dropped_messages:
        return old_summary
    dropped_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in dropped_messages
    )
    prompt = f"Existing summary:\n{old_summary or '(none yet)'}\n\nNew exchanges:\n{dropped_text}\n\nUpdated summary:"
    try:
        return generate_plain(model, tokenizer, SUMMARY_SYSTEM_PROMPT, prompt, max_new_tokens=100)
    except Exception:
        return old_summary


def condense_question(model, tokenizer, question, history_text):
    if not history_text.strip() or not PRONOUN_PAT.search(question):
        return question
    prompt = f"Conversation history:\n{history_text}\n\nLatest question: {question}\n\nStandalone query:"
    try:
        rewritten = generate_plain(model, tokenizer, CONDENSE_SYSTEM_PROMPT, prompt, max_new_tokens=60)
        rewritten = rewritten.strip().strip('"').split("\n")[0]
        return rewritten if 3 < len(rewritten) < 300 else question
    except Exception:
        return question


# ============================================================
# RAG PIPELINE
# ============================================================
RAG_SYSTEM_PROMPT = """You are a marketing data assistant for a retail business.

STRICT RULES:
1. Answer ONLY from DATASET FACTS, EXACT DATABASE RECORD, and RETRIEVED CONTEXT provided.
2. If the answer isn't in what you were given, reply exactly: "That isn't in my knowledge base."
   Never guess or invent numbers, names, customers or policies.
3. For counts, totals and averages, use ONLY DATASET FACTS. Never infer a total from a few examples.
4. For a specific customer, use ONLY the EXACT DATABASE RECORD.
5. Use conversation history only to understand references, never as a source of facts.
6. Be concise: two or three sentences unless asked for more."""


def answer_question(question, vectorstore, df, facts_text, model, tokenizer,
                    history_text, summary_text, placeholder, k=3,
                    use_threshold=True, condense=True, max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
    search_query = condense_question(model, tokenizer, question, history_text) if condense else question

    scored = vectorstore.similarity_search_with_score(search_query, k=k)
    kept = [(d, s) for d, s in scored if s <= DISTANCE_THRESHOLD] if use_threshold else scored

    exact_record = exact_customer_lookup(question, df)

    if not kept and not exact_record:
        msg = "That isn't in my knowledge base."
        placeholder.markdown(msg)
        return msg, [], search_query

    context = "\n\n".join(f"- {d.page_content}" for d, _ in kept) or "(no relevant passages found)"

    prompt_parts = []
    if summary_text:
        prompt_parts.append(f"EARLIER CONVERSATION SUMMARY:\n{summary_text}")
    if history_text:
        prompt_parts.append(f"RECENT CONVERSATION:\n{history_text}")
    if facts_text and AGG_PAT.search(question):
        prompt_parts.append(f"DATASET FACTS (authoritative, precomputed):\n{facts_text}")
    if exact_record:
        prompt_parts.append(exact_record)
    prompt_parts.append(f"RETRIEVED CONTEXT:\n{context}")
    prompt_parts.append(f"QUESTION: {question}")

    answer = stream_generate(
        model, tokenizer, RAG_SYSTEM_PROMPT, "\n\n".join(prompt_parts), placeholder,
        max_new_tokens=max_new_tokens,
    )
    sources = [{"text": d.page_content, "score": float(s), "origin": d.metadata.get("source", "pdf")}
               for d, s in kept]
    return answer, sources, search_query


# ============================================================
# SESSION STATE
# ============================================================
for key, default in [
    ("messages", []),
    ("vectorstore", None),
    ("df", None),
    ("facts_text", None),
    ("csv_preview", None),
    ("kb_summary", None),
    ("summary_text", ""),
    ("summarised_upto", 0),
    ("qa_cache", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("📊 Knowledge Sources")
    st.caption("Uses the bundled marketing strategy PDF and customer segments CSV by default. Upload your own to override.")

    pdf_file = st.file_uploader("Marketing strategy (PDF)", type=["pdf"])
    csv_file = st.file_uploader("Customer segmentation (CSV)", type=["csv"])
    use_defaults = st.checkbox("Use bundled default files if nothing uploaded", value=True)

    build_clicked = st.button("🔨 Build / Rebuild Knowledge Base", width="stretch")

    st.divider()
    st.subheader("Retrieval")
    top_k = st.slider("Chunks retrieved per question (k)", 2, 10, 3)
    use_threshold = st.toggle("Filter weak matches (anti-hallucination)", value=True)
    show_sources = st.toggle("Show retrieved sources", value=True)

    st.divider()
    st.subheader("Memory")
    max_turns = st.slider("Buffer window (turns kept verbatim)", 2, 12, 6)
    use_summary = st.toggle("Summarise older turns", value=True)
    condense = st.toggle("Rewrite follow-up questions", value=True)

    st.divider()
    st.subheader("Speed")
    max_new_tokens = st.slider("Max answer length (tokens)", 60, 400, DEFAULT_MAX_NEW_TOKENS, step=20)
    use_cache_toggle = st.toggle("Cache repeated questions (demo speedup)", value=True)

    if st.session_state.vectorstore is not None:
        try:
            model_for_check, _ = load_model_and_tokenizer()
            p = next(model_for_check.parameters())
            st.caption(f"Model device: `{p.device}` · dtype: `{p.dtype}`")
        except Exception:
            pass

    if st.session_state.summary_text:
        with st.expander("🧠 Summary of older turns"):
            st.write(st.session_state.summary_text)

    if st.session_state.messages:
        with st.expander(f"💬 Full history ({len(st.session_state.messages)} messages)"):
            for m in st.session_state.messages:
                who = "**You:**" if m["role"] == "user" else "**Assistant:**"
                st.markdown(f"{who} {m['content'][:200]}")

    st.divider()
    if st.button("🗑️ Clear conversation", width="stretch"):
        st.session_state.messages = []
        st.session_state.summary_text = ""
        st.session_state.summarised_upto = 0
        st.session_state.qa_cache = {}
        st.rerun()

    if st.session_state.vectorstore is not None:
        st.success(f"✅ Knowledge base ready — {st.session_state.kb_summary}")
    else:
        st.warning("⚠️ No knowledge base loaded yet")

    if st.session_state.csv_preview is not None:
        with st.expander("Preview loaded CSV"):
            st.dataframe(st.session_state.csv_preview.head(10), width="stretch")

    if st.session_state.facts_text:
        with st.expander("📐 Computed dataset facts"):
            st.code(st.session_state.facts_text)


# ============================================================
# BUILD INDEX
# ============================================================
if build_clicked:
    with st.spinner("Loading documents..."):
        if pdf_file is not None:
            pdf_docs = load_pdf_docs_from_upload(pdf_file)
        elif use_defaults and os.path.exists(DEFAULT_PDF_PATH):
            pdf_docs = load_pdf_docs_from_path(DEFAULT_PDF_PATH)
        else:
            pdf_docs = []

        if csv_file is not None:
            df = pd.read_csv(csv_file)
        elif use_defaults and os.path.exists(DEFAULT_CSV_PATH):
            df = pd.read_csv(DEFAULT_CSV_PATH)
        else:
            df = None

        csv_docs, facts_docs, facts_text = [], [], None
        if df is not None:
            csv_docs = load_csv_docs_from_df(df)
            facts_text, facts_docs = build_data_facts(df)

    if not pdf_docs and not csv_docs:
        st.sidebar.error("No files found — upload files or ensure the default PDF/CSV exist in the app directory.")
    else:
        with st.spinner("Embedding documents and loading the model... (first run downloads the LLM)"):
            embedder = load_embedder()
            vectorstore = build_index(pdf_docs, csv_docs, facts_docs, embedder)
            load_model_and_tokenizer()

        st.session_state.vectorstore = vectorstore
        st.session_state.df = df
        st.session_state.facts_text = facts_text
        st.session_state.csv_preview = df
        st.session_state.kb_summary = (
            f"{len(pdf_docs)} PDF chunks + {len(csv_docs)} CSV records + {len(facts_docs)} fact blocks"
        )
        st.sidebar.success(f"Indexed {len(pdf_docs) + len(csv_docs) + len(facts_docs)} chunks.")
        st.rerun()


# ============================================================
# MAIN CHAT UI
# ============================================================
st.title("💬 Marketing Knowledge Assistant")
st.caption("Ask about customer segments, churn risk, or campaign strategy — with conversation memory.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources") and show_sources:
            with st.expander(f"📎 {len(msg['sources'])} sources used"):
                if msg.get("search_query") and msg["search_query"] != msg.get("original_question"):
                    st.caption(f"Retrieval query used: _{msg['search_query']}_")
                for i, s in enumerate(msg["sources"], 1):
                    st.markdown(f"**{i}. [{s['origin']}] distance {s['score']:.3f}** — {s['text'][:300]}")

user_input = st.chat_input("Ask a question about your customers or strategy...")

if user_input:
    if st.session_state.vectorstore is None:
        st.chat_message("assistant").warning("Please build the knowledge base first using the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("▌")

            model, tokenizer = load_model_and_tokenizer()
            prior = st.session_state.messages[:-1]

            if use_summary:
                window_start = max(0, len(prior) - max_turns * 2)
                if window_start - st.session_state.summarised_upto >= 4:
                    dropped = prior[st.session_state.summarised_upto:window_start]
                    st.session_state.summary_text = update_rolling_summary(
                        model, tokenizer, st.session_state.summary_text, dropped
                    )
                    st.session_state.summarised_upto = window_start

            history_text = format_chat_history(prior, max_turns)

            cache_key = (user_input, top_k, use_threshold, max_new_tokens)
            cached = st.session_state.qa_cache.get(cache_key) if use_cache_toggle and not prior else None

            if cached:
                answer, sources, search_query = cached
                placeholder.markdown(answer)
            else:
                answer, sources, search_query = answer_question(
                    question=user_input,
                    vectorstore=st.session_state.vectorstore,
                    df=st.session_state.df,
                    facts_text=st.session_state.facts_text,
                    model=model,
                    tokenizer=tokenizer,
                    history_text=history_text,
                    summary_text=st.session_state.summary_text if use_summary else "",
                    placeholder=placeholder,
                    k=top_k,
                    use_threshold=use_threshold,
                    condense=condense,
                    max_new_tokens=max_new_tokens,
                )
                if use_cache_toggle and not prior:
                    st.session_state.qa_cache[cache_key] = (answer, sources, search_query)

            if sources and show_sources:
                with st.expander(f"📎 {len(sources)} sources used"):
                    if search_query != user_input:
                        st.caption(f"Retrieval query used: _{search_query}_")
                    for i, s in enumerate(sources, 1):
                        st.markdown(f"**{i}. [{s['origin']}] distance {s['score']:.3f}** — {s['text'][:300]}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "search_query": search_query,
            "original_question": user_input,
        })

if not st.session_state.messages:
    st.info(
        "👋 **Try asking:**\n"
        "- How many customers are in the Champions segment?\n"
        "- Which segment has the highest average monetary value?\n"
        "- What's customer 12347's segment and churn risk?\n"
        "- What should we do for At Risk customers?\n"
        "- Then follow up with: *What segment are they in?* — to test memory."
    )
