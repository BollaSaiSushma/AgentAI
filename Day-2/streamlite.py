# streamlit_selfrag.py
import os
import json
import time
from typing import List, Dict, Any, Tuple

import streamlit as st

# IMPORTANT: this import assumes you saved your code as selfrag_core.py in the same folder.
# If you used a different filename, change the import below accordingly.
import self_rag as core
from qdrant_client.http import models as qm


st.set_page_config(page_title="Self-RAG Console & Chat", page_icon="🧠", layout="wide")
st.title("🧠 Self-RAG (Qdrant) — Console & Chat")

# -----------------------------
# Sidebar: Settings
# -----------------------------
with st.sidebar:
    st.header("Settings")

    # Secrets / env defaults
    default_qdrant = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL", "http://localhost:6333"))
    default_key = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY", ""))
    default_coll = st.secrets.get("QDRANT_COLLECTION", os.getenv("QDRANT_COLLECTION", "rag_demo_pdf_chunks"))
    default_model = st.secrets.get("EMBED_MODEL", os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    default_openai = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

    qdrant_url = st.text_input("Qdrant URL", value=default_qdrant)
    qdrant_key = st.text_input("Qdrant API Key (if cloud)", value=default_key, type="password")
    collection = st.text_input("Collection", value=default_coll)
    embed_model = st.text_input("Embedding model", value=default_model)

    st.markdown("---")
    st.subheader("Self-RAG Knobs")
    k_initial = st.slider("Top-k (initial)", 4, 24, int(os.getenv("SELF_RAG_K", "8")), 1)
    k_expand = st.slider("Top-k (expand)", 4, 24, int(os.getenv("SELF_RAG_K_EXPAND", "8")), 1)
    max_rounds = st.slider("Max refine rounds", 1, 5, int(os.getenv("SELF_RAG_MAX_ROUNDS", "3")), 1)
    ctx_chars = st.slider("Context size (chars)", 800, 8000, int(os.getenv("SELF_RAG_CONTEXT_CHARS", "2600")), 100)
    min_signal = st.slider("Min top-signal (gate)", 0.0, 1.0, float(os.getenv("SELF_RAG_MIN_SIGNAL", "0.32")), 0.01)

    st.markdown("---")
    if default_openai:
        st.success("OPENAI_API_KEY found ✅")
    else:
        st.info("Tip: put OPENAI_API_KEY in .streamlit/secrets.toml or your env")

# -----------------------------
# Build / cache Self-RAG core
# -----------------------------
@st.cache_resource(show_spinner=True)
def build_core(
    qdrant_url: str,
    qdrant_key: str,
    collection: str,
    embed_model: str,
    k_initial: int,
    k_expand: int,
    max_rounds: int,
    ctx_chars: int,
    min_signal: float,
):
    # Set module-level knobs so core classes use them
    core.QDRANT_URL = qdrant_url
    core.QDRANT_API_KEY = qdrant_key or None
    core.COLLECTION_NAME = collection
    core.EMBED_MODEL = embed_model
    core.K_INITIAL = k_initial
    core.K_EXPAND = k_expand
    core.MAX_ROUNDS = max_rounds
    core.MAX_CONTEXT_CHARS = ctx_chars
    core.TOP_SIGNAL_MIN = min_signal

    retriever = core.QdrantRetriever(collection, embed_model)
    llm = core.get_llm()
    selfrag = core.SelfRAG(retriever, llm)
    return selfrag, retriever

try:
    SELF_RAG, RETRIEVER = build_core(
        qdrant_url, qdrant_key, collection, embed_model, k_initial, k_expand, max_rounds, ctx_chars, min_signal
    )
except Exception as e:
    st.error(f"Failed to initialize Self-RAG: {e}")
    st.stop()

# -----------------------------
# Helpers
# -----------------------------
def fetch_citation_snippets(citations: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Return list of (header, text) for each citation by querying Qdrant via page/chunk.
    """
    out = []
    for c in citations:
        page, chunk = c.get("page"), c.get("chunk")
        if page is None or chunk is None:
            continue
        flt = qm.Filter(
            must=[
                qm.FieldCondition(key="page", match=qm.MatchValue(value=page)),
                qm.FieldCondition(key="chunk_index", match=qm.MatchValue(value=chunk)),
            ]
        )
        pts, _ = RETRIEVER.client.scroll(
            collection_name=RETRIEVER.collection,
            limit=1,
            with_payload=True,
            with_vectors=False,
            scroll_filter=flt,
        )
        if pts:
            pl = pts[0].payload or {}
            header = f"{pl.get('source') or 'doc'} | p:{pl.get('page')} | c:{pl.get('chunk_index')}"
            text = (pl.get("text") or "").strip()
            out.append((header, text))
    return out

# -----------------------------
# Tabs
# -----------------------------
tab_console, tab_chat = st.tabs(["🖥 Console", "💬 Chat"])

# ===== Console tab =====
with tab_console:
    st.subheader("Console Runner")
    q = st.text_input("Ask a question", placeholder="e.g., What does the policy say about prior authorization?")
    go = st.button("Run Self-RAG", type="primary", use_container_width=True)

    if go and q.strip():
        t0 = time.time()
        try:
            res = SELF_RAG.ask(q.strip())
        except Exception as e:
            st.error(f"Self-RAG error: {e}")
            st.stop()
        dt = (time.time() - t0) * 1000

        st.markdown(f"**Latency:** `{dt:.0f} ms`  **Rounds:** `{res.rounds}`")

        with st.container(border=True):
            st.markdown("#### ✅ Answer")
            st.write(res.answer)

        if res.citations:
            with st.container(border=True):
                st.markdown("#### 📎 Citations")
                st.write(", ".join([f"[{c['idx']}] p:{c.get('page')} c:{c.get('chunk')}" for c in res.citations]))

            # Show the actual cited text blocks
            with st.expander("Show cited snippets"):
                for head, txt in fetch_citation_snippets(res.citations):
                    st.markdown(f"**{head}**")
                    st.code(txt)

        with st.expander("Debug (JSON)"):
            st.code(json.dumps(res.debug, indent=2))

# ===== Chat tab =====
with tab_chat:
    st.subheader("Chat with Self-RAG")

    if "chat_msgs" not in st.session_state:
        st.session_state.chat_msgs: List[Dict[str, str]] = []

    # render history
    for msg in st.session_state.chat_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Type your question…")
    if user_q:
        st.session_state.chat_msgs.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving & grounding…"):
                t0 = time.time()
                try:
                    res = SELF_RAG.ask(user_q)
                except Exception as e:
                    st.error(f"Self-RAG error: {e}")
                    st.stop()
                dt = (time.time() - t0) * 1000
                st.caption(f"{dt:.0f} ms")
                st.markdown(res.answer)
                if res.citations:
                    st.caption("Citations: " + ", ".join([f"[{c['idx']}] p:{c.get('page')} c:{c.get('chunk')}" for c in res.citations]))

        st.session_state.chat_msgs.append({"role": "assistant", "content": res.answer})
