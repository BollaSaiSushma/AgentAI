import os
import textwrap
import json
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer

# Pinecone & OpenAI imports (lazy-checks inside functions)
from pinecone import Pinecone

# Config (can override via env)
load_dotenv()
INDEX_NAME = os.getenv("INDEX_NAME", "rag-demo-index")
NAMESPACE = os.getenv("NAMESPACE", "pdf-chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "5000"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
USE_LLM = True  # set False to force extractive mode

# -----------------------------
# Utilities
# -----------------------------
def load_embedder(model_name: str) -> SentenceTransformer:
    print(f"Loading embedder: {model_name} ...")
    return SentenceTransformer(model_name)

def embed_text(encoder: SentenceTransformer, texts: List[str]) -> np.ndarray:
    # normalize_embeddings=True during encoding keeps vectors unit-length for cosine similarity
    vecs = encoder.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")

# -----------------------------
# Pinecone helpers
# -----------------------------
def get_pinecone_client() -> Pinecone:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY missing in environment (.env)")
    # Optionally pass environment if required by your Pinecone deployment
    p = Pinecone(api_key=api_key)
    return p

def query_pinecone(index, embedding: List[float], top_k: int = TOP_K, namespace: Optional[str] = NAMESPACE):
    """
    Returns list of matches: each match is dict with id, score, metadata (should contain 'text','page','chunk_index')
    """
    # API shape depends on Pinecone SDK; this uses index.query(queries=[embedding], top_k=..., include_metadata=True)
    resp = index.query(queries=[embedding.tolist()], top_k=top_k, include_metadata=True, namespace=namespace)
    # resp.results[0].matches -> list of matches
    results = []
    if not resp or not getattr(resp, "results", None):
        return results

    matches = resp.results[0].matches
    for m in matches:
        # m.id, m.score, m.metadata expected
        results.append({
            "id": m.id,
            "score": float(m.score),
            "metadata": dict(m.metadata or {})
        })
    return results

# -----------------------------
# Prompt assembly & LLM
# -----------------------------
def make_context_from_matches(matches: List[Dict[str, Any]], char_limit: int = MAX_CONTEXT_CHARS) -> str:
    parts = []
    total = 0
    for m in matches:
        md = m["metadata"]
        page = md.get("page", "?")
        chunk = md.get("chunk_index", "?")
        text = md.get("text", "")[:8000]  # cap per chunk
        snippet = f"[p.{page}#{chunk}]\n{text}\n"
        if total + len(snippet) > char_limit:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n".join(parts)

def build_messages(question: str, context: str) -> List[Dict[str, str]]:
    system = (
        "You are a helpful, precise assistant. Answer using ONLY the provided context. "
        "If the context doesn't contain the answer, say you don't know."
    )
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "- Provide a concise answer.\n"
        "- Quote or cite the context using tags like [p.<page>#<chunk_index>] where appropriate.\n"
        "- If the answer is not present in the context, respond: 'I don't see the answer in the provided documents.'"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def call_openai_chat(messages: List[Dict[str, str]], model: str = OPENAI_MODEL) -> str:
    # Lazy import so script still works without OpenAI present (fallback extractive)
    try:
        from openai import OpenAI
    except Exception as ex:
        raise RuntimeError("openai package not installed or not importable. pip install openai") from ex
    client = OpenAI()
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.0)
    return resp.choices[0].message.content.strip()

# -----------------------------
# Extractive fallback
# -----------------------------
def extractive_response(question: str, matches: List[Dict[str, Any]], top_n: int = 3) -> str:
    q_tokens = set(question.lower().split())
    scored = []
    for m in matches:
        text = m["metadata"].get("text", "")
        toks = set(text.lower().split())
        overlap = len(q_tokens & toks)
        scored.append((overlap, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    best = [m for _, m in scored[:top_n] if _ > 0]
    if not best:
        # If there is no token overlap, still show the top matches
        best = [m for _, m in scored[:top_n]]
    parts = ["Extractive (no LLM) answer — top snippets:"]
    for m in best:
        md = m["metadata"]
        page = md.get("page", "?")
        chunk = md.get("chunk_index", "?")
        text = textwrap_shorten(md.get("text", ""), 300)
        parts.append(f"- [p.{page}#{chunk}] {text}")
    return "\n".join(parts)

def textwrap_shorten(s: str, width: int = 200) -> str:
    return textwrap.shorten(" ".join(s.split()), width=width, placeholder="...")

# -----------------------------
# Chat loop
# -----------------------------
def start_chat():
    # Load resources
    encoder = load_embedder(EMBED_MODEL)

    pine = get_pinecone_client()
    index = pine.Index(INDEX_NAME)  # matches your upsert code style

    have_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    use_llm = USE_LLM and have_openai_key

    print("Pinecone RAG chatbot ready. Type a question (or 'exit').")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
        if not q or q.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        # 1) embed query
        q_emb = embed_text(encoder, [q])[0]  # single-vector

        # 2) query pinecone
        try:
            matches = query_pinecone(index, q_emb, top_k=TOP_K, namespace=NAMESPACE)
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            continue

        if not matches:
            print("Bot: No relevant documents found in Pinecone.")
            continue

        # 3) assemble context and answer
        context = make_context_from_matches(matches, char_limit=MAX_CONTEXT_CHARS)

        if use_llm:
            messages = build_messages(q, context)
            try:
                answer = call_openai_chat(messages)
            except Exception as e:
                print(f"(LLM error: {e}) — falling back to extractive response.")
                answer = extractive_response(q, matches)
        else:
            answer = extractive_response(q, matches)

        # 4) print answer + sources
        print("\nBot:")
        print(answer)
        print("\nSources:")
        for m in matches:
            md = m["metadata"]
            page = md.get("page", "?")
            chunk = md.get("chunk_index", "?")
            print(f"  - [p.{page}#{chunk}] id={m['id']} score={m['score']:.4f}")

if __name__ == "__main__":
    start_chat()
