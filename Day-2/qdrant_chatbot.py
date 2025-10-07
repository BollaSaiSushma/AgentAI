import os
import textwrap
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# ---- Config (can override via .env) ----
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_demo_pdf_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "5000"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ---- Helpers ----
def load_embedder(model_name: str) -> SentenceTransformer:
    print(f"Loading embedder: {model_name} ...")
    return SentenceTransformer(model_name)

def embed_query(encoder: SentenceTransformer, text: str) -> np.ndarray:
    vec = encoder.encode([text], normalize_embeddings=True, show_progress_bar=False)
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError("Embedding contains NaN/Inf")
    return arr

def qdrant_client(url: str = QDRANT_URL, api_key: Optional[str] = QDRANT_API_KEY) -> QdrantClient:
    return QdrantClient(url=url, api_key=api_key)

def ensure_collection_exists(client: QdrantClient, collection_name: str, expected_dim: Optional[int] = None):
    # If collection missing, raise (we assume vectors already upserted)
    cols = [c.name for c in client.get_collections().collections]
    if collection_name not in cols:
        raise RuntimeError(f"Collection '{collection_name}' not found in Qdrant at {QDRANT_URL}. Found: {cols}")
    if expected_dim is not None:
        info = client.get_collection(collection_name)
        dim = info.vectors.size if info.vectors else None
        if dim and expected_dim and dim != expected_dim:
            raise RuntimeError(f"Dimension mismatch: collection dim={dim} but embedder dim={expected_dim}")

def search_qdrant(client: QdrantClient, collection: str, vector: List[float], top_k: int = TOP_K):
    # returns list of (id, score, payload)
    result = client.search(
        collection_name=collection,
        query_vector=vector.tolist() if isinstance(vector, np.ndarray) else vector,
        limit=top_k,
        with_payload=True,
    )
    # result: List[ScoredPoint]
    matches = []
    for r in result:
        matches.append({
            "id": r.id,
            "score": float(r.score) if r.score is not None else None,
            "payload": r.payload or {}
        })
    return matches

def make_context(matches: List[Dict[str, Any]], char_limit: int = MAX_CONTEXT_CHARS) -> str:
    parts = []
    total = 0
    for m in matches:
        md = m["payload"]
        page = md.get("page", "?")
        chunk_index = md.get("chunk_index", "?")
        text = md.get("text", "")
        snippet = f"[p.{page}#{chunk_index}]\n{text}\n\n"
        if total + len(snippet) > char_limit:
            break
        parts.append(snippet)
        total += len(snippet)
    return "".join(parts)

def build_prompt(question: str, context: str) -> List[Dict[str, str]]:
    system = (
        "You are a concise, accurate assistant. Answer using ONLY the provided context. "
        "If the answer is not present, say you don't know."
    )
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "- Provide a short answer and cite context pieces using [p.<page>#<chunk_index>].\n"
        "- If the context doesn't contain the answer, respond: 'I don't see the answer in the provided documents.'"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def call_openai_chat(messages: List[Dict[str, str]], model: str = OPENAI_MODEL) -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package not installed. pip install openai") from e
    client = OpenAI()
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.0)
    return resp.choices[0].message.content.strip()

def extractive_fallback(question: str, matches: List[Dict[str, Any]], top_n: int = 3) -> str:
    # naive token overlap scoring, return top snippets
    qset = set(question.lower().split())
    scored = []
    for m in matches:
        text = (m["payload"].get("text", "")).lower()
        overlap = len(qset & set(text.split()))
        scored.append((overlap, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    best = [m for s, m in scored[:top_n]]
    if not best:
        return "I don't see the answer in the provided documents."
    out = ["Extractive fallback — top snippets:"]
    for m in best:
        md = m["payload"]
        page = md.get("page", "?")
        chunk = md.get("chunk_index", "?")
        snippet = textwrap.shorten(md.get("text", ""), width=300, placeholder="...")
        out.append(f"- [p.{page}#{chunk}] {snippet}")
    return "\n".join(out)

# ---- Chat loop ----
def start_chat():
    encoder = load_embedder(EMBED_MODEL)
    client = qdrant_client()
    # validate collection exists and dims match embedder (best-effort)
    try:
        expected_dim = encoder.encode(["hi"], normalize_embeddings=True).shape[1]
        ensure_collection_exists(client, COLLECTION_NAME, expected_dim=expected_dim)
    except Exception as e:
        print("Warning / validation error:", e)
        # continue anyway - user may know what they're doing

    have_openai = bool(os.getenv("OPENAI_API_KEY"))
    use_llm = have_openai

    print("Qdrant RAG chatbot ready. Type a question (or 'exit').")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
        if not q or q.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        # embed
        try:
            qvec = embed_query(encoder, q)
        except Exception as e:
            print("Embedding error:", e)
            continue

        # search
        try:
            matches = search_qdrant(client, COLLECTION_NAME, qvec, top_k=TOP_K)
        except Exception as e:
            print("Error searching Qdrant:", e)
            continue

        if not matches:
            print("Bot: no relevant documents found in Qdrant.")
            continue

        context = make_context(matches, char_limit=MAX_CONTEXT_CHARS)

        if use_llm:
            messages = build_prompt(q, context)
            try:
                answer = call_openai_chat(messages)
            except Exception as e:
                print(f"(LLM error: {e}) — falling back to extractive reply.")
                answer = extractive_fallback(q, matches)
        else:
            answer = extractive_fallback(q, matches)

        print("\nBot:")
        print(answer)
        print("\nSources:")
        for m in matches:
            md = m["payload"]
            page = md.get("page", "?")
            chunk = md.get("chunk_index", "?")
            score = m.get("score")
            print(f" - [p.{page}#{chunk}] id={m['id']} score={score}")

if __name__ == "__main__":
    start_chat()
