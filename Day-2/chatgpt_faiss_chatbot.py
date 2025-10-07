import os
import json
import textwrap
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ------------- Paths / Config (match your index build) -------------
INDEX_PATH = "faiss_index.bin"
META_PATH = "faiss_meta.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # same as build step
TOP_K = 5                                               # retrieval depth
MAX_CONTEXT_CHARS = 6000                                # guardrail for prompt size

# ------------- Optional LLM (OpenAI) -------------
USE_LLM = True  # set False to force extractive mode
OPENAI_MODEL = "gpt-4o-mini"  # light & cheap; change to your preferred model

# ------------- Loaders -------------
def load_index(index_path: str) -> faiss.IndexIDMap2:
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {os.path.abspath(index_path)}")
    return faiss.read_index(index_path)

def load_meta(meta_path: str) -> Dict[str, Any]:
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {os.path.abspath(meta_path)}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ------------- Embedding -------------
def load_embedder(model_name: str) -> SentenceTransformer:
    # normalize_embeddings=True gives unit vectors → cosine via inner product
    return SentenceTransformer(model_name)

def embed_queries(encoder: SentenceTransformer, texts: List[str]) -> np.ndarray:
    vecs = encoder.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    return vecs.astype("float32")

# ------------- Retrieval -------------
def search(index: faiss.Index, qvec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    # qvec shape (1, dim)
    scores, ids = index.search(qvec, top_k)
    return scores[0], ids[0]

def gather_hits(meta: Dict[str, Any], hit_ids: List[int]) -> List[Dict[str, Any]]:
    # meta["items"] is aligned with ids 0..N-1 created at build time
    items = meta.get("items", [])
    out = []
    for hid in hit_ids:
        if hid == -1:
            continue
        it = items[hid]
        out.append({
            "id": it["id"],
            "page": it["page"],
            "chunk_index": it["chunk_index"],
            "text": it["text"]
        })
    return out

# ------------- Prompt assembly -------------
def make_context(docs: List[Dict[str, Any]], limit_chars: int = MAX_CONTEXT_CHARS) -> str:
    ctx_parts = []
    total = 0
    for d in docs:
        piece = f"[p.{d['page']}#{d['chunk_index']}]\n{d['text']}\n"
        if total + len(piece) > limit_chars:
            break
        ctx_parts.append(piece)
        total += len(piece)
    return "\n".join(ctx_parts)

def build_prompt(question: str, context: str) -> List[Dict[str, str]]:
    system = (
        "You are a precise assistant. Answer using ONLY the provided context. "
        "If the answer isn't in the context, say you don't see it."
    )
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "- Cite sources in the form [p.<page>#<chunk>].\n"
        "- Be concise and accurate.\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

# ------------- Optional: LLM call (OpenAI) -------------
def call_llm(messages: List[Dict[str, str]], model: str) -> str:
    # Lazy import to avoid hard dependency if user runs in extractive mode
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
    return resp.choices[0].message.content.strip()

# ------------- Extractive fallback (no LLM) -------------
def extractive_answer(question: str, docs: List[Dict[str, Any]]) -> str:
    """
    Very simple fallback: return the most relevant snippets (no generation).
    """
    # naive score = token overlap
    q_tokens = set(question.lower().split())
    scored = []
    for d in docs:
        toks = set(d["text"].lower().split())
        score = len(q_tokens & toks)
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)

    best = [d for _, d in scored[:3]]
    if not best:
        return "I don't see the answer in the indexed content."

    out = ["Top relevant snippets (no-LLM mode):"]
    for d in best:
        out.append(f"- [p.{d['page']}#{d['chunk_index']}] " + textwrap.shorten(d["text"], width=300))
    return "\n".join(out)

# ------------- Chat Loop -------------
def start_chat():
    load_dotenv()
    have_key = bool(os.getenv("OPENAI_API_KEY"))
    use_llm = USE_LLM and have_key

    # Load FAISS + meta + encoder
    print("Loading FAISS index and metadata...")
    index = load_index(INDEX_PATH)
    meta = load_meta(META_PATH)
    encoder = load_embedder(EMBED_MODEL)

    print("Ready! Type your question (or 'exit').")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not q or q.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break

        # 1) Embed & retrieve
        qvec = embed_queries(encoder, [q])  # shape (1, dim)
        scores, ids = search(index, qvec, TOP_K)
        docs = gather_hits(meta, ids.tolist())

        if not docs:
            print("Bot: I couldn't find anything relevant in the index.")
            continue

        # 2) Build context
        context = make_context(docs, MAX_CONTEXT_CHARS)

        # 3) Answer
        if use_llm:
            messages = build_prompt(q, context)
            try:
                answer = call_llm(messages, OPENAI_MODEL)
            except Exception as e:
                print(f"(LLM error: {e}) Falling back to extractive mode.")
                answer = extractive_answer(q, docs)
        else:
            answer = extractive_answer(q, docs)

        # 4) Show answer + sources
        print("\nBot:")
        print(answer)
        print("\nSources:")
        for d in docs:
            print(f"  - [p.{d['page']}#{d['chunk_index']}]")

if __name__ == "__main__":
    start_chat()
