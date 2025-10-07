from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from openai import OpenAI

# ------------------ env & settings ------------------
load_dotenv()

@dataclass
class Settings:
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")
    collection: str = os.getenv("QDRANT_COLLECTION", "rag_demo_pdf_chunks")
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SET = Settings()

# ------------------ embedder ------------------
class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> List[float]:
        vec = self.model.encode(
            text,
            convert_to_tensor=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return np.asarray(vec, dtype=float).ravel().tolist()

# ------------------ llm ------------------
class LLM:
    def __init__(self, api_key: str | None, model_name: str):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = model_name

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

# ------------------ retriever ------------------
@dataclass
class RetrievedChunk:
    text: str
    score: float
    id: str | None
    payload: Dict[str, Any] | None

class QdrantRetriever:
    def __init__(self, client: QdrantClient, collection: str, embedder: Embedder):
        self.client = client
        self.collection = collection
        self.embedder = embedder

    @staticmethod
    def _text_from_payload(p: Dict[str, Any] | None) -> str:
        if not p: return ""
        for k in ("text", "chunk", "content"):
            if p.get(k):
                return str(p[k])
        return ""

    def _filter(self, must: Dict[str, Any] | None) -> Filter | None:
        if not must: return None
        return Filter(must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in must.items()])

    def search(self, query: str, k: int = 8, score_threshold: float | None = None,
               must_filter: Dict[str, Any] | None = None) -> List[RetrievedChunk]:
        qvec = self.embedder.embed(query)
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=qvec,
            limit=k,
            with_payload=True,
            with_vectors=False,
            query_filter=self._filter(must_filter),
            score_threshold=score_threshold
        )
        return [RetrievedChunk(
            text=self._text_from_payload(h.payload),
            score=h.score, id=str(h.id), payload=h.payload or {}
        ) for h in hits]

# ------------------ CRAG prompts ------------------
CRAG_SYSTEM = """You are a careful, citation-first assistant. Use ONLY the provided context to answer.
If the context is insufficient, say what is missing. Cite with [S1], [S2], ... Keep answers concise."""

VERIFY_SYSTEM = """You are a strict verifier. Given a draft and context, return JSON only:
{"support":"HIGH|MEDIUM|LOW","missing":["..."]}"""

REWRITE_SYSTEM = """You reformulate queries for vector search. Given the user's question and missing facts,
return 3 diverse rewrites as a JSON list of strings."""

# ------------------ helpers ------------------
def format_context(chunks: List[RetrievedChunk], max_chars: int = 6000) -> Tuple[str, List[str]]:
    seen = set()
    unique: List[RetrievedChunk] = []
    for c in sorted(chunks, key=lambda x: x.score, reverse=True):
        t = (c.text or "").strip()
        if t and t not in seen:
            seen.add(t)
            unique.append(c)

    ctx_lines, sources, used = [], [], 0
    for i, ch in enumerate(unique, 1):
        entry = f"[S{i}] {ch.text}"
        if used + len(entry) > max_chars:
            break
        ctx_lines.append(entry)
        src = (ch.payload or {}).get("source_path") or (ch.payload or {}).get("source") or ch.id or f"chunk_{i}"
        sources.append(str(src))
        used += len(entry)
    return "\n\n".join(ctx_lines), sources

def generate_answer(llm: LLM, question: str, context_text: str, sources: List[str]) -> str:
    src_lines = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sources))
    user = f"""User Question:
{question}

Context (numbered):
{context_text}

Sources:
{src_lines}

Write a grounded answer. Cite with [S#] where relevant."""
    return llm.complete(CRAG_SYSTEM, user)

def verify_answer(llm: LLM, draft: str, context_text: str) -> Dict[str, Any]:
    user = f"""Draft:
{draft}

Context:
{context_text}

JSON only."""
    raw = llm.complete(VERIFY_SYSTEM, user, temperature=0.0)
    try:
        return json.loads(raw)
    except Exception:
        return {"support":"LOW","missing":["Verifier returned unparseable output"]}

def rewrite_queries(llm: LLM, question: str, missing: List[str] | None = None) -> List[str]:
    miss = "\n".join(f"- {m}" for m in (missing or []))
    user = f"""Question:
{question}

Missing (optional):
{miss}

Return 3 rewrites as a JSON list."""
    raw = llm.complete(REWRITE_SYSTEM, user, temperature=0.7)
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            return [str(x) for x in arr if x]
    except Exception:
        pass
    return [question + " (more specific)", question + " (add key terms)", question + " (narrow scope)"]

# ------------------ CRAG orchestrator ------------------
@dataclass
class CRAGConfig:
    k_initial: int = 8
    k_correction: int = 12
    score_threshold: float | None = None
    must_filter: Dict[str, Any] | None = None
    accept_support_levels: tuple[str, ...] = ("HIGH", "MEDIUM")
    max_rounds: int = 2

class CorrectiveRAG:
    def __init__(self, retriever: QdrantRetriever, llm: LLM, embedder: Embedder, cfg: CRAGConfig = CRAGConfig()):
        self.retriever = retriever
        self.llm = llm
        self.embedder = embedder
        self.cfg = cfg

    def answer(self, question: str) -> Dict[str, Any]:
        # Round 0: retrieve -> answer -> verify
        chunks = self.retriever.search(
            question,
            k=self.cfg.k_initial,
            score_threshold=self.cfg.score_threshold,
            must_filter=self.cfg.must_filter
        )
        context_text, sources = format_context(chunks)
        draft = generate_answer(self.llm, question, context_text, sources)
        verdict = verify_answer(self.llm, draft, context_text)

        if verdict.get("support","LOW") in self.cfg.accept_support_levels:
            return {"answer": draft, "support": verdict["support"], "rounds": 0,
                    "used_rewrites": [], "missing": verdict.get("missing", []), "sources": sources}

        # Corrective rounds
        rewrites_used: List[str] = []
        missing = verdict.get("missing", [])
        all_chunks = list(chunks)

        for r in range(1, self.cfg.max_rounds + 1):
            rewrites = rewrite_queries(self.llm, question, missing)
            rewrites_used.extend(rewrites)

            for rw in rewrites:
                more = self.retriever.search(
                    rw,
                    k=self.cfg.k_correction,
                    score_threshold=self.cfg.score_threshold,
                    must_filter=self.cfg.must_filter
                )
                all_chunks.extend(more)

            context_text, sources = format_context(all_chunks)
            draft = generate_answer(self.llm, question, context_text, sources)
            verdict = verify_answer(self.llm, draft, context_text)

            if verdict.get("support","LOW") in self.cfg.accept_support_levels:
                return {"answer": draft, "support": verdict["support"], "rounds": r,
                        "used_rewrites": rewrites_used, "missing": verdict.get("missing", []), "sources": sources}
            missing = verdict.get("missing", [])

        # Fallback if still LOW
        return {
            "answer": f"{draft}\n\n—\nI couldn’t find enough grounded evidence in the knowledge base.",
            "support": verdict.get("support","LOW"),
            "rounds": self.cfg.max_rounds,
            "used_rewrites": rewrites_used,
            "missing": missing,
            "sources": sources
        }

# ------------------ bootstrap ------------------
def build_crag() -> CorrectiveRAG:
    client = QdrantClient(url=SET.qdrant_url, api_key=SET.qdrant_api_key)
    embedder = Embedder(SET.embed_model)
    retriever = QdrantRetriever(client, SET.collection, embedder)
    llm = LLM(SET.openai_api_key, SET.openai_model)
    cfg = CRAGConfig(
        k_initial=8,
        k_correction=12,
        score_threshold=None,   # e.g., 0.2 for normalized cosine scores
        must_filter=None,       # e.g., {"doc_type":"kb"} or {"tenant":"prod"}
        accept_support_levels=("HIGH","MEDIUM"),
        max_rounds=2
    )
    return CorrectiveRAG(retriever, llm, embedder, cfg)

if __name__ == "__main__":
    crag = build_crag()
    # Try something from the sample PDF themes (renewables / data eng / call-center ops)
    question = "Give the key themes in the document for renewable energy and data engineering."
    result = crag.answer(question)
    print("Support:", result["support"])
    print("Rounds:", result["rounds"])
    print("Sources:", result["sources"])
    print("Missing:", result.get("missing"))
    print("\n=== ANSWER ===\n")
    print(result["answer"])
