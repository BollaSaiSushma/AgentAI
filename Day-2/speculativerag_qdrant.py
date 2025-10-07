from __future__ import annotations
import os, json, re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# ---------------------------
# Settings
# ---------------------------
@dataclass
class Settings:
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "rag_demo_pdf_chunks")
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SET = Settings()

# ---------------------------
# Embeddings
# ---------------------------
class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> List[float]:
        v = self.model.encode(
            text,
            convert_to_tensor=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return np.asarray(v, dtype=float).ravel().tolist()

# ---------------------------
# LLM Wrapper (OpenAI)
# ---------------------------
from openai import OpenAI

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

# ---------------------------
# Qdrant Retriever
# ---------------------------
@dataclass
class Hit:
    text: str
    score: float
    id: str | None
    payload: Dict[str, Any] | None

class QdrantRetriever:
    def __init__(self, client: QdrantClient, collection: str, embedder: Embedder):
        self.client = client
        self.collection = collection
        self.embedder = embedder

    def _filter(self, must: Dict[str, Any] | None) -> Filter | None:
        if not must:
            return None
        return Filter(must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in must.items()])

    @staticmethod
    def _text_from_payload(p: Dict[str, Any] | None) -> str:
        if not p:
            return ""
        for k in ("text", "chunk", "content"):
            if p.get(k):
                return str(p[k])
        return ""

    def search(self, query: str, k: int = 8, score_threshold: float | None = None,
               must_filter: Dict[str, Any] | None = None) -> List[Hit]:
        vec = self.embedder.embed(query)
        res = self.client.search(
            collection_name=self.collection,
            query_vector=vec,
            limit=k,
            with_payload=True,
            with_vectors=False,
            query_filter=self._filter(must_filter),
            score_threshold=score_threshold
        )
        return [Hit(text=self._text_from_payload(h.payload),
                    score=h.score, id=str(h.id), payload=h.payload or {}) for h in res]

# ---------------------------
# Speculative Query Generation
# ---------------------------
SPEC_SYSTEM = """Generate multiple short and diverse search queries to retrieve context from a vector DB.
Return ONLY a JSON list of strings. No markdown or commentary."""

def generate_speculative_queries(llm: LLM, question: str, n: int = 6) -> List[str]:
    user = f"Question:\n{question}\n\nGenerate {n} diverse retrieval queries (synonyms, rephrasings, alternate phrasings). Return JSON list only."
    raw = llm.complete(SPEC_SYSTEM, user, temperature=0.7)
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            seen, out = set(), []
            for q in arr:
                q2 = re.sub(r"\s+", " ", str(q).strip())
                if q2 and q2.lower() not in seen:
                    seen.add(q2.lower())
                    out.append(q2)
            return out
    except Exception:
        pass
    return [question, f"{question} definition", f"{question} examples"]

# ---------------------------
# Fusion & Re-ranking
# ---------------------------
def reciprocal_rank_fusion(runs: List[List[Hit]], k: int = 50, c: float = 60.0) -> List[Hit]:
    scores, best = {}, {}
    for run in runs:
        for rank, h in enumerate(run[:k], start=1):
            rid = h.id or f"{hash(h.text)}"
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (c + rank)
            if rid not in best or h.score > best[rid].score:
                best[rid] = h
    return sorted(best.values(), key=lambda x: scores[x.id or f"{hash(x.text)}"], reverse=True)

def mmr(query_vec: List[float], cands: List[Hit], embedder: Embedder, lam: float = 0.7, top_k: int = 12) -> List[Hit]:
    texts = [h.text for h in cands]
    if not texts:
        return []
    M = SentenceTransformer(SET.embed_model).encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    M = np.asarray(M, dtype=float)
    q = np.asarray(query_vec, dtype=float)
    def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
    selected, chosen_idx, pool = [], [], list(range(len(cands)))
    while pool and len(selected) < top_k:
        best_i, best_score = None, -1e9
        for i in pool:
            s_q = cos(M[i], q)
            s_rep = 0.0 if not chosen_idx else max(cos(M[i], M[j]) for j in chosen_idx)
            score = lam * s_q - (1 - lam) * s_rep
            if score > best_score:
                best_score, best_i = score, i
        selected.append(cands[best_i])
        chosen_idx.append(best_i)
        pool.remove(best_i)
    return selected

# ---------------------------
# Context & Answering
# ---------------------------
ANSWER_SYSTEM = """You are a careful, citation-first assistant.
Use ONLY the provided context to answer. If insufficient, say what is missing.
Cite with [S1], [S2], ... and keep answers concise and factual."""

def build_context(hits: List[Hit], max_chars: int = 6000) -> Tuple[str, List[str]]:
    seen, used = set(), 0
    ctx_lines, sources = [], []
    for i, h in enumerate(hits, start=1):
        t = (h.text or "").strip()
        if not t or t.lower() in seen:
            continue
        entry = f"[S{i}] {t}"
        if used + len(entry) > max_chars:
            break
        ctx_lines.append(entry)
        src = (h.payload or {}).get("source") or h.id or f"chunk_{i}"
        sources.append(str(src))
        used += len(entry)
        seen.add(t.lower())
    return "\n\n".join(ctx_lines), sources

def answer_with_citations(llm: LLM, question: str, context_text: str, sources: List[str]) -> str:
    src_lines = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sources))
    user = f"Question:\n{question}\n\nContext:\n{context_text}\n\nSources:\n{src_lines}\n\nWrite a grounded answer with [S#] citations."
    return llm.complete(ANSWER_SYSTEM, user, temperature=0.2)

# ---------------------------
# Verifier
# ---------------------------
VERIFY_SYSTEM = """Check if the draft answer is supported by the context.
Return JSON only: {"support":"HIGH|MEDIUM|LOW", "missing":["..."]}"""

def verify_answer(llm: LLM, draft: str, context_text: str) -> Dict[str, Any]:
    raw = llm.complete(VERIFY_SYSTEM, f"Draft:\n{draft}\n\nContext:\n{context_text}\n\nJSON only.", temperature=0.0)
    try:
        return json.loads(raw)
    except Exception:
        return {"support":"LOW","missing":["Could not parse verifier output"]}

# ---------------------------
# Speculative RAG Orchestrator
# ---------------------------
@dataclass
class SpecRAGConfig:
    n_queries: int = 6
    k_per_query: int = 6
    rrf_k: int = 30
    mmr_lambda: float = 0.7
    final_k: int = 12
    must_filter: Dict[str, Any] | None = None
    score_threshold: float | None = None
    verify: bool = True

class SpeculativeRAG:
    def __init__(self, retriever: QdrantRetriever, llm: LLM, embedder: Embedder, cfg: SpecRAGConfig = SpecRAGConfig()):
        self.retriever = retriever
        self.llm = llm
        self.embedder = embedder
        self.cfg = cfg

    def answer(self, question: str) -> Dict[str, Any]:
        hypos = generate_speculative_queries(self.llm, question, self.cfg.n_queries)
        runs = [self.retriever.search(q, k=self.cfg.k_per_query) for q in hypos]
        fused = reciprocal_rank_fusion(runs, k=self.cfg.rrf_k)
        qvec = self.embedder.embed(question)
        reranked = mmr(qvec, fused, self.embedder, lam=self.cfg.mmr_lambda, top_k=self.cfg.final_k)
        context_text, sources = build_context(reranked)
        draft = answer_with_citations(self.llm, question, context_text, sources)
        result = {"answer": draft, "sources": sources, "queries": hypos, "stage": "draft"}
        if self.cfg.verify:
            verdict = verify_answer(self.llm, draft, context_text)
            result["verify"] = verdict
        return result

# ---------------------------
# Bootstrap
# ---------------------------
def build_spec_rag() -> SpeculativeRAG:
    client = QdrantClient(url=SET.qdrant_url, api_key=SET.qdrant_api_key)
    embedder = Embedder(SET.embed_model)
    retriever = QdrantRetriever(client, SET.qdrant_collection, embedder)
    llm = LLM(SET.openai_api_key, SET.openai_model)
    return SpeculativeRAG(retriever, llm, embedder, SpecRAGConfig())

if __name__ == "__main__":
    rag = build_spec_rag()
    question = "Summarize key themes from the uploaded PDF about renewable energy and AI."
    out = rag.answer(question)
    print("\n=== STAGE:", out.get("stage","draft"), "===\n")
    print("QUERIES:", out.get("queries"))
    print("SOURCES:", out.get("sources"))
    if "verify" in out:
        print("VERIFY:", out["verify"])
    print("\n=== ANSWER ===\n")
    print(out["answer"])
