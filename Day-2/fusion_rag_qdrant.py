from __future__ import annotations
import os, re, json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# ============ Load env ============
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

# ============ Embeddings ============
class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> List[float]:
        v = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(v, dtype=float).ravel().tolist()

# ============ Optional LLM (OpenAI) ============
class LLM:
    def __init__(self, api_key: str | None, model_name: str):
        self.enabled = bool(api_key)
        if self.enabled:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = model_name

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        if not self.enabled:
            raise RuntimeError("LLM disabled (no OPENAI_API_KEY).")
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

# ============ Qdrant retriever ============
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
        if not must: return None
        return Filter(must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in must.items()])

    @staticmethod
    def _text_from_payload(p: Dict[str, Any] | None) -> str:
        if not p: return ""
        for k in ("text","chunk","content"):
            if p.get(k): return str(p[k])
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
        return [Hit(text=self._text_from_payload(h.payload), score=h.score, id=str(h.id), payload=h.payload or {}) for h in res]

# ============ Fusion building blocks ============
def simple_keywords(q: str, topn: int = 6) -> List[str]:
    # heuristic: lowercase tokens, drop very short/common words, uniq preserve order
    stop = set("""a an the of to in on for with about and or from as by is are be was were that this those these it its into over under at within across between into how what why when which""".split())
    toks = [t for t in re.findall(r"[A-Za-z0-9\-_/]+", q.lower()) if len(t) > 2 and t not in stop]
    seen, keep = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t); keep.append(t)
    # build short phrases
    phrases = keep[:topn]
    if len(keep) >= 2:
        phrases.append(" ".join(keep[:2]))
    if len(keep) >= 3:
        phrases.append(" ".join(keep[:3]))
    return phrases[:topn+2]

SPEC_SYS = "Generate 5 diverse, short search queries for vector retrieval. Return ONLY JSON list."
def llm_rewrites(llm: LLM, question: str) -> List[str]:
    if not llm or not llm.enabled:
        return []
    raw = llm.complete(SPEC_SYS, f"Question:\n{question}\nReturn JSON list only.", temperature=0.7)
    try:
        arr = json.loads(raw)
        return [str(x).strip() for x in arr if str(x).strip()]
    except Exception:
        return []

def reciprocal_rank_fusion(runs: List[List[Hit]], k: int = 50, c: float = 60.0) -> List[Hit]:
    scores: Dict[str, float] = {}
    best: Dict[str, Hit] = {}
    for run in runs:
        for rank, h in enumerate(run[:k], start=1):
            rid = h.id or f"{hash(h.text)}"
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (c + rank)
            if rid not in best or (h.score or 0) > (best[rid].score or 0):
                best[rid] = h
    return sorted(best.values(), key=lambda x: scores[x.id or f"{hash(x.text)}"], reverse=True)

def mmr(query_vec: List[float], cands: List[Hit], embedder: Embedder, lam: float = 0.7, top_k: int = 12) -> List[Hit]:
    if not cands: return []
    texts = [h.text for h in cands]
    M = SentenceTransformer(SET.embed_model).encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    M = np.asarray(M, dtype=float)
    q = np.asarray(query_vec, dtype=float)
    def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    selected, chosen_idx, pool = [], [], list(range(len(cands)))
    while pool and len(selected) < top_k:
        best_i, best_score = None, -1e9
        for i in pool:
            s_q = cos(M[i], q)
            s_rep = 0.0 if not chosen_idx else max(cos(M[i], M[j]) for j in chosen_idx)
            score = lam*s_q - (1-lam)*s_rep
            if score > best_score: best_i, best_score = i, score
        selected.append(cands[best_i]); chosen_idx.append(best_i); pool.remove(best_i)
    return selected

# ============ Answering (with citations) ============
ANSWER_SYS = """You are a careful, citation-first assistant.
Use ONLY the provided context to answer. If insufficient, say what is missing.
Cite with [S1], [S2], ... Keep answers concise and factual."""
VERIFY_SYS = """Verify if the draft is fully grounded in the context.
Return JSON only: {"support":"HIGH|MEDIUM|LOW","missing":["..."]}"""

class Answerer:
    def __init__(self, llm: LLM | None):
        self.llm = llm

    def build_context(self, hits: List[Hit], max_chars: int = 6000) -> Tuple[str, List[str]]:
        seen, used = set(), 0
        ctx_lines, sources = [], []
        for i, h in enumerate(hits, start=1):
            t = (h.text or "").strip()
            if not t or t.lower() in seen: continue
            entry = f"[S{i}] {t}"
            if used + len(entry) > max_chars: break
            ctx_lines.append(entry)
            src = (h.payload or {}).get("source_path") or (h.payload or {}).get("source") or h.id or f"chunk_{i}"
            sources.append(str(src))
            used += len(entry); seen.add(t.lower())
        return "\n\n".join(ctx_lines), sources

    def answer(self, question: str, context_text: str, sources: List[str]) -> str:
        if not self.llm or not self.llm.enabled:
            # fallback non-LLM answer (very basic)
            return f"(No LLM) Context suggests:\n{context_text[:900]}\n\nSources: {sources}"
        src_lines = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sources))
        user = f"Question:\n{question}\n\nContext:\n{context_text}\n\nSources:\n{src_lines}\n\nWrite a grounded answer with [S#] citations."
        return self.llm.complete(ANSWER_SYS, user, temperature=0.2)

    def verify(self, draft: str, context_text: str) -> Dict[str, Any]:
        if not self.llm or not self.llm.enabled:
            return {"support":"MEDIUM","missing":[]}
        raw = self.llm.complete(VERIFY_SYS, f"Draft:\n{draft}\n\nContext:\n{context_text}\n\nJSON only.", temperature=0.0)
        try: return json.loads(raw)
        except Exception: return {"support":"LOW","missing":["Unparseable verifier output"]}

# ============ Orchestrator ============
@dataclass
class FusionConfig:
    k_per_view: int = 6
    rrf_k: int = 30
    mmr_lambda: float = 0.7
    final_k: int = 12
    must_filter: Dict[str, Any] | None = None
    score_threshold: float | None = None
    use_llm_rewrites: bool = True

class FusionRAG:
    def __init__(self, retriever: QdrantRetriever, embedder: Embedder, llm: LLM | None, cfg: FusionConfig = FusionConfig()):
        self.retriever = retriever
        self.embedder = embedder
        self.llm = llm
        self.cfg = cfg
        self.answerer = Answerer(llm)

    def _build_views(self, question: str) -> List[str]:
        views = [question]
        views += simple_keywords(question, topn=6)
        if self.cfg.use_llm_rewrites and self.llm and self.llm.enabled:
            views += llm_rewrites(self.llm, question)
        # de-dup, keep order
        seen, out = set(), []
        for v in views:
            v2 = re.sub(r"\s+"," ",v.strip())
            if v2 and v2.lower() not in seen:
                seen.add(v2.lower()); out.append(v2)
        return out[:10]  # cap to keep retrieval small

    def answer(self, question: str) -> Dict[str, Any]:
        views = self._build_views(question)

        # retrieve for each view
        runs: List[List[Hit]] = []
        for q in views:
            hits = self.retriever.search(
                q,
                k=self.cfg.k_per_view,
                score_threshold=self.cfg.score_threshold,
                must_filter=self.cfg.must_filter
            )
            runs.append(hits)

        # fuse & dedup
        fused = reciprocal_rank_fusion(runs, k=self.cfg.rrf_k)
        qvec = self.embedder.embed(question)
        reranked = mmr(qvec, fused, self.embedder, lam=self.cfg.mmr_lambda, top_k=self.cfg.final_k)

        # build context & answer
        ctx, sources = self.answerer.build_context(reranked)
        draft = self.answerer.answer(question, ctx, sources)
        verdict = self.answerer.verify(draft, ctx)

        return {
            "question": question,
            "views_used": views,
            "sources": sources,
            "verify": verdict,
            "answer": draft
        }

# ============ bootstrap ============
def build_fusion_rag() -> FusionRAG:
    client = QdrantClient(url=SET.qdrant_url, api_key=SET.qdrant_api_key)
    embedder = Embedder(SET.embed_model)
    retriever = QdrantRetriever(client, SET.collection, embedder)
    llm = LLM(SET.openai_api_key, SET.openai_model) if SET.openai_api_key else None
    cfg = FusionConfig(
        k_per_view=6,
        rrf_k=30,
        mmr_lambda=0.7,
        final_k=12,
        must_filter=None,        # e.g. {"doc_type":"kb"} or {"tenant":"prod"}
        score_threshold=None,
        use_llm_rewrites=True
    )
    return FusionRAG(retriever, embedder, llm, cfg)

if __name__ == "__main__":
    rag = build_fusion_rag()
    question = "Summarize the key points on renewable energy and data engineering."
    out = rag.answer(question)
    print("VIEWS:", out["views_used"])
    print("SOURCES:", out["sources"])
    print("VERIFY:", out["verify"])
    print("\n=== ANSWER ===\n", out["answer"])
