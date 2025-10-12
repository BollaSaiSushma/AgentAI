from __future__ import annotations
import os, re, json, time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import urlparse

import numpy as np
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# ============ Load env ============
load_dotenv()

@dataclass
class Settings:
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    collection: str = os.getenv("QDRANT_COLLECTION", "rag_demo_pdf_chunks")
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    google_doc_url: Optional[str] = os.getenv("GOOGLE_DOC_URL")

SET = Settings()

# ============ Embeddings ============
class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> List[float]:
        v = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(v, dtype=float).ravel().tolist()

    def embed_many(self, texts: List[str]) -> np.ndarray:
        v = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(v, dtype=float)

# ============ Optional LLM (OpenAI) ============
class LLM:
    def __init__(self, api_key: Optional[str], model_name: str):
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

# ============ Common Hit type ============
@dataclass
class Hit:
    text: str
    score: float
    id: Optional[str]
    payload: Dict[str, Any] | None

# ============ Qdrant retriever ============
class QdrantRetriever:
    def __init__(self, client: QdrantClient, collection: str, embedder: Embedder):
        self.client = client
        self.collection = collection
        self.embedder = embedder

    def _filter(self, must: Optional[Dict[str, Any]]) -> Optional[Filter]:
        if not must: return None
        return Filter(must=[FieldCondition(key=k, match=MatchValue(value=v)) for k,v in must.items()])

    @staticmethod
    def _text_from_payload(p: Optional[Dict[str, Any]]) -> str:
        if not p: return ""
        for k in ("text","chunk","content"):
            if p.get(k): return str(p[k])
        return ""

    def search(self, query: str, k: int = 8, score_threshold: Optional[float] = None,
               must_filter: Optional[Dict[str, Any]] = None) -> List[Hit]:
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

# ============ Google Doc retriever (in-memory) ============
def _doc_id_from_url(url: str) -> Optional[str]:
    try:
        # works for links like /d/<id>/edit or /d/<id>/
        parts = urlparse(url).path.split("/")
        if "d" in parts:
            i = parts.index("d")
            return parts[i+1] if i+1 < len(parts) else None
    except Exception:
        pass
    return None

def _fetch_google_doc_text(doc_url: str) -> str:
    """
    Tries to download as plain text using the public export endpoint.
    Requires the doc to be shared with 'Anyone with the link' (no auth flow here).
    """
    doc_id = _doc_id_from_url(doc_url)
    if not doc_id:
        raise RuntimeError("Could not parse Google Doc ID from URL.")
    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    r = requests.get(export_url, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to fetch Google Doc (HTTP {r.status_code}). Is it publicly accessible?")
    return r.text

def _chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 200) -> List[str]:
    text = " ".join(text.split())
    out, n, start, step = [], len(text), 0, max(1, chunk_chars - overlap)
    while start < n:
        end = min(n, start + chunk_chars)
        out.append(text[start:end])
        start += step
    return out

class GoogleDocRetriever:
    """
    Downloads/embeds the Google Doc once, then answers retrieval queries in-memory.
    """
    def __init__(self, embedder: Embedder, doc_url: Optional[str], source_name: str = "google_doc"):
        self.embedder = embedder
        self.doc_url = doc_url
        self.source_name = source_name
        self.ready = False
        self.chunks: List[str] = []
        self.embs: Optional[np.ndarray] = None
        self.source_id = None
        if doc_url:
            try:
                t0 = time.time()
                txt = _fetch_google_doc_text(doc_url)
                self.chunks = _chunk_text(txt, chunk_chars=1200, overlap=200)
                if self.chunks:
                    self.embs = self.embedder.embed_many(self.chunks)
                    self.source_id = _doc_id_from_url(doc_url) or "google_doc"
                    self.ready = True
                print(f"[GoogleDocRetriever] Loaded {len(self.chunks)} chunks in {time.time()-t0:.2f}s")
            except Exception as e:
                print(f"[GoogleDocRetriever] WARN: {e}. Google Doc will be skipped.")

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12)
        return float(np.dot(a, b) / denom)

    def search(self, query: str, k: int = 8) -> List[Hit]:
        if not self.ready or self.embs is None or not self.chunks:
            return []
        q = np.asarray(self.embedder.embed(query), dtype=float)
        sims = (self.embs @ q) / (np.linalg.norm(self.embs, axis=1) * np.linalg.norm(q) + 1e-12)
        idx = np.argsort(-sims)[:k]
        out: List[Hit] = []
        for i in idx:
            out.append(Hit(
                text=self.chunks[int(i)],
                score=float(sims[int(i)]),
                id=f"{self.source_name}_{self.source_id}_{int(i)}",
                payload={"source": f"{self.source_name}:{self.source_id}"}
            ))
        return out

# ============ Fusion building blocks ============
def simple_keywords(q: str, topn: int = 6) -> List[str]:
    stop = set("""a an the of to in on for with about and or from as by is are be was were that this those these it its into over under at within across between into how what why when which""".split())
    toks = [t for t in re.findall(r"[A-Za-z0-9\-_/]+", q.lower()) if len(t) > 2 and t not in stop]
    seen, keep = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t); keep.append(t)
    phrases = keep[:topn]
    if len(keep) >= 2: phrases.append(" ".join(keep[:2]))
    if len(keep) >= 3: phrases.append(" ".join(keep[:3]))
    return phrases[:topn+2]

SPEC_SYS = "Generate 5 diverse, short search queries for vector retrieval. Return ONLY JSON list."
def llm_rewrites(llm: LLM, question: str) -> List[str]:
    if not llm or not llm.enabled: return []
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
    M = embedder.embed_many(texts)
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
    def __init__(self, llm: Optional[LLM]):
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
    must_filter: Optional[Dict[str, Any]] = None
    score_threshold: Optional[float] = None
    use_llm_rewrites: bool = True

class FusionRAG:
    def __init__(self, qdrant: QdrantRetriever, gdoc: GoogleDocRetriever, embedder: Embedder, llm: Optional[LLM], cfg: FusionConfig = FusionConfig()):
        self.qdrant = qdrant
        self.gdoc = gdoc
        self.embedder = embedder
        self.llm = llm
        self.cfg = cfg
        self.answerer = Answerer(llm)

    def _build_views(self, question: str) -> List[str]:
        views = [question] + simple_keywords(question, topn=6)
        if self.cfg.use_llm_rewrites and self.llm and self.llm.enabled:
            views += llm_rewrites(self.llm, question)
        # de-dup, preserve order
        seen, out = set(), []
        for v in views:
            v2 = re.sub(r"\s+"," ",v.strip())
            if v2 and v2.lower() not in seen:
                seen.add(v2.lower()); out.append(v2)
        return out[:12]

    def answer(self, question: str) -> Dict[str, Any]:
        views = self._build_views(question)

        # Retrieve per view from BOTH sources
        runs: List[List[Hit]] = []
        for q in views:
            # Qdrant
            runs.append(self.qdrant.search(
                q, k=self.cfg.k_per_view,
                score_threshold=self.cfg.score_threshold,
                must_filter=self.cfg.must_filter
            ))
            # Google Doc (in-memory)
            runs.append(self.gdoc.search(q, k=self.cfg.k_per_view))

        # Fuse, then MMR de-dup
        fused = reciprocal_rank_fusion(runs, k=self.cfg.rrf_k)
        qvec = self.embedder.embed(question)
        reranked = mmr(qvec, fused, self.embedder, lam=self.cfg.mmr_lambda, top_k=self.cfg.final_k)

        # Build context & answer
        ctx, sources = self.answerer.build_context(reranked)
        draft = self.answerer.answer(question, ctx, sources)
        verdict = self.answerer.verify(draft, ctx)

        return {"question": question, "views_used": views, "sources": sources, "verify": verdict, "answer": draft}

# ============ bootstrap ============
def build_fusion_rag() -> FusionRAG:
    embedder = Embedder(SET.embed_model)

    # Qdrant retriever
    client = QdrantClient(url=SET.qdrant_url, api_key=SET.qdrant_api_key)
    qdrant = QdrantRetriever(client, SET.collection, embedder)

    # Google Doc retriever (in-memory)
    gdoc = GoogleDocRetriever(embedder, SET.google_doc_url, source_name="google_doc")

    # LLM for rewrites/answer/verify (optional)
    llm = LLM(SET.openai_api_key, SET.openai_model) if SET.openai_api_key else None

    cfg = FusionConfig(
        k_per_view=6,
        rrf_k=30,
        mmr_lambda=0.7,
        final_k=12,
        must_filter=None,
        score_threshold=None,
        use_llm_rewrites=True
    )
    return FusionRAG(qdrant, gdoc, embedder, llm, cfg)

if __name__ == "__main__":
    rag = build_fusion_rag()
    question = "Summarize the key points across the PDF and the Google Doc."
    out = rag.answer(question)
    print("VIEWS:", out["views_used"])
    print("SOURCES:", out["sources"])
    print("VERIFY:", out["verify"])
    print("\n=== ANSWER ===\n", out["answer"])
