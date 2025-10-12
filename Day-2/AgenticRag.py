from __future__ import annotations
import os, re, json, time, glob
os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import urlparse

import numpy as np
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# ---------------- Env ----------------
load_dotenv()

@dataclass
class Settings:
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "rag_demo_pdf_chunks")
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    google_doc_url: Optional[str] = os.getenv("GOOGLE_DOC_URL")
    local_docs_dir: str = os.getenv("LOCAL_DOCS_DIR", "./kb")

SET = Settings()

# -------------- Embeddings --------------
class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    def embed(self, text: str) -> List[float]:
        v = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(v, dtype=float).ravel().tolist()
    def embed_many(self, texts: List[str]) -> np.ndarray:
        v = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(v, dtype=float)

# -------------- LLM --------------
class LLM:
    def __init__(self, api_key: Optional[str], model_name: str):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model_name
    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=temperature,
        )
        return r.choices[0].message.content.strip()

# -------------- Common Hit --------------
@dataclass
class Hit:
    text: str
    score: float
    id: Optional[str]
    payload: Dict[str, Any] | None

# -------------- Qdrant Retriever --------------
class QdrantRetriever:
    def __init__(self, client: QdrantClient, collection: str, embedder: Embedder):
        self.client = client; self.collection = collection; self.embedder = embedder
    def _filter(self, must: Optional[Dict[str, Any]]): 
        if not must: return None
        return Filter(must=[FieldCondition(key=k, match=MatchValue(value=v)) for k,v in must.items()])
    @staticmethod
    def _txt(p: Optional[Dict[str,Any]]): 
        if not p: return ""
        for k in ("text","chunk","content"):
            if p.get(k): return str(p[k])
        return ""
    def search(self, query: str, k=6, must_filter=None, score_threshold=None) -> List[Hit]:
        vec = self.embedder.embed(query)
        res = self.client.search(
            collection_name=self.collection, query_vector=vec, limit=k,
            with_payload=True, with_vectors=False, query_filter=self._filter(must_filter),
            score_threshold=score_threshold
        )
        return [Hit(text=self._txt(h.payload), score=h.score, id=str(h.id), payload=h.payload or {}) for h in res]

# -------------- Google Doc Retriever (in-memory) --------------
def _doc_id_from_url(url: str) -> Optional[str]:
    try:
        parts = urlparse(url).path.split("/")
        if "d" in parts:
            i = parts.index("d"); return parts[i+1] if i+1 < len(parts) else None
    except: pass
    return None

def _fetch_google_doc_text(doc_url: str) -> str:
    did = _doc_id_from_url(doc_url)
    if not did: raise RuntimeError("Could not parse Google Doc ID from URL.")
    r = requests.get(f"https://docs.google.com/document/d/{did}/export?format=txt", timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Google Doc fetch failed ({r.status_code}). Ensure sharing = Anyone with link.")
    return r.text

def _chunk_text(text: str, chunk_chars=1200, overlap=200) -> List[str]:
    text = " ".join(text.split()); out=[]; n=len(text)
    step=max(1, chunk_chars-overlap); start=0
    while start<n:
        out.append(text[start:min(n, start+chunk_chars)])
        start += step
    return out

class GoogleDocRetriever:
    def __init__(self, embedder: Embedder, url: Optional[str], source="google_doc"):
        self.ready=False; self.source=source; self.embedder=embedder
        self.chunks: List[str] = []; self.embs: Optional[np.ndarray] = None; self.sid=None
        if url:
            try:
                t=_fetch_google_doc_text(url)
                self.chunks=_chunk_text(t); self.embs=self.embedder.embed_many(self.chunks)
                self.sid=_doc_id_from_url(url) or "google_doc"; self.ready=True
            except Exception as e:
                print(f"[GoogleDoc] WARN: {e}")
    def search(self, query: str, k=6) -> List[Hit]:
        if not self.ready or self.embs is None: return []
        q=np.asarray(self.embedder.embed(query), dtype=float)
        sims=(self.embs@q)/(np.linalg.norm(self.embs,axis=1)*np.linalg.norm(q)+1e-12)
        idx=np.argsort(-sims)[:k]
        return [Hit(text=self.chunks[i], score=float(sims[i]), id=f"{self.source}_{self.sid}_{i}",
                    payload={"source": f"{self.source}:{self.sid}"}) for i in idx]

# -------------- Local Folder Retriever (txt/md/pdf) --------------
from pypdf import PdfReader
def _load_file_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt",".md",".rst"):
        return open(path, "r", encoding="utf-8", errors="ignore").read()
    if ext==".pdf":
        try:
            r=PdfReader(path); return " ".join((p.extract_text() or "") for p in r.pages)
        except Exception: return ""
    return ""

class LocalFolderRetriever:
    def __init__(self, embedder: Embedder, folder: str, source="local"):
        self.ready=False; self.source=source; self.embedder=embedder
        self.chunks: List[str]=[]; self.embs: Optional[np.ndarray]=None; self.index_to_src: List[str]=[]
        if os.path.isdir(folder):
            texts=[]
            for path in glob.glob(os.path.join(folder, "**/*"), recursive=True):
                if os.path.isfile(path) and os.path.splitext(path)[1].lower() in (".txt",".md",".pdf"):
                    t=_load_file_text(path); 
                    if t: texts.append((path, _chunk_text(t)))
            for path, chunks in texts:
                for ch in chunks:
                    self.chunks.append(ch); self.index_to_src.append(path)
            if self.chunks:
                self.embs=self.embedder.embed_many(self.chunks); self.ready=True
    def search(self, query: str, k=6) -> List[Hit]:
        if not self.ready or self.embs is None: return []
        q=np.asarray(self.embedder.embed(query), dtype=float)
        sims=(self.embs@q)/(np.linalg.norm(self.embs,axis=1)*np.linalg.norm(q)+1e-12)
        idx=np.argsort(-sims)[:k]
        return [Hit(text=self.chunks[i], score=float(sims[i]), id=f"{self.source}_{i}",
                    payload={"source": f"{self.source}:{self.index_to_src[i]}"}) for i in idx]

# -------------- Fusion utilities --------------
def rrf(runs: List[List[Hit]], k=50, c=60.0) -> List[Hit]:
    scores: Dict[str, float]={}; best: Dict[str,Hit]={}
    for run in runs:
        for rank,h in enumerate(run[:k],1):
            rid=h.id or f"{hash(h.text)}"; scores[rid]=scores.get(rid,0.0)+1.0/(c+rank)
            if rid not in best or (h.score or 0)> (best[rid].score or 0): best[rid]=h
    return sorted(best.values(), key=lambda h: scores[h.id or f"{hash(h.text)}"], reverse=True)

def mmr(query_vec: List[float], cands: List[Hit], embedder: Embedder, lam=0.7, top_k=12) -> List[Hit]:
    if not cands: return []
    texts=[h.text for h in cands]; M=embedder.embed_many(texts); q=np.asarray(query_vec,dtype=float)
    def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    sel,sel_idx, pool=[],[], list(range(len(cands)))
    while pool and len(sel)<top_k:
        best_i,best=-1,-1e9
        for i in pool:
            s_q=cos(M[i],q); s_rep=0.0 if not sel_idx else max(cos(M[i],M[j]) for j in sel_idx)
            score=lam*s_q - (1-lam)*s_rep
            if score>best: best=score; best_i=i
        sel.append(cands[best_i]); sel_idx.append(best_i); pool.remove(best_i)
    return sel

def kw_views(q: str, topn=6) -> List[str]:
    stop=set("a an the of to in on for with about and or from as by is are be was were that this those these it its into over under at within across between into how what why when which".split())
    toks=[t for t in re.findall(r"[A-Za-z0-9\-_/]+",q.lower()) if len(t)>2 and t not in stop]
    seen, keep=set(),[]
    for t in toks:
        if t not in seen: seen.add(t); keep.append(t)
    out=keep[:topn]
    if len(keep)>=2: out.append(" ".join(keep[:2]))
    if len(keep)>=3: out.append(" ".join(keep[:3]))
    return out[:topn+2]

# -------------- Agent Policies --------------
ANSWER_SYS="""You are a careful, citation-first assistant.
Use ONLY the provided context to answer. If insufficient, say what is missing.
Cite with [S1], [S2], ... Keep answers concise and factual."""
VERIFY_SYS='''Verify if the draft is grounded in the context.
Return JSON only: {"support":"HIGH|MEDIUM|LOW","missing":["..."]}'''
REWRITE_SYS='''You reformulate queries for retrieval. Given question + missing facts, return 4 short rewrites as a JSON list.'''

class Answerer:
    def __init__(self, llm: LLM): self.llm = llm
    def build_context(self, hits: List[Hit], max_chars=6000) -> Tuple[str,List[str]]:
        seen, used=set(),0; ctx,src=[],[]
        for i,h in enumerate(hits,1):
            t = (h.text or "").strip()
            if not t or t.lower() in seen:
                continue
            entry=f"[S{i}] {t}"
            if used+len(entry)>max_chars: break
            ctx.append(entry)
            s=(h.payload or {}).get("source_path") or (h.payload or {}).get("source") or h.id or f"chunk_{i}"
            src.append(str(s)); used+=len(entry); seen.add(t.lower())
        return "\n\n".join(ctx), src
    def answer(self, question: str, ctx: str, sources: List[str]) -> str:
        src_lines="\n".join(f"{i+1}. {s}" for i,s in enumerate(sources))
        user=f"Question:\n{question}\n\nContext:\n{ctx}\n\nSources:\n{src_lines}\n\nWrite a grounded answer with [S#] citations."
        return self.llm.complete(ANSWER_SYS, user, temperature=0.2)
    def verify(self, draft: str, ctx: str) -> Dict[str,Any]:
        raw=self.llm.complete(VERIFY_SYS, f"Draft:\n{draft}\n\nContext:\n{ctx}\n\nJSON only.", temperature=0.0)
        try: return json.loads(raw)
        except Exception: return {"support":"LOW","missing":["Unparseable verifier output"]}
    def rewrites(self, question: str, missing: List[str]) -> List[str]:
        miss="\n".join(f"- {m}" for m in (missing or []))
        raw=self.llm.complete(REWRITE_SYS, f"Question:\n{question}\n\nMissing:\n{miss}\nReturn JSON list.", temperature=0.7)
        try:
            arr=json.loads(raw)
            return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            return [question+" details", question+" examples", question+" definition", question+" process"]

# -------------- Agentic RAG Orchestrator --------------
@dataclass
class AgentConfig:
    k_per_view:int=6; rrf_k:int=30; mmr_lambda:float=0.7; final_k:int=12
    score_threshold:Optional[float]=None; must_filter:Optional[Dict[str,Any]]=None
    corrective_rounds:int=1

class AgenticRAG:
    def __init__(self, qdr: QdrantRetriever, gdoc: GoogleDocRetriever, localr: LocalFolderRetriever,
                 embedder: Embedder, llm: LLM, cfg: AgentConfig=AgentConfig()):
        self.qdr=qdr; self.gdoc=gdoc; self.localr=localr
        self.embedder=embedder; self.llm=llm; self.cfg=cfg; self.answerer=Answerer(llm)

    def _views(self, question:str)->List[str]:
        views=[question]+kw_views(question)
        # You could add an initial small rewrite burst if you want more breadth
        return list(dict.fromkeys(v.strip() for v in views if v.strip()))[:12]

    def _retrieve_all(self, view: str) -> List[Hit]:
        runs=[]
        runs.append(self.qdr.search(view, k=self.cfg.k_per_view,
                                    must_filter=self.cfg.must_filter, score_threshold=self.cfg.score_threshold))
        runs.append(self.gdoc.search(view, k=self.cfg.k_per_view) if self.gdoc else [])
        runs.append(self.localr.search(view, k=self.cfg.k_per_view) if self.localr else [])
        return sum([runs[0], runs[1], runs[2]], [])

    def _fuse(self, views: List[str]) -> List[Hit]:
        per_view_runs=[]
        for v in views:
            hits=self._retrieve_all(v)
            per_view_runs.append(hits)
        fused=rrf(per_view_runs, k=self.cfg.rrf_k)
        qvec=self.embedder.embed(" ".join(views[:1]))  # main question signal
        return mmr(qvec, fused, self.embedder, lam=self.cfg.mmr_lambda, top_k=self.cfg.final_k)

    def chat_answer(self, question: str, history: List[Dict[str,str]] | None=None) -> Dict[str,Any]:
        # 1) initial planning (use history lightly if wanted)
        views=self._views(question)
        hits=self._fuse(views)
        ctx,src=self.answerer.build_context(hits)
        draft=self.answerer.answer(question, ctx, src)
        verdict=self.answerer.verify(draft, ctx)
        if verdict.get("support","LOW") in ("HIGH","MEDIUM") or self.cfg.corrective_rounds<=0:
            return {"answer":draft, "verify":verdict, "sources":src, "views":views}

        # 2) corrective loop (agent decides to expand)
        missing=verdict.get("missing",[])
        for _ in range(self.cfg.corrective_rounds):
            rew=self.answerer.rewrites(question, missing)
            # run retrieval for rewrites too
            per_view_runs=[]
            for v in rew:
                per_view_runs.append(self._retrieve_all(v))
            # fuse all (old + new)
            fused=rrf([hits]+per_view_runs, k=self.cfg.rrf_k)
            qvec=self.embedder.embed(question)
            reranked=mmr(qvec, fused, self.embedder, lam=self.cfg.mmr_lambda, top_k=self.cfg.final_k)
            ctx,src=self.answerer.build_context(reranked)
            draft=self.answerer.answer(question, ctx, src)
            verdict=self.answerer.verify(draft, ctx)
            if verdict.get("support","LOW") in ("HIGH","MEDIUM"):
                return {"answer":draft, "verify":verdict, "sources":src, "views":views+rew}
            missing=verdict.get("missing",[])
            hits=reranked
        return {"answer":draft, "verify":verdict, "sources":src, "views":views}

# ---------------- Bootstrap helpers ----------------
def build_agent() -> AgenticRAG:
    emb=Embedder(SET.embed_model)
    qclient=QdrantClient(url=SET.qdrant_url, api_key=SET.qdrant_api_key)
    qdr=QdrantRetriever(qclient, SET.qdrant_collection, emb)
    gdoc=GoogleDocRetriever(emb, SET.google_doc_url) if SET.google_doc_url else GoogleDocRetriever(emb, None)
    localr=LocalFolderRetriever(emb, SET.local_docs_dir) if SET.local_docs_dir else LocalFolderRetriever(emb, "")
    llm=LLM(SET.openai_api_key, SET.openai_model)
    cfg=AgentConfig(k_per_view=6, rrf_k=30, mmr_lambda=0.7, final_k=12, corrective_rounds=1)
    return AgenticRAG(qdr, gdoc, localr, emb, llm, cfg)

# ---------------- FastAPI Chatbot ----------------
if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel

    agent = build_agent()
    app = FastAPI(title="Agentic RAG Chatbot")

    class ChatRequest(BaseModel):
        question: str
        history: Optional[List[Dict[str,str]]] = None

    @app.post("/ask")
    def ask(req: ChatRequest):
        out = agent.chat_answer(req.question, req.history or [])
        return out

    # Run: python agentic_rag_chatbot.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
