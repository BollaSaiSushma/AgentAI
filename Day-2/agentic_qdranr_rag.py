import os
import json
import time
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from rank_bm25 import BM25Okapi

# ==============================
# Config (env overridable)
# ==============================
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "rag_demo_pdf_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # optional for local

AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "6"))
AGENT_K = int(os.getenv("AGENT_K", "10"))
AGENT_K_EXPAND = int(os.getenv("AGENT_K_EXPAND", "8"))
MAX_CONTEXT_CHARS = int(os.getenv("AGENT_CONTEXT_CHARS", "2600"))
CRITIC_MIN_GROUNDED = float(os.getenv("AGENT_CRITIC_MIN", "0.7"))

# ==============================
# LLM plumbing (OpenAI by default)
# ==============================

class LLM:
    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        raise NotImplementedError

class OpenAILLM(LLM):
    def __init__(self, model: Optional[str] = None):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
        )
        return resp.choices[0].message.content.strip()

def get_llm() -> LLM:
    if os.getenv("OPENAI_API_KEY"):
        return OpenAILLM()
    class _NoLLM(LLM):
        def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
            raise RuntimeError("No LLM configured. Set OPENAI_API_KEY or implement LLM.complete()")
    return _NoLLM()

# ==============================
# Retrieval Tools
# ==============================

@dataclass
class Hit:
    id: str
    text: str
    page: Optional[int]
    chunk_index: Optional[int]
    score: float
    source: Optional[str]

def _normalize(scores: List[float]) -> List[float]:
    if not scores: return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9: return [0.5 for _ in scores]
    return [(s - lo)/(hi - lo) for s in scores]

class QdrantSearchTool:
    name = "qdrant.search"
    desc = "Semantic search in Qdrant. Inputs: {query:str, k:int}. Returns JSON hits."
    def __init__(self, client: QdrantClient, model: SentenceTransformer, collection: str):
        self.client = client
        self.model = model
        self.collection = collection
        self.ok = True
        try:
            _ = client.get_collections()
        except Exception as e:
            print(f"[WARN] Qdrant not reachable: {e}")
            self.ok = False
    def run(self, query: str, k: int) -> List[Hit]:
        if not self.ok: return []
        vec = self.model.encode([query], normalize_embeddings=True)[0]
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=vec.tolist(), limit=k, with_payload=True
        )
        out=[]
        for h in hits:
            pl = h.payload or {}
            out.append(Hit(
                id=str(h.id),
                text=(pl.get("text") or ""),
                page=pl.get("page"),
                chunk_index=pl.get("chunk_index"),
                score=float(h.score or 0.0),
                source=pl.get("source"),
            ))
        return out

class QdrantNeighborsTool:
    name = "qdrant.neighbors"
    desc = "Get ±1 neighbor chunks by page/chunk_index. Inputs: {page:int, chunk_index:int}."
    def __init__(self, client: QdrantClient, collection: str):
        self.client = client
        self.collection = collection
    def run(self, page: int, chunk_index: int) -> List[Hit]:
        out=[]
        for idx in (chunk_index-1, chunk_index+1):
            if idx <= 0: continue
            flt = qm.Filter(must=[
                qm.FieldCondition(key="page", match=qm.MatchValue(value=page)),
                qm.FieldCondition(key="chunk_index", match=qm.MatchValue(value=idx)),
            ])
            pts, _ = self.client.scroll(
                collection_name=self.collection, limit=1, with_payload=True, with_vectors=False, scroll_filter=flt
            )
            if pts:
                p=pts[0]; pl=p.payload or {}
                out.append(Hit(
                    id=str(p.id),
                    text=(pl.get("text") or ""),
                    page=pl.get("page"),
                    chunk_index=pl.get("chunk_index"),
                    score=0.0, source=pl.get("source"),
                ))
        return out

class BM25SearchTool:
    name = "bm25.search"
    desc = "Lexical BM25 over the corpus cached from Qdrant payloads. Inputs: {query:str, k:int}."
    def __init__(self, client: QdrantClient, collection: str, cap: int = 100000):
        self.client = client; self.collection = collection; self.cap = cap
        self.idx_tokens : List[List[str]] = []
        self.meta : List[Dict[str,Any]] = []
        self._build()
    def _tok(self, t: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9_]+", t.lower())
    def _build(self):
        seen=0; next_page=None
        while True:
            pts, next_page = self.client.scroll(
                collection_name=self.collection, limit=256, with_payload=True, with_vectors=False, offset=next_page
            )
            if not pts: break
            for p in pts:
                pl=p.payload or {}; txt=(pl.get("text") or "").strip()
                if not txt: continue
                self.idx_tokens.append(self._tok(txt))
                self.meta.append({
                    "id": str(p.id),
                    "text": txt,
                    "page": pl.get("page"),
                    "chunk_index": pl.get("chunk_index"),
                    "source": pl.get("source"),
                })
                seen += 1
                if seen >= self.cap: break
            if not next_page or seen >= self.cap: break
        if not self.idx_tokens:
            self.bm25=None
        else:
            self.bm25 = BM25Okapi(self.idx_tokens)
    def run(self, query: str, k: int) -> List[Hit]:
        if not getattr(self, "bm25", None): return []
        scores = self.bm25.get_scores(self._tok(query))
        idxs = np.argsort(scores)[::-1][:k]
        out=[]
        for i in idxs:
            m=self.meta[i]
            out.append(Hit(
                id=m["id"], text=m["text"], page=m["page"], chunk_index=m["chunk_index"],
                score=float(scores[i]), source=m["source"]
            ))
        return out

# ==============================
# Agent State & Memory
# ==============================
@dataclass
class Evidence:
    hits: Dict[str, Hit] = field(default_factory=dict)
    def add_many(self, hs: List[Hit]):
        for h in hs:
            if h and h.id not in self.hits:
                self.hits[h.id]=h
    def to_sorted(self) -> List[Hit]:
        arr=list(self.hits.values())
        arr.sort(key=lambda h: h.score, reverse=True)
        return arr

def stitch_context(hits: List[Hit], limit_chars: int) -> Tuple[str, List[Dict[str,Any]], float]:
    if not hits: return "", [], 0.0
    scores = [h.score for h in hits if h.score]
    top_signal = max(_normalize(scores) or [0.0])
    parts=[]; cites=[]; total=0
    for i, h in enumerate(hits, start=1):
        header = f"[{i}] {h.source or 'doc'} | p:{h.page or '?'} | c:{h.chunk_index or '?'}"
        block = f"{header}\n{(h.text or '').strip()}\n"
        if total + len(block) > limit_chars and parts:
            break
        parts.append(block); total += len(block)
        cites.append({"idx": i, "page": h.page, "chunk": h.chunk_index, "source": h.source})
    return "\n".join(parts).strip(), cites, top_signal

# ==============================
# Agent Prompts
# ==============================
SYSTEM_PLANNER = """You are a research planner. Decide the next best ACTION as JSON.
Choose only from:
- {"tool":"qdrant.search","args":{"query": "...", "k": 10}}
- {"tool":"bm25.search","args":{"query": "...", "k": 10}}
- {"tool":"qdrant.neighbors","args":{"page": 3, "chunk_index": 5}}
- {"tool":"stop","args":{}}

Guidelines:
- Start with 1-3 focused sub-queries.
- Prefer qdrant.search; if query is short/code-like/numeric, also use bm25.search.
- After you get promising hits with page+chunk_index, pull neighbors once.
- Stop when you likely have enough to answer.
Return JSON ONLY."""

TEMPLATE_PLANNER = """QUESTION:
{q}

OBSERVATION (latest tool results summary):
{obs}
"""

SYSTEM_SYNTHESIZER = """You write an answer using only the CONTEXT and include inline citations like [p:PAGE,c:CHUNK]. If info not present, state that plainly."""
TEMPLATE_SYNTHESIZER = """QUESTION:
{q}

CONTEXT:
{context}

Answer concisely with citations.
"""

SYSTEM_CRITIC = """You are a strict critic. Score groundedness from 0 to 1 and suggest missing evidence as a list. Return JSON: {"groundedness":0.0,"needs_more":true/false,"missing":[...]}"""
TEMPLATE_CRITIC = """QUESTION:
{q}

DRAFT:
{draft}

CONTEXT:
{context}
"""

SYSTEM_REWRITER = """You refine the draft using EXTRA CONTEXT. Keep only grounded claims and ensure citations cover claims."""
TEMPLATE_REWRITER = """QUESTION:
{q}

EXTRA CONTEXT:
{extra}

CURRENT DRAFT:
{draft}
"""

# ==============================
# Agent Orchestrator
# ==============================
class AgenticRAG:
    def __init__(self):
        load_dotenv()
        self.llm = get_llm()
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        # Tools
        self.t_qsearch = QdrantSearchTool(self.client, self.embedder, COLLECTION_NAME)
        self.t_neighbors = QdrantNeighborsTool(self.client, COLLECTION_NAME)
        self.t_bm25 = BM25SearchTool(self.client, COLLECTION_NAME)
        # Memory
        self.evidence = Evidence()

    # ---- Tool dispatcher ----
    def _run_tool(self, action: Dict[str,Any]) -> Tuple[str, List[Hit]]:
        tool = action.get("tool")
        args = action.get("args", {})
        try:
            if tool == "qdrant.search":
                q = args.get("query",""); k=int(args.get("k", AGENT_K))
                hits = self.t_qsearch.run(q, k)
                return f"qdrant.search returned {len(hits)} hits for '{q}'", hits
            if tool == "bm25.search":
                q = args.get("query",""); k=int(args.get("k", AGENT_K))
                hits = self.t_bm25.run(q, k)
                return f"bm25.search returned {len(hits)} hits for '{q}'", hits
            if tool == "qdrant.neighbors":
                page=int(args.get("page")); ci=int(args.get("chunk_index"))
                hits = self.t_neighbors.run(page, ci)
                return f"neighbors for p:{page} c:{ci} -> {len(hits)}", hits
            if tool == "stop":
                return "stop", []
        except Exception as e:
            return f"tool error: {e}", []
        return "unknown tool", []

    def _summarize_observation(self, new_hits: List[Hit], last_note: str) -> str:
        if not new_hits:
            return last_note
        # brief peek at first two
        previews=[]
        for h in new_hits[:2]:
            t=(h.text or "")
            previews.append(f"(p:{h.page},c:{h.chunk_index}) {t[:120].replace('\\n',' ')}")
        return last_note + " | " + " ; ".join(previews)

    def _plan_next(self, question: str, obs: str) -> Dict[str,Any]:
        plan_raw = self.llm.complete(SYSTEM_PLANNER, TEMPLATE_PLANNER.format(q=question, obs=obs))
        # keep only json (best effort)
        try:
            j = json.loads(plan_raw)
            # normalize single-action or multi-actions
            if isinstance(j, dict) and "tool" in j:
                return j
            if isinstance(j, list) and j and isinstance(j[0], dict):
                # take the first action for simplicity
                return j[0]
        except Exception:
            pass
        # If it fails to produce JSON, do a safe default
        return {"tool":"qdrant.search","args":{"query":question,"k":AGENT_K}}

    def ask(self, question: str) -> Dict[str,Any]:
        self.evidence = Evidence()
        observation = "start"
        steps=0
        stopped=False

        # ---- Agent loop ----
        while steps < AGENT_MAX_STEPS:
            action = self._plan_next(question, observation)
            if action.get("tool") == "stop":
                stopped=True
                break

            note, hits = self._run_tool(action)
            self.evidence.add_many(hits)
            observation = self._summarize_observation(hits, note)
            steps += 1

            # Heuristic: once we have enough with page+chunk, pull one neighbor round then stop
            have_struct = [h for h in self.evidence.hits.values() if h.page and h.chunk_index]
            if have_struct and steps < AGENT_MAX_STEPS:
                # one neighbor expansion then allow planner to decide stop
                nb = have_struct[0]
                note2, hits2 = self._run_tool({"tool":"qdrant.neighbors","args":{"page": nb.page, "chunk_index": nb.chunk_index}})
                self.evidence.add_many(hits2)
                observation = self._summarize_observation(hits2, note2)
                steps += 1

        # ---- Synthesize ----
        hits_sorted = self.evidence.to_sorted()
        context, citations, top_signal = stitch_context(hits_sorted, MAX_CONTEXT_CHARS)
        draft = self.llm.complete(SYSTEM_SYNTHESIZER, TEMPLATE_SYNTHESIZER.format(q=question, context=context))

        # ---- Critic loop (one pass, optional expand) ----
        critic = self.llm.complete(SYSTEM_CRITIC, TEMPLATE_CRITIC.format(q=question, draft=draft, context=context))
        groundedness, needs_more, missing = 0.0, False, []
        try:
            obj=json.loads(critic)
            groundedness=float(obj.get("groundedness",0.0))
            needs_more=bool(obj.get("needs_more",False))
            missing=[str(x) for x in obj.get("missing",[])]
        except Exception:
            groundedness=0.0; needs_more=True; missing=[]

        if needs_more and groundedness < CRITIC_MIN_GROUNDED:
            # expand with missing hints or reuse question
            queries = missing or [question]
            for mq in queries[:3]:
                _, h1 = self._run_tool({"tool":"qdrant.search","args":{"query": mq, "k": AGENT_K_EXPAND}})
                self.evidence.add_many(h1)
            hits_sorted = self.evidence.to_sorted()
            context, citations, _ = stitch_context(hits_sorted, MAX_CONTEXT_CHARS)
            draft = self.llm.complete(SYSTEM_REWRITER, TEMPLATE_REWRITER.format(q=question, extra=context, draft=draft))

        # safety: ensure at least one citation token
        if "[p:" not in draft and citations:
            c=citations[0]
            draft += f" [p:{c.get('page','?')},c:{c.get('chunk','?')}]"

        return {
            "answer": draft.strip(),
            "citations": citations,
            "debug": {
                "steps": steps,
                "stopped": stopped,
                "top_signal": top_signal,
                "observation_tail": observation[-240:],
                "groundedness": groundedness,
                "missing": missing,
            }
        }

# ==============================
# CLI demo
# ==============================
if __name__ == "__main__":
    load_dotenv()
    rag = AgenticRAG()
    print(f"Agentic RAG ready. Qdrant={QDRANT_URL} collection={COLLECTION_NAME}")
    while True:
        try:
            q = input("\nQ: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit","quit"}: break
        t0=time.time()
        res = rag.ask(q)
        dt=(time.time()-t0)*1000
        print(f"\n--- ANSWER ({dt:.0f} ms) ---\n{res['answer']}\n")
        print("--- CITATIONS ---")
        for c in res["citations"]:
            print(f"[{c['idx']}] {c.get('source')} | p:{c.get('page')} | c:{c.get('chunk')}")
        print("--- DEBUG ---")
        print(res["debug"])
