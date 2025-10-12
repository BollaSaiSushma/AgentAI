import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# -----------------------------
# Config
# -----------------------------
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "rag_demo_pdf_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # optional local
MAX_CONTEXT_CHARS = int(os.getenv("SELF_RAG_CONTEXT_CHARS", "2600"))
K_INITIAL = int(os.getenv("SELF_RAG_K", "8"))
K_EXPAND = int(os.getenv("SELF_RAG_K_EXPAND", "8"))
MAX_ROUNDS = int(os.getenv("SELF_RAG_MAX_ROUNDS", "3"))
TOP_SIGNAL_MIN = float(os.getenv("SELF_RAG_MIN_SIGNAL", "0.32"))

# -----------------------------
# Minimal LLM interface
# -----------------------------
class LLM:
    """Plug your model here by implementing .complete(system, user, temperature)."""
    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        raise NotImplementedError

class OpenAILLM(LLM):
    """Uses OpenAI Chat Completions if OPENAI_API_KEY is set."""
    def __init__(self, model: str = None):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()

def get_llm() -> LLM:
    if os.getenv("OPENAI_API_KEY"):
        return OpenAILLM()
    # Fallback noop (raises) to make it obvious if not configured
    class _Noop(LLM):
        def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
            raise RuntimeError("No LLM configured. Set OPENAI_API_KEY or implement LLM.complete().")
    return _Noop()

# -----------------------------
# Retrieval utils
# -----------------------------
def normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.5 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]

@dataclass
class Hit:
    id: str
    text: str
    page: Optional[int]
    chunk_index: Optional[int]
    score: float
    source: Optional[str]

class QdrantRetriever:
    def __init__(self, collection: str, embed_model: str):
        self.collection = collection
        self.model = SentenceTransformer(embed_model)
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        # sanity: try a light call
        try:
            _ = self.client.get_collections()
            self._ok = True
        except Exception as e:
            print(f"[WARN] Qdrant not reachable at {QDRANT_URL}: {e}")
            self._ok = False

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode([text], normalize_embeddings=True)[0]

    def search(self, query: str, k: int) -> List[Hit]:
        if not self._ok:
            return []
        vec = self.embed(query)
        try:
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=vec.tolist(),
                limit=k,
                with_payload=True
            )
        except Exception as e:
            print(f"[WARN] search failed: {e}")
            return []
        out: List[Hit] = []
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

    def neighbor(self, page: int, chunk_index: int) -> List[Hit]:
        """Return ±1 neighbors on same page."""
        if not self._ok:
            return []
        results: List[Hit] = []
        for neighbor_idx in (chunk_index - 1, chunk_index + 1):
            if neighbor_idx <= 0:
                continue
            flt = qm.Filter(
                must=[
                    qm.FieldCondition(key="page", match=qm.MatchValue(value=page)),
                    qm.FieldCondition(key="chunk_index", match=qm.MatchValue(value=neighbor_idx)),
                ]
            )
            pts, _ = self.client.scroll(
                collection_name=self.collection,
                limit=1,
                with_payload=True,
                with_vectors=False,
                scroll_filter=flt
            )
            if pts:
                p = pts[0]
                pl = p.payload or {}
                results.append(Hit(
                    id=str(p.id),
                    text=(pl.get("text") or ""),
                    page=pl.get("page"),
                    chunk_index=pl.get("chunk_index"),
                    score=0.0,
                    source=pl.get("source"),
                ))
        return results

# -----------------------------
# Prompts
# -----------------------------
SYSTEM_SELF_QUERY = """You rewrite questions into 1-4 focused, searchable sub-queries.
Keep them short, specific, and without punctuation except quotes when needed. Output one per line, no numbers."""
TEMPLATE_SELF_QUERY = """User Question:
{q}

Produce up to 4 sub-queries (one per line)."""

SYSTEM_DRAFTER = """You are a careful assistant that answers strictly using the provided CONTEXT.
- Cite sources inline like [p:PAGE,c:CHUNK].
- If something is not in context, say you cannot find it.
- Be concise, structured, and factual."""
TEMPLATE_DRAFTER = """QUESTION:
{q}

CONTEXT:
{context}

Write the best answer grounded in the context with inline citations.
"""

SYSTEM_CRITIC = """You are a strict fact-checker. Rate groundedness 0-1 and suggest what to retrieve next.
Return JSON only with keys: groundedness (0..1), missing_facts (array of strings), needs_more (true/false)."""
TEMPLATE_CRITIC = """QUESTION:
{q}

DRAFT:
{draft}

CONTEXT:
{context}

Respond with JSON ONLY.
"""

SYSTEM_REWRITER = """You improve the draft using the extra CONTEXT. Keep all claims grounded with citations. Be concise."""
TEMPLATE_REWRITER = """QUESTION:
{q}

EXTRA CONTEXT:
{extra}

CURRENT DRAFT:
{draft}

Revise the draft to incorporate extra evidence, with proper [p:PAGE,c:CHUNK] citations.
"""

# -----------------------------
# Self-RAG Orchestrator
# -----------------------------
@dataclass
class SelfRAGResult:
    answer: str
    citations: List[Dict[str, Any]]
    rounds: int
    debug: Dict[str, Any]

class SelfRAG:
    def __init__(self, retriever: QdrantRetriever, llm: LLM):
        self.retriever = retriever
        self.llm = llm

    def _make_context(self, hits: List[Hit]) -> Tuple[str, List[Dict[str, Any]], float]:
        """Stitch within char budget, return text + citations + top_signal."""
        if not hits:
            return "", [], 0.0
        scores = [h.score for h in hits if h.score]
        sig = max(normalize_scores(scores) or [0.0])
        parts: List[str] = []
        cites: List[Dict[str, Any]] = []
        total = 0
        for i, h in enumerate(hits, start=1):
            header = f"[{i}] {h.source or 'doc'} | p:{h.page or '?'} | c:{h.chunk_index or '?'}"
            block = f"{header}\n{(h.text or '').strip()}\n"
            if total + len(block) > MAX_CONTEXT_CHARS and parts:
                break
            parts.append(block)
            total += len(block)
            cites.append({"idx": i, "page": h.page, "chunk": h.chunk_index, "source": h.source})
        return "\n".join(parts).strip(), cites, sig

    def _self_queries(self, question: str) -> List[str]:
        out = self.llm.complete(SYSTEM_SELF_QUERY, TEMPLATE_SELF_QUERY.format(q=question))
        # one per line
        queries = [ln.strip(" -•\t") for ln in out.splitlines() if ln.strip()]
        # de-dup small variations
        uniq = []
        seen = set()
        for q in queries[:4]:
            k = q.lower()
            if k not in seen:
                seen.add(k)
                uniq.append(q)
        return uniq or [question]

    def ask(self, question: str) -> SelfRAGResult:
        # 1) generate sub-queries
        subqs = self._self_queries(question)

        # 2) initial retrieval (union of sub-query hits + neighbors)
        hits_map: Dict[str, Hit] = {}
        for sq in subqs:
            for h in self.retriever.search(sq, k=K_INITIAL):
                hits_map[h.id] = h
                # neighbors for continuity
                if h.page and h.chunk_index:
                    for nb in self.retriever.neighbor(h.page, h.chunk_index):
                        hits_map.setdefault(nb.id, nb)

        hits = list(hits_map.values())
        context, citations, top_signal = self._make_context(hits)

        # 3) initial draft
        draft = self.llm.complete(SYSTEM_DRAFTER, TEMPLATE_DRAFTER.format(q=question, context=context))

        debug = {
            "sub_queries": subqs,
            "top_signal": top_signal,
            "rounds": 1,
        }

        # If context is very weak, try to expand once even before critique
        if top_signal < TOP_SIGNAL_MIN:
            for sq in subqs:
                for h in self.retriever.search(sq, k=K_EXPAND):
                    if h.id not in hits_map:
                        hits_map[h.id] = h
                        if h.page and h.chunk_index:
                            for nb in self.retriever.neighbor(h.page, h.chunk_index):
                                hits_map.setdefault(nb.id, nb)
            hits = list(hits_map.values())
            context, citations, top_signal = self._make_context(hits)
            draft = self.llm.complete(SYSTEM_DRAFTER, TEMPLATE_DRAFTER.format(q=question, context=context))
            debug["top_signal_after_expand0"] = top_signal
            debug["rounds"] = 1

        # 4) critique + refine loop
        rounds = 1
        while rounds < MAX_ROUNDS:
            critic_json = self.llm.complete(SYSTEM_CRITIC, TEMPLATE_CRITIC.format(q=question, draft=draft, context=context))
            # naive JSON parse (no external deps)
            groundedness, needs_more, missing_list = 0.0, True, []
            try:
                import json
                obj = json.loads(critic_json)
                groundedness = float(obj.get("groundedness", 0.0))
                needs_more = bool(obj.get("needs_more", False))
                missing_list = [str(x) for x in obj.get("missing_facts", [])]
            except Exception:
                # if parse fails, assume we need one more round
                groundedness, needs_more, missing_list = 0.0, True, []

            debug[f"critic_round_{rounds}"] = {
                "groundedness": groundedness,
                "needs_more": needs_more,
                "missing": missing_list,
            }

            if not needs_more and groundedness >= 0.7:
                break  # good enough

            # 4a) expand retrieval with missing hints (or reuse sub-queries)
            queries_to_expand = missing_list or subqs
            for mq in queries_to_expand:
                for h in self.retriever.search(mq, k=K_EXPAND):
                    if h.id not in hits_map:
                        hits_map[h.id] = h
                        if h.page and h.chunk_index:
                            for nb in self.retriever.neighbor(h.page, h.chunk_index):
                                hits_map.setdefault(nb.id, nb)

            hits = list(hits_map.values())
            context, citations, top_signal = self._make_context(hits)

            # 4b) rewrite/improve the draft with extra context
            draft = self.llm.complete(SYSTEM_REWRITER, TEMPLATE_REWRITER.format(q=question, extra=context, draft=draft))

            rounds += 1
            debug["rounds"] = rounds

        # Final answer is the last draft
        answer = draft.strip()

        # Best-effort: ensure at least one citation token like [p: ,c:]
        if "[p:" not in answer:
            # attach the first citation if available
            if citations:
                c = citations[0]
                answer += f" [p:{c.get('page','?')},c:{c.get('chunk','?')}]"

        return SelfRAGResult(
            answer=answer,
            citations=citations,
            rounds=rounds,
            debug=debug,
        )

# -----------------------------
# CLI demo
# -----------------------------
if __name__ == "__main__":
    """
    Usage:
      1) Run your existing ingestion to populate Qdrant (collection: rag_demo_pdf_chunks).
      2) Set env vars as needed:
         - QDRANT_URL, QDRANT_API_KEY (if cloud)
         - OPENAI_API_KEY (if using OpenAI LLM)
         - (optional) OPENAI_MODEL
      3) Run: python self_rag_qdrant.py
    """
    load_dotenv()
    retriever = QdrantRetriever(COLLECTION_NAME, EMBED_MODEL)
    llm = get_llm()
    selfrag = SelfRAG(retriever, llm)

    print(f"Self-RAG with Qdrant @ {QDRANT_URL} | collection={COLLECTION_NAME}")
    while True:
        try:
            q = input("\nQ: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        t0 = time.time()
        res = selfrag.ask(q)
        dt = (time.time() - t0) * 1000

        print(f"\n--- ANSWER ({dt:.0f} ms, rounds={res.rounds}) ---\n{res.answer}\n")
        print("--- CITATIONS ---")
        for c in res.citations:
            print(f"[{c['idx']}] {c.get('source')} | p:{c.get('page')} | c:{c.get('chunk')}")
        print("--- DEBUG ---")
        print(res.debug)
