import os
import re
import math
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


# -----------------------------
# Config
# -----------------------------
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "rag_demo_pdf_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # optional for local

# Retrieval defaults
DEFAULT_K = 12
MAX_NEIGHBORS_PER_HIT = 1   # on each side (±1)
SCORE_MIN_CONFIDENCE = 0.35  # lower means stricter gate
BM25_CORPUS_MAX = 100000     # safety cap


# -----------------------------
# Small utilities
# -----------------------------
_token_pat = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _token_pat.findall(text or "")]

def normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.5 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]

def is_keywordish(q: str) -> bool:
    # If the query looks like exact match / code / numbers / short noun phrase
    has_quote = '"' in q or "'" in q
    has_digits = any(c.isdigit() for c in q)
    has_symbols = any(c in "-_:/\\." for c in q)
    shortish = len(q.split()) <= 6
    return has_quote or has_digits or has_symbols or shortish

def pick_k_by_margin(sim_scores: List[float], base_k: int = DEFAULT_K) -> int:
    """If the top is much stronger than the rest, retrieve fewer; otherwise more."""
    if not sim_scores:
        return base_k
    sims = sorted(sim_scores, reverse=True)
    top = sims[0]
    second = sims[1] if len(sims) > 1 else 0.0
    gap = top - second
    if gap > 0.25:
        return max(4, base_k // 2)
    elif gap < 0.08:
        return min(24, base_k + 8)
    return base_k

def mmr(
    query_vec: np.ndarray,
    cand_vecs: np.ndarray,
    lamb: float = 0.65,
    top_k: int = 10
) -> List[int]:
    """Maximal Marginal Relevance; returns indices into cand_vecs."""
    if len(cand_vecs) == 0:
        return []
    selected = []
    remaining = list(range(len(cand_vecs)))
    # Precompute similarities
    qsim = cand_vecs @ query_vec  # cosine if normalized
    dsim = cand_vecs @ cand_vecs.T
    # Normalize to [0,1]
    qsim_n = normalize_scores(qsim.tolist())
    dsim = (dsim + 1) / 2.0  # [-1,1] -> [0,1] heuristic
    while remaining and len(selected) < top_k:
        mmr_scores = []
        for i in remaining:
            redundancy = 0.0
            if selected:
                redundancy = max(dsim[i, j] for j in selected)
            score = lamb * qsim_n[i] - (1 - lamb) * redundancy
            mmr_scores.append((score, i))
        mmr_scores.sort(reverse=True)
        _, best = mmr_scores[0]
        selected.append(best)
        remaining.remove(best)
    return selected


# -----------------------------
# Models
# -----------------------------
@dataclass
class RetrievedChunk:
    point_id: str
    text: str
    page: Optional[int]
    chunk_index: Optional[int]
    score_sem: float
    score_lex: float
    source: Optional[str]

@dataclass
class AdaptiveResult:
    query: str
    context: str
    citations: List[Dict[str, Any]]  # [{"source":..., "page":..., "chunk_index":..., "score":...}, ...]
    used_strategy: str
    debug: Dict[str, Any]


# -----------------------------
# Corpus cache for BM25
# -----------------------------
class CorpusCache:
    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.docs_tokens: List[List[str]] = []
        self.meta: List[Dict[str, Any]] = []  # aligns with docs_tokens index

    def build(self, client: QdrantClient, collection: str):
        docs_tokens: List[List[str]] = []
        meta: List[Dict[str, Any]] = []

        scroll_filter = None
        next_page = None
        seen = 0

        while True:
            points, next_page = client.scroll(
                collection_name=collection,
                limit=256,
                with_payload=True,
                with_vectors=False,
                offset=next_page
            )
            if not points:
                break
            for p in points:
                payload = p.payload or {}
                text = (payload.get("text") or "").strip()
                if not text:
                    continue
                docs_tokens.append(tokenize(text))
                meta.append({
                    "id": str(p.id),
                    "text": text,
                    "page": payload.get("page"),
                    "chunk_index": payload.get("chunk_index"),
                    "source": payload.get("source"),
                })
                seen += 1
                if seen >= BM25_CORPUS_MAX:
                    break
            if not next_page or seen >= BM25_CORPUS_MAX:
                break

        if not docs_tokens:
            raise RuntimeError("BM25 corpus build found no documents in Qdrant.")

        self.docs_tokens = docs_tokens
        self.meta = meta
        self.bm25 = BM25Okapi(docs_tokens)


# -----------------------------
# Adaptive Retriever
# -----------------------------
class AdaptiveRetriever:
    def __init__(self):
        load_dotenv()
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.model = SentenceTransformer(EMBED_MODEL)
        self.corpus = CorpusCache()
        # Build BM25 once (lazy)
        self._bm25_ready = False

    def _ensure_bm25(self):
        if not self._bm25_ready:
            self.corpus.build(self.client, COLLECTION_NAME)
            self._bm25_ready = True

    def _semantic_search(self, query_vec: np.ndarray, limit: int) -> List[RetrievedChunk]:
        hits = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec.tolist(),
            limit=limit,
            with_payload=True
        )
        out: List[RetrievedChunk] = []
        for h in hits:
            pl = h.payload or {}
            out.append(
                RetrievedChunk(
                    point_id=str(h.id),
                    text=(pl.get("text") or ""),
                    page=pl.get("page"),
                    chunk_index=pl.get("chunk_index"),
                    score_sem=float(h.score or 0.0),
                    score_lex=0.0,
                    source=pl.get("source"),
                )
            )
        return out

    def _lexical_search(self, query: str, limit: int) -> List[RetrievedChunk]:
        self._ensure_bm25()
        tokens = tokenize(query)
        scores = self.corpus.bm25.get_scores(tokens)  # type: ignore
        # Take top-N indices
        idxs = np.argsort(scores)[::-1][:limit]
        out: List[RetrievedChunk] = []
        for i in idxs:
            m = self.corpus.meta[i]
            out.append(
                RetrievedChunk(
                    point_id=m["id"],
                    text=m["text"],
                    page=m["page"],
                    chunk_index=m["chunk_index"],
                    score_sem=0.0,
                    score_lex=float(scores[i]),
                    source=m["source"],
                )
            )
        return out

    def _combine_hybrid(
        self,
        sem: List[RetrievedChunk],
        lex: List[RetrievedChunk],
        w_sem: float,
        w_lex: float,
        query_vec: np.ndarray,
        top_k: int
    ) -> List[RetrievedChunk]:
        # Merge by point_id, keep best text/payload
        by_id: Dict[str, RetrievedChunk] = {}
        for s in sem:
            by_id[s.point_id] = s
        for l in lex:
            if l.point_id in by_id:
                # merge scores
                cur = by_id[l.point_id]
                cur.score_lex = max(cur.score_lex, l.score_lex)
            else:
                by_id[l.point_id] = l

        items = list(by_id.values())
        # Normalize channels independently
        sem_scores = normalize_scores([x.score_sem for x in items])
        lex_scores = normalize_scores([x.score_lex for x in items])

        # Combined score
        combined = [w_sem * s + w_lex * l for s, l in zip(sem_scores, lex_scores)]
        # Prepare for MMR: use embeddings to diversify
        cand_vecs = self.model.encode([it.text for it in items], normalize_embeddings=True)
        order = mmr(query_vec, cand_vecs, lamb=0.7, top_k=min(top_k, len(items)))
        ranked = [items[i] for i in order]
        # Attach combined score into score_sem for convenience in output
        for i, idx in enumerate(order):
            items[idx].score_sem = combined[idx]
        return ranked

    def _expand_neighbors(self, hits: List[RetrievedChunk], window: int = MAX_NEIGHBORS_PER_HIT) -> List[RetrievedChunk]:
        if window <= 0:
            return hits

        expanded: Dict[str, RetrievedChunk] = {h.point_id: h for h in hits}
        for h in hits:
            if h.page is None or h.chunk_index is None:
                continue
            for offset in range(1, window + 1):
                for neighbor_idx in (h.chunk_index - offset, h.chunk_index + offset):
                    flt = qm.Filter(
                        must=[
                            qm.FieldCondition(key="page", match=qm.MatchValue(value=h.page)),
                            qm.FieldCondition(key="chunk_index", match=qm.MatchValue(value=neighbor_idx)),
                        ]
                    )
                    # scroll one neighbor
                    points, _ = self.client.scroll(
                        collection_name=COLLECTION_NAME,
                        limit=1,
                        with_payload=True,
                        with_vectors=False,
                        scroll_filter=flt
                    )
                    if points:
                        p = points[0]
                        if str(p.id) not in expanded:
                            pl = p.payload or {}
                            expanded[str(p.id)] = RetrievedChunk(
                                point_id=str(p.id),
                                text=(pl.get("text") or ""),
                                page=pl.get("page"),
                                chunk_index=pl.get("chunk_index"),
                                score_sem=0.0,
                                score_lex=0.0,
                                source=pl.get("source"),
                            )
        return list(expanded.values())

    def _decide_weights(self, query: str) -> Tuple[float, float, str]:
        """Return (w_sem, w_lex, strategy_label)."""
        q = query.strip()
        # Heuristics:
        if is_keywordish(q):
            # favor lexical if query is short/specific/contains digits/symbols
            return (0.35, 0.65, "hybrid_lexical_tilt")
        if len(q.split()) > 20:
            # long question likely benefits from semantic
            return (0.75, 0.25, "hybrid_semantic_tilt")
        # default balanced
        return (0.55, 0.45, "hybrid_balanced")

    def ask(self, query: str, base_k: int = DEFAULT_K, max_context_chars: int = 2200) -> AdaptiveResult:
        q = query.strip()
        if not q:
            raise ValueError("Query is empty.")

        q_vec = self.model.encode([q], normalize_embeddings=True)[0]
        # quick semantic probe to choose k adaptively
        probe = self._semantic_search(q_vec, limit=max(8, base_k // 2))
        sem_probe_scores = [h.score_sem for h in probe]
        k = pick_k_by_margin(sem_probe_scores, base_k=base_k)

        # strategy + weights
        w_sem, w_lex, label = self._decide_weights(q)

        # run both channels (we’ll prune later)
        sem_hits = self._semantic_search(q_vec, limit=max(k * 2, 20))
        lex_hits = self._lexical_search(q, limit=max(k * 2, 20))

        # combine + mmr re-rank
        ranked = self._combine_hybrid(sem_hits, lex_hits, w_sem, w_lex, q_vec, top_k=k)

        # neighbor expansion for continuity
        expanded = self._expand_neighbors(ranked, window=MAX_NEIGHBORS_PER_HIT)

        # final compaction: re-score expanded set quickly by semantic to preserve relevance
        # (use dot with normalized vectors)
        all_texts = [r.text for r in expanded]
        all_vecs = self.model.encode(all_texts, normalize_embeddings=True)
        sim = (all_vecs @ q_vec).tolist()
        sim_n = normalize_scores(sim)
        pairs = list(zip(sim_n, expanded))
        pairs.sort(key=lambda x: x[0], reverse=True)

        # answerability gate
        top_signal = pairs[0][0] if pairs else 0.0
        if top_signal < SCORE_MIN_CONFIDENCE or not pairs:
            return AdaptiveResult(
                query=q,
                context="",
                citations=[],
                used_strategy=label,
                debug={
                    "top_signal": top_signal,
                    "k": k,
                    "reason": "low_confidence",
                }
            )

        # build context within char budget
        chosen: List[RetrievedChunk] = []
        total_chars = 0
        for s, r in pairs:
            if total_chars + len(r.text) > max_context_chars and chosen:
                break
            chosen.append(r)
            total_chars += len(r.text)

        # dedupe by (page, chunk_index) and preserve order
        seen_keys = set()
        final_chunks: List[RetrievedChunk] = []
        for r in chosen:
            key = (r.page, r.chunk_index)
            if key not in seen_keys:
                seen_keys.add(key)
                final_chunks.append(r)

        # stitch context with neat separators
        parts = []
        citations = []
        for i, r in enumerate(final_chunks, start=1):
            header = f"[{i}] {r.source or 'doc'} | Page {r.page or '?'} | Chunk {r.chunk_index or '?'}"
            parts.append(header)
            parts.append(r.text.strip())
            parts.append("")  # blank line
            citations.append({
                "idx": i,
                "source": r.source,
                "page": r.page,
                "chunk_index": r.chunk_index,
            })

        context = "\n".join(parts).strip()

        return AdaptiveResult(
            query=q,
            context=context,
            citations=citations,
            used_strategy=label,
            debug={
                "k": k,
                "top_signal": top_signal,
                "sem_probe_scores": sem_probe_scores[:5],
                "counts": {
                    "semantic": len(sem_hits),
                    "lexical": len(lex_hits),
                    "expanded_final": len(final_chunks),
                }
            }
        )


# -----------------------------
# CLI demo
# -----------------------------
if __name__ == "__main__":
    """
    Usage:
      1) Ensure your collection exists and is populated by your ingestion script.
      2) Set env vars if needed:
         - QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, EMBED_MODEL
      3) Run: python adaptive_rag_qdrant.py
    """
    load_dotenv()
    retriever = AdaptiveRetriever()
    print(f"Connected to Qdrant at {QDRANT_URL}, collection={COLLECTION_NAME}")
    print("Embedding model:", EMBED_MODEL)
    print("\nType a question (or 'exit'):\n")
    while True:
        q = input("Q: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        t0 = time.time()
        res = retriever.ask(q)
        dt = time.time() - t0

        print("\n=== ADAPTIVE RETRIEVAL ===")
        print("Strategy:", res.used_strategy)
        print("Top signal:", f"{res.debug.get('top_signal', 0):.3f}")
        print("k:", res.debug.get("k"))
        print(f"(retrieval {dt*1000:.0f} ms)\n")

        if not res.context:
            print("No strong matches. Try broadening the query or using different keywords.\n")
            continue

        print("----- CONTEXT (for your LLM) -----\n")
        print(res.context)
        print("\n----- CITATIONS -----")
        for c in res.citations:
            print(f"[{c['idx']}] {c.get('source')} | Page {c.get('page')} | Chunk {c.get('chunk_index')}")
        print("\n")
