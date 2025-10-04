#!/usr/bin/env python3
"""
Simple RAG chunking toolbox for a PDF.

Included methods:
  - fixed: fixed-size character chunks with overlap
  - sentence: sentence-based sliding window
  - semantic: greedy sentence grouping by semantic similarity (uses sentence-transformers if available; falls back to TF-IDF if scikit-learn is available)
  - regex: split by a regex boundary, then trim with fixed-size if needed
  - hierarchical: headings -> paragraphs -> sentences -> fixed-size fallback

Outputs: JSONL files per method in --outdir.

Usage:
  pip install PyPDF2
  # (optional for semantic): pip install sentence-transformers  OR  pip install scikit-learn
  python rag_chunking_simple_plus.py --pdf rag_demo_corpus.pdf --outdir ./out_chunks
"""

import argparse
import json
import os
import re
from typing import List, Tuple

# -------------------- Optional imports for semantic chunking --------------------
_HAS_ST = False
_HAS_SK = False
_ST_MODEL = None

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _HAS_ST = True
except Exception:
    try:
        # Try to see if numpy is available anyway for TF-IDF cosine
        import numpy as np  # type: ignore
    except Exception:
        pass

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    _HAS_SK = True
except Exception:
    pass

# -------------------- Required import --------------------
try:
    from PyPDF2 import PdfReader
except Exception:
    raise SystemExit("PyPDF2 is required. Install with: pip install PyPDF2")


# -------------------- I/O and normalization --------------------
def load_pdf_text(path: str) -> List[Tuple[int, str]]:
    """Return list of (page_number_1_based, text)."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i, text))
    return pages


def normalize_whitespace(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    lines = [ln.strip() for ln in s.split("\n")]
    return "\n".join(lines).strip()


def join_pages(pages: List[Tuple[int, str]]) -> str:
    return "\n\n".join(normalize_whitespace(t) for _, t in pages)


# -------------------- Sentence Splitter --------------------
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])(?:(?:\"|'|\))?)(?=\s+(?=[A-Z0-9\[]))"
)

def split_sentences(text: str) -> List[str]:
    protected = ["e.g.", "i.e.", "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "vs.", "etc.", "Fig.", "Eq."]
    tmp = text
    repl = {}
    for idx, abbr in enumerate(sorted(protected, key=len, reverse=True)):
        token = f"<ABBR_{idx}>"
        repl[token] = abbr
        tmp = tmp.replace(abbr, token)
    parts = [p.strip() for p in re.split(_SENTENCE_BOUNDARY, tmp) if p and p.strip()]
    out = []
    for p in parts:
        for token, abbr in repl.items():
            p = p.replace(token, abbr)
        out.append(p)
    return out


# -------------------- Heading Splitter --------------------
HEADING_PATTERNS = [
    re.compile(r"^(?:\d+(?:\.\d+)*)\s+[A-Z].+", re.MULTILINE),  # 1. / 1.1 style
    re.compile(r"^(?:#{1,6})\s+.+", re.MULTILINE),              # Markdown-style
    re.compile(r"^[A-Z][A-Z \-\d]{6,}$", re.MULTILINE),         # ALL CAPS
    re.compile(r"^(?:Section|Chapter)\s+\d+[:.)-]\s+.+", re.MULTILINE),
]

def split_by_headings(text: str) -> List[str]:
    marker = "\n<SECTION_BREAK>\n"
    marked = text
    for pat in HEADING_PATTERNS:
        marked = re.sub(pat, lambda m: marker + m.group(0), marked)
    parts = [p.strip() for p in marked.split(marker) if p.strip()]
    return parts if parts else [text]


# -------------------- Chunkers --------------------
def chunk_fixed_chars(text: str, size=1000, overlap=150) -> List[dict]:
    size = max(1, size)
    overlap = max(0, overlap)
    step = max(1, size - overlap)
    out = []
    for i in range(0, len(text), step):
        piece = text[i:i+size]
        if piece:
            out.append({"id": f"fixed_{i}", "content": piece, "start": i, "end": i + len(piece)})
    return out


def chunk_sentences_sliding(text: str, window=6, overlap=2) -> List[dict]:
    sents = split_sentences(text)
    step = max(1, window - overlap)
    out = []
    for i in range(0, len(sents), step):
        win = sents[i:i+window]
        if win:
            piece = " ".join(win)
            out.append({"id": f"sent_{i}", "content": piece, "start_sent": i, "end_sent": i + len(win)})
    return out


def chunk_regex(text: str, pattern: str, max_chars=1000, overlap=150) -> List[dict]:
    """Split on regex boundaries; if a part is still too large, trim with fixed-size."""
    try:
        reg = re.compile(pattern, re.MULTILINE)
    except re.error as e:
        raise SystemExit(f"Invalid regex pattern: {e}")
    marker = "\n<REGEX_BREAK>\n"
    marked = re.sub(reg, lambda m: marker + m.group(0), text)
    parts = [p.strip() for p in marked.split(marker) if p.strip()] or [text]
    out = []
    for idx, part in enumerate(parts):
        if len(part) <= max_chars:
            out.append({"id": f"regex_{idx}", "content": part})
        else:
            out.extend(chunk_fixed_chars(part, size=max_chars, overlap=overlap))
    return out


def chunk_hierarchical(text: str, max_chars=1000, overlap=150) -> List[dict]:
    out = []
    parts = split_by_headings(text)
    for part in parts:
        if len(part) <= max_chars:
            out.append({"id": f"hier_h_{len(out)}", "content": part, "level": 1})
        else:
            paras = [p.strip() for p in part.split("\n\n") if p.strip()]
            for para in paras:
                if len(para) <= max_chars:
                    out.append({"id": f"hier_p_{len(out)}", "content": para, "level": 2})
                else:
                    sents = split_sentences(para)
                    # group sentences up to max_chars
                    cur, cur_len = [], 0
                    for s in sents:
                        if cur_len + len(s) + 1 > max_chars:
                            if cur:
                                piece = " ".join(cur)
                                out.append({"id": f"hier_s_{len(out)}", "content": piece, "level": 3})
                            cur = [s]
                            cur_len = len(s)
                        else:
                            cur.append(s)
                            cur_len += len(s) + 1
                    if cur:
                        piece = " ".join(cur)
                        out.append({"id": f"hier_s_{len(out)}", "content": piece, "level": 3})
    # final fallback for oversize chunks
    final = []
    for ch in out:
        c = ch["content"]
        if len(c) > max_chars:
            final.extend(chunk_fixed_chars(c, size=max_chars, overlap=overlap))
        else:
            final.append(ch)
    return final


# -------------------- Semantic chunking --------------------
def _ensure_st_model(model_name: str = "all-MiniLM-L6-v2"):
    global _ST_MODEL
    if _ST_MODEL is None:
        _ST_MODEL = SentenceTransformer(model_name)
    return _ST_MODEL

def _cosine(a, b):
    # a, b are 1D numpy arrays
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

def chunk_semantic(text: str, max_chars=1000, similarity_threshold=0.65) -> List[dict]:
    """
    Greedy sentence grouping by semantic similarity to current chunk centroid.
    - If sentence similarity drops below threshold or size limit reached, start a new chunk.
    - Uses sentence-transformers if available; else TF-IDF (if scikit-learn available).
    - If neither is available, falls back to sentence windowing (non-semantic).
    """
    sents = split_sentences(text)
    if not sents:
        return []

    if _HAS_ST:
        model = _ensure_st_model()
        vecs = model.encode(sents, convert_to_numpy=True, normalize_embeddings=True)
        def rep_mean(vec_list):
            return np.mean(np.stack(vec_list, axis=0), axis=0)
    elif _HAS_SK and 'np' in globals():
        # TF-IDF vectors; use cosine from sklearn
        tfidf = TfidfVectorizer().fit_transform(sents)  # sparse
        # convert rows to dense for small docs
        vecs = tfidf.toarray()
        def rep_mean(vec_list):
            return np.mean(np.stack(vec_list, axis=0), axis=0)
    else:
        # Fallback: non-semantic sentence sliding window
        return chunk_sentences_sliding(text, window=6, overlap=2)

    out = []
    cur_idxs = []
    cur_chars = 0
    cur_rep = None

    for i, s in enumerate(sents):
        v = vecs[i]
        s_len = len(s)
        if not cur_idxs:
            cur_idxs = [i]
            cur_chars = s_len
            cur_rep = v
            continue

        # check length first
        if cur_chars + s_len + 1 > max_chars:
            piece = " ".join(sents[cur_idxs[0]:cur_idxs[-1]+1])
            out.append({"id": f"sem_{len(out)}", "content": piece, "start_sent": cur_idxs[0], "end_sent": cur_idxs[-1]+1})
            cur_idxs = [i]
            cur_chars = s_len
            cur_rep = v
            continue

        # check semantic similarity
        sim = _cosine(cur_rep, v)
        if sim < similarity_threshold:
            piece = " ".join(sents[cur_idxs[0]:cur_idxs[-1]+1])
            out.append({"id": f"sem_{len(out)}", "content": piece, "start_sent": cur_idxs[0], "end_sent": cur_idxs[-1]+1})
            cur_idxs = [i]
            cur_chars = s_len
            cur_rep = v
        else:
            cur_idxs.append(i)
            cur_chars += s_len + 1
            # update centroid
            if _HAS_ST or _HAS_SK:
                # efficient running mean of vectors
                n = len(cur_idxs)
                cur_rep = (cur_rep * (n - 1) + v) / n

    if cur_idxs:
        piece = " ".join(sents[cur_idxs[0]:cur_idxs[-1]+1])
        out.append({"id": f"sem_{len(out)}", "content": piece, "start_sent": cur_idxs[0], "end_sent": cur_idxs[-1]+1})

    return out


# -------------------- Main --------------------
def write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to PDF")
    ap.add_argument("--outdir", default="./out_chunks", help="Output directory")
    # sizes/overlaps
    ap.add_argument("--max_chars", type=int, default=1000, help="Max characters per chunk")
    ap.add_argument("--overlap", type=int, default=150, help="Character overlap (fixed / hierarchical fallback)")
    ap.add_argument("--sent_window", type=int, default=6, help="Sentences/window for the fallback sentence method")
    ap.add_argument("--sent_overlap", type=int, default=2, help="Sentence overlap for the fallback sentence method")
    # regex
    ap.add_argument("--regex", default=r"^(?:\d+(?:\.\d+)*)\s+[A-Z].+", help="Regex boundary for regex-based chunking (MULTILINE)")
    # semantic
    ap.add_argument("--sem_threshold", type=float, default=0.65, help="Similarity threshold for semantic chunking")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    pages = load_pdf_text(args.pdf)
    text = join_pages(pages)

    # fixed (with overlap)
    out_fixed = chunk_fixed_chars(text, size=args.max_chars, overlap=args.overlap)
    write_jsonl(os.path.join(args.outdir, "chunks_fixed.jsonl"), out_fixed)

    # sentence sliding (parameters exposed in help but using defaults here)
    out_sentence = chunk_sentences_sliding(text, window=args.sent_window, overlap=args.sent_overlap)
    write_jsonl(os.path.join(args.outdir, "chunks_sentence.jsonl"), out_sentence)

    # regex-based
    out_regex = chunk_regex(text, pattern=args.regex, max_chars=args.max_chars, overlap=args.overlap)
    write_jsonl(os.path.join(args.outdir, "chunks_regex.jsonl"), out_regex)

    # hierarchical
    out_hier = chunk_hierarchical(text, max_chars=args.max_chars, overlap=args.overlap)
    write_jsonl(os.path.join(args.outdir, "chunks_hierarchical.jsonl"), out_hier)

    # semantic
    out_sem = chunk_semantic(text, max_chars=args.max_chars, similarity_threshold=args.sem_threshold)
    write_jsonl(os.path.join(args.outdir, "chunks_semantic.jsonl"), out_sem)

    summary = {
        "fixed": len(out_fixed),
        "sentence": len(out_sentence),
        "regex": len(out_regex),
        "hierarchical": len(out_hier),
        "semantic": len(out_sem),
        "note_semantic_backend": "sentence-transformers" if _HAS_ST else ("tf-idf" if _HAS_SK else "fallback_sentence_window")
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
