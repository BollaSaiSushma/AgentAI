import os
import json
import uuid
from typing import List, Dict, Any, Iterable

import numpy as np
from tqdm import tqdm
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss


# -----------------------------
# Config
# -----------------------------
PDF_PATH = r"C:\Users\saisu\OneDrive\Desktop\AgentAI\AgentAI\Day-2\sample_rag_document.pdf"   # update if needed
INDEX_PATH = "faiss_index.bin"
META_PATH = "faiss_meta.json"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim


# -----------------------------
# PDF utils (robust with checks)
# -----------------------------
def assert_is_pdf(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {os.path.abspath(pdf_path)}")
    with open(pdf_path, "rb") as f:
        magic = f.read(5)
    if not magic.startswith(b"%PDF-"):
        raise ValueError(
            f"Not a PDF (header={magic!r}). Check PDF_PATH: {os.path.abspath(pdf_path)}"
        )

def load_pdf_text(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text per page; falls back to PyMuPDF if pypdf struggles."""
    assert_is_pdf(pdf_path)
    try:
        reader = PdfReader(pdf_path)
        pages = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            txt = " ".join(txt.split())
            pages.append({"page": i + 1, "text": txt})
        if not pages:
            raise ValueError("No extractable text via pypdf.")
        return pages
    except Exception as e:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            pages = []
            for i, pg in enumerate(doc, start=1):
                txt = pg.get_text() or ""
                txt = " ".join(txt.split())
                pages.append({"page": i, "text": txt})
            doc.close()
            if not pages:
                raise
            print("Note: used PyMuPDF fallback for extraction.")
            return pages
        except Exception:
            raise RuntimeError(
                f"Failed to read PDF. Original error: {e}. "
                "Try re-exporting the PDF or install PyMuPDF (pymupdf)."
            )


# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    out, n = [], len(text)
    step = max(1, chunk_size - overlap)
    i = 0
    while i < n:
        out.append(text[i:i+chunk_size])
        i += step
    return out

def chunk_pages(pages: List[Dict[str, Any]], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    chunks = []
    for p in pages:
        pieces = chunk_text(p["text"], chunk_size, overlap)
        for idx, ch in enumerate(pieces, start=1):
            chunks.append({
                "uid": str(uuid.uuid4()),
                "page": p["page"],
                "chunk_index": idx,
                "text": ch
            })
    return chunks


# -----------------------------
# Helpers
# -----------------------------
def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / norms).astype("float32")

def batched(indices: List[int], size: int) -> Iterable[List[int]]:
    for i in range(0, len(indices), size):
        yield indices[i:i+size]


# -----------------------------
# Main
# -----------------------------
def main():
    # 1) Read & chunk PDF
    pages = load_pdf_text(PDF_PATH)
    chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        raise RuntimeError("No text extracted from PDF.")
    print(f"Pages: {len(pages)} | Chunks: {len(chunks)}")

    # 2) Embeddings (normalized for cosine)
    model = SentenceTransformer(EMBED_MODEL)
    print("Computing embeddings...")
    texts = [c["text"] for c in chunks]
    vecs = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    vecs = vecs.astype("float32")  # ensure float32

    dim = vecs.shape[1]

    # 3) Build FAISS index (cosine via normalized vectors + inner product)
    base = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap2(base)

    # Prepare IDs (int64) aligned with chunks
    ids = np.arange(len(chunks), dtype="int64")
    index.add_with_ids(vecs, ids)

    # 4) Save index and metadata
    faiss.write_index(index, INDEX_PATH)
    meta = {
        "source": os.path.basename(PDF_PATH),
        "embedding_model": EMBED_MODEL,
        "dim": dim,
        # store metadata in the same order as FAISS ids (we used 0..N-1)
        "items": [
            {
                "id": int(ids[i]),
                "uid": chunks[i]["uid"],
                "page": chunks[i]["page"],
                "chunk_index": chunks[i]["chunk_index"],
                "text": chunks[i]["text"]
            }
            for i in range(len(chunks))
        ]
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    print("Done! ✅")
    print(f"Saved index → {INDEX_PATH}")
    print(f"Saved metadata → {META_PATH}")
    print(f"Vectors: {index.ntotal} | Dim: {dim}")


if __name__ == "__main__":
    main()
