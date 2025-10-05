import os
import uuid
from typing import List, Dict, Any, Iterable

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

import chromadb

# -----------------------------
# Config
# -----------------------------
PDF_PATH = r"C:\Users\saisu\OneDrive\Desktop\AgentAI\AgentAI\Day-2\sample_rag_document.pdf"      # update if needed
CHROMA_PATH = "./chroma_db"               # folder for persistent storage
COLLECTION_NAME = "rag_pdf_chunks"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim


# -----------------------------
# PDF utils (with validation + fallback)
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
    """Extract text per page; fall back to PyMuPDF if pypdf struggles."""
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
    vecs = vecs.astype("float32")  # Chroma expects list[float]; we'll convert per add()

    # 3) Persistent Chroma client & collection (cosine)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # Ensure cosine metric (HNSW space)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # 4) Add in batches (ids must be strings)
    source_name = os.path.basename(PDF_PATH)
    BATCH = 256
    all_ids = [str(i) for i in range(len(chunks))]
    all_metas = [
        {
            "uid": chunks[i]["uid"],
            "page": chunks[i]["page"],
            "chunk_index": chunks[i]["chunk_index"],
            "source": source_name
        }
        for i in range(len(chunks))
    ]

    print("Upserting into Chroma...")
    for idxs in tqdm(list(batched(list(range(len(chunks))), BATCH))):
        collection.add(
            ids=[all_ids[i] for i in idxs],
            documents=[texts[i] for i in idxs],
            embeddings=[vecs[i].tolist() for i in idxs],
            metadatas=[all_metas[i] for i in idxs],
        )

    print("Done! ✅")
    print(f"Chroma path: {os.path.abspath(CHROMA_PATH)}")
    print(f"Collection: {COLLECTION_NAME}  |  Count: {collection.count()}")


if __name__ == "__main__":
    main()
