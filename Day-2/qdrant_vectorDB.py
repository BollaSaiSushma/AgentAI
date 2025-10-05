import os
import uuid
from typing import List, Dict, Any, Iterable

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# -----------------------------
# Config
# -----------------------------
PDF_PATH = r"C:\Users\saisu\OneDrive\Desktop\AgentAI\AgentAI\Day-2\sample_rag_document.pdf"        # path to your PDF
COLLECTION_NAME = "rag_demo_pdf_chunks"     # Qdrant collection
CHUNK_SIZE = 900                             # ~200–300 tokens
CHUNK_OVERLAP = 150                          # preserve context across chunks
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim

# -----------------------------
# Utilities
# -----------------------------
def load_pdf_text(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from each page of the PDF.
    Returns a list: [{"page": int, "text": str}, ...]
    """
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        txt = " ".join(txt.split())
        pages.append({"page": i + 1, "text": txt})
    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Character-based sliding window with overlap.
    """
    if not text:
        return []
    out = []
    start = 0
    n = len(text)
    step = max(1, chunk_size - overlap)
    while start < n:
        end = min(n, start + chunk_size)
        out.append(text[start:end])
        start += step
        if start >= n:
            break
    return out


def chunk_pages(pages: List[Dict[str, Any]],
                chunk_size: int,
                overlap: int) -> List[Dict[str, Any]]:
    """
    Produce chunk records with useful metadata.
    """
    chunks = []
    for p in pages:
        pieces = chunk_text(p["text"], chunk_size, overlap)
        for i, ch in enumerate(pieces, start=1):
            chunks.append({
                "id": str(uuid.uuid4()),
                "page": p["page"],
                "chunk_index": i,
                "text": ch
            })
    return chunks


def batched(items: List[Any], size: int):
    for i in range(0, len(items), size):
        yield items[i:i+size]


# -----------------------------
# Main
# -----------------------------
def main():
    load_dotenv()
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")  # optional for local

    # 1) Read PDF & make chunks
    pages = load_pdf_text(PDF_PATH)
    chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        raise RuntimeError("No text extracted; check your PDF path/content.")
    print(f"Pages: {len(pages)} | Chunks: {len(chunks)}")

    # 2) Embeddings
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    print("Computing embeddings...")
    vectors = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    dim = vectors.shape[1]

    # 3) Connect to Qdrant
    print(f"Connecting to Qdrant at {qdrant_url} ...")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # 4) Ensure collection exists
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        print(f"Creating collection '{COLLECTION_NAME}' (dim={dim}) ...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )

    # 5) Upsert points in batches
    print("Upserting vectors into Qdrant...")
    for batch_idx in tqdm(list(batched(list(range(len(chunks))), 128))):
        points = []
        for i in batch_idx:
            payload = {
                "text": chunks[i]["text"],
                "page": chunks[i]["page"],
                "chunk_index": chunks[i]["chunk_index"],
                "source": os.path.basename(PDF_PATH),
            }
            points.append(
                qm.PointStruct(
                    id=chunks[i]["id"],
                    vector=vectors[i].tolist(),
                    payload=payload
                )
            )
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    # 6) Quick sanity: collection info
    info = client.get_collection(COLLECTION_NAME)
    print("Done! ✅")
    print("Status:", info.status)
    print("Vectors count (approx):", client.count(COLLECTION_NAME, exact=False).count)
    print(f"Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
