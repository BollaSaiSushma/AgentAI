import os
import math
import uuid
from typing import List, Dict, Any, Iterable

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from pinecone import Pinecone, ServerlessSpec

# -----------------------------
# Config
# -----------------------------
PDF_PATH = r"C:\Users\saisu\OneDrive\Desktop\AgentAI\AgentAI\Day-2\sample_rag_document.pdf"   # path to your PDF (downloaded file)
INDEX_NAME = "rag-demo-index"
NAMESPACE = "pdf-chunks"               # optional logical partition in Pinecone

# Chunking params (token-free approximation with characters)
CHUNK_SIZE = 900        # ~200-300 tokens for typical English
CHUNK_OVERLAP = 150     # overlap to preserve context across chunks

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim

# -----------------------------
# Utilities
# -----------------------------
def load_pdf_text(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from each page of the PDF.
    Returns a list of dicts: [{"page": int, "text": str}, ...]
    """
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        # Normalize whitespace a bit
        txt = " ".join(txt.split())
        pages.append({"page": i + 1, "text": txt})
    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Simple character-based sliding window chunker with overlap.
    """
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    step = max(1, chunk_size - overlap)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        start += step
        if start >= n:
            break
    return chunks


def chunk_pages(pages: List[Dict[str, Any]],
                chunk_size: int,
                overlap: int) -> List[Dict[str, Any]]:
    """
    Turn pages into chunk records with metadata.
    """
    out = []
    for p in pages:
        pieces = chunk_text(p["text"], chunk_size, overlap)
        for i, ch in enumerate(pieces, start=1):
            out.append({
                "id": str(uuid.uuid4()),
                "page": p["page"],
                "chunk_index": i,
                "text": ch
            })
    return out


def batched(iterable: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    """
    Yield items from an iterable in lists of size batch_size.
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# -----------------------------
# Main
# -----------------------------
def main():
    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise RuntimeError("Missing PINECONE_API_KEY in environment or .env")

    # 1) Read PDF and make chunks
    pages = load_pdf_text(PDF_PATH)
    chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        raise RuntimeError("No text extracted—check your PDF path/content.")

    print(f"Pages: {len(pages)}  |  Chunks: {len(chunks)}")

    # 2) Embeddings
    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c["text"] for c in chunks]
    print("Computing embeddings...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    # 3) Connect to Pinecone and ensure index exists
    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=pinecone_api_key)

    # If index does not exist, create a serverless index with the model's dim
    dim = len(embeddings[0])
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"Creating index '{INDEX_NAME}' (dim={dim})...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready
        pc.describe_index(INDEX_NAME)  # raises if not found; serverless is typically quick

    index = pc.Index(INDEX_NAME)

    # 4) Upsert vectors in batches with metadata
    print("Upserting vectors...")
    batch_size = 100
    for batch_ids in tqdm(list(batched(range(len(chunks)), batch_size))):
        vectors = []
        for i in batch_ids:
            meta = {
                "text": chunks[i]["text"],
                "page": chunks[i]["page"],
                "chunk_index": chunks[i]["chunk_index"],
                "source": os.path.basename(PDF_PATH),
            }
            vectors.append({
                "id": chunks[i]["id"],
                "values": embeddings[i].tolist(),
                "metadata": meta
            })
        index.upsert(vectors=vectors, namespace=NAMESPACE)

    print("Done! ✅")
    print(f"Index: {INDEX_NAME} | Namespace: {NAMESPACE}")
    print("Example: try a similarity search next.")


if __name__ == "__main__":
    main()
