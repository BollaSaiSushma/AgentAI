import os
import uuid
from typing import List, Dict, Any
import certifi
from pymongo import MongoClient
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from tqdm import tqdm


# -----------------------------
# Config
# -----------------------------
PDF_PATH = r"C:\Users\saisu\OneDrive\Desktop\AgentAI\AgentAI\Day-2\sample_rag_document.pdf"        # place this file next to the script
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim


def load_pdf_text(pdf_path: str) -> List[Dict[str, Any]]:
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        txt = " ".join(txt.split())
        pages.append({"page": i + 1, "text": txt})
    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
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


def chunk_pages(pages: List[Dict[str, Any]], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    chunks = []
    for p in pages:
        pieces = chunk_text(p["text"], chunk_size, overlap)
        for i, ch in enumerate(pieces, start=1):
            chunks.append({
                "_id": str(uuid.uuid4()),
                "page": p["page"],
                "chunk_index": i,
                "text": ch
            })
    return chunks


def ensure_vector_search_index(db, collection_name: str, index_name: str, dims: int):
    """
    Create an Atlas Search vector index if it doesn't exist.
    Uses the 'createSearchIndexes' command (Atlas feature).
    """
    try:
        # If the index already exists, this command will error; we catch and skip.
        db.command({
            "createSearchIndexes": collection_name,
            "indexes": [{
                "name": index_name,
                "definition": {
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            "embedding": {
                                "type": "knnVector",
                                "dimensions": dims,
                                "similarity": "cosine"
                            },
                            "page": {"type": "number"},
                            "chunk_index": {"type": "number"},
                            "source": {"type": "string"},
                            "text": {"type": "string"}
                        }
                    }
                }
            }]
        })
        print(f"Created Search index '{index_name}'.")
    except OperationFailure as e:
        # If index exists or permissions limited, print and continue
        msg = str(e)
        if "already exists" in msg or "Index already exists" in msg:
            print(f"Search index '{index_name}' already exists. Continuing.")
        else:
            print(f"Note: could not create search index via API: {e}")
            print("If needed, create it once from Atlas UI → Collections → Search Indexes.")


def main():
    load_dotenv()

    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB", "rag_demo")
    coll_name = os.getenv("MONGODB_COLLECTION", "pdf_chunks")
    index_name = os.getenv("SEARCH_INDEX_NAME", "vector_index")

    if not uri:
        raise RuntimeError("Missing MONGODB_URI in .env")

    # 1) PDF → chunks
    pages = load_pdf_text(PDF_PATH)
    chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        raise RuntimeError("No text extracted; check PDF_PATH/contents.")
    print(f"Pages: {len(pages)} | Chunks: {len(chunks)}")

    # 2) Embeddings
    model = SentenceTransformer(EMBED_MODEL)
    print("Computing embeddings...")
    vecs = model.encode([c["text"] for c in chunks], batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    dims = vecs.shape[1]

    # 3) MongoDB connect
    client = MongoClient(
    uri,
    serverSelectionTimeoutMS=10000,
    tls=True,
    tlsCAFile=certifi.where(),
    # DO NOT pass tlsAllowInvalidCertificates at all
    tlsDisableOCSPEndpointCheck=True,
    )
    client.admin.command("ping")
    # 4) Ensure Search (vector) index exists
    ensure_vector_search_index(db, coll_name, index_name, dims)

    # 5) Insert (or replace) documents
    # Each document: { _id, text, page, chunk_index, source, embedding: [float,...] }
    print("Upserting documents...")
    docs = []
    for i, c in enumerate(chunks):
        docs.append({
            "_id": c["_id"],
            "text": c["text"],
            "page": c["page"],
            "chunk_index": c["chunk_index"],
            "source": os.path.basename(PDF_PATH),
            "embedding": vecs[i].tolist(),
        })

    # Quick approach: replace in small bulks
    from pymongo import ReplaceOne
    ops = [ReplaceOne({"_id": d["_id"]}, d, upsert=True) for d in docs]
    # Use reasonable batch sizing if very large
    BATCH = 500
    for i in tqdm(range(0, len(ops), BATCH)):
        coll.bulk_write(ops[i:i+BATCH], ordered=False)

    print("Done! ✅")
    print(f"DB: {db_name}  Collection: {coll_name}  Index: {index_name}")
    print("Tip: give Atlas a minute to finish indexing before searching.")


if __name__ == "__main__":
    main()
