import os
import uuid
from typing import List, Dict, Any, Iterable

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)

# -----------------------------
# Config
# -----------------------------
PDF_PATH = r"C:\Users\saisu\OneDrive\Desktop\AgentAI\AgentAI\Day-2\sample_rag_document.pdf"       # put the file next to this script or update the path
COLLECTION_NAME = "rag_pdf_chunks"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim embeddings

# -----------------------------
# Helpers
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
    """
    Extracts text per page. Falls back to PyMuPDF if pypdf fails.
    Returns: [{"page": int, "text": str}, ...]
    """
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

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    chunks, n = [], len(text)
    step = max(1, chunk_size - overlap)
    i = 0
    while i < n:
        chunks.append(text[i:i+chunk_size])
        i += step
    return chunks

def chunk_pages(pages: List[Dict[str, Any]], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    out = []
    for p in pages:
        pieces = chunk_text(p["text"], chunk_size, overlap)
        for idx, ch in enumerate(pieces, start=1):
            out.append({
                "id": str(uuid.uuid4()),
                "page": p["page"],
                "chunk_index": idx,
                "text": ch
            })
    return out

def batched(iterable: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]

# -----------------------------
# Milvus schema / connect
# -----------------------------
def connect_milvus():
    uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    token = os.getenv("MILVUS_TOKEN") or None
    username = os.getenv("MILVUS_USERNAME") or None
    password = os.getenv("MILVUS_PASSWORD") or None

    # For Zilliz Cloud you typically pass either token OR username/password
    connections.connect(
        alias="default",
        uri=uri,
        token=token,
        user=username,
        password=password,
    )
    return uri

def ensure_collection(name: str, dim: int) -> Collection:
    if utility.has_collection(name):
        return Collection(name)

    # Define fields (VARCHAR requires max_length)
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False),
        FieldSchema(name="page", dtype=DataType.INT64),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields=fields, description="RAG PDF chunks")

    coll = Collection(name=name, schema=schema)

    # Create a vector index (HNSW; COSINE because we normalize embeddings)
    coll.create_index(
        field_name="embedding",
        index_params={
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 8, "efConstruction": 64}
        }
    )
    # Load into memory for search/insert speed
    coll.load()
    return coll

# -----------------------------
# Main
# -----------------------------
def main():
    load_dotenv()

    # 1) PDF → chunks
    pages = load_pdf_text(PDF_PATH)
    chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        raise RuntimeError("No text extracted from PDF.")
    print(f"Pages: {len(pages)} | Chunks: {len(chunks)}")

    # 2) Embeddings
    model = SentenceTransformer(EMBED_MODEL)
    print("Computing embeddings...")
    vecs = model.encode([c["text"] for c in chunks], batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    dim = vecs.shape[1]

    # 3) Connect to Milvus & ensure collection
    uri = connect_milvus()
    print(f"Connected to Milvus at {uri}")
    coll = ensure_collection(COLLECTION_NAME, dim)
    print(f"Collection ready: {COLLECTION_NAME}")

    # Optional: clean out older docs for the same source to avoid duplicates
    # from pymilvus import MutationResult
    # coll.delete(expr='source == "sample_rag_document.pdf"')

    # 4) Insert in batches
    source_name = os.path.basename(PDF_PATH)
    BATCH = 256
    total = 0
    for idxs in tqdm(list(batched(list(range(len(chunks))), BATCH))):
        # Milvus insert expects column-major data in field order
        ids = [chunks[i]["id"] for i in idxs]
        pages_c = [int(chunks[i]["page"]) for i in idxs]
        chunk_idx = [int(chunks[i]["chunk_index"]) for i in idxs]
        sources = [source_name for _ in idxs]
        texts = [chunks[i]["text"] for i in idxs]
        vectors = [vecs[i].tolist() for i in idxs]

        coll.insert([ids, pages_c, chunk_idx, sources, texts, vectors])
        total += len(idxs)

    # Ensure data is persisted and indexed
    coll.flush()
    print(f"Done! ✅ Inserted {total} vectors into '{COLLECTION_NAME}'")
    print("num_entities:", coll.num_entities)

    # Load for immediate querying (safe if already loaded)
    coll.load()

    # 5) Quick demo search
    query = "What is retrieval-augmented generation and why use overlapping chunks?"
    qvec = model.encode([query], normalize_embeddings=True)[0].tolist()
    res = coll.search(
        data=[qvec],
        anns_field="embedding",
        param={"ef": 64},          # HNSW runtime param
        limit=5,
        metric_type="COSINE",
        output_fields=["page", "chunk_index", "source", "text"],
        expr=f'source == "{source_name}"'  # optional filter
    )
    print("\nTop matches:")
    for hit in res[0]:
        meta = hit.entity
        print(f"- score={hit.distance:.4f} | p.{meta.get('page')} chunk #{meta.get('chunk_index')}")
        print((meta.get('text') or "")[:220], "...\n")


if __name__ == "__main__":
    main()
