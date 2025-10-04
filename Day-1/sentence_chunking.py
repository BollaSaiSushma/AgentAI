# pip install PyPDF2 nltk  # (nltk optional; script falls back to regex if punkt isn't available)

import re
import csv
from pathlib import Path

# --- Config ---
PDF_PATH = r"C:\Users\saisu\OneDrive\Desktop\AgentAI\AgentAI\Day-1\sample_text_for_chunking.pdf"   # <-- change if your file is elsewhere
OUT_CSV  = "sentence_chunks.csv"

# --- PDF text extraction (PyPDF2) ---
def extract_pdf_text(pdf_path: str) -> list[tuple[int, str]]:
    """Return list of (page_number, page_text) 1-indexed."""
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        pages.append((i, txt))
    return pages

# --- Sentence splitter (nltk if available, else regex) ---
def split_sentences(text: str) -> list[str]:
    # Try nltk's Punkt (best quality). Fallback to regex if not present.
    try:
        import nltk
        try:
            _ = nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        from nltk.tokenize import sent_tokenize
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        # Simple regex fallback: split on ., !, ? followed by whitespace + capital/quote/bracket
        pattern = r"(?<=[\.!?])\s+(?=[A-Z\"'(\[])"
        # normalize newlines -> spaces so splitting works better
        clean = re.sub(r"\s+", " ", text).strip()
        parts = re.split(pattern, clean)
        # Ensure punctuation at end if lost (optional—kept simple)
        return [p.strip() for p in parts if p.strip()]

def main(pdf_path: str, out_csv: str) -> None:
    # 1) Read PDF pages
    assert Path(pdf_path).exists(), f"File not found: {pdf_path}"
    pages = extract_pdf_text(pdf_path)

    # 2) Chunk into sentences, keep page numbers
    rows = []
    chunk_id = 1
    for page_num, page_text in pages:
        if not page_text:
            continue
        # Replace hard linebreaks with spaces for cleaner splitting
        page_text = re.sub(r"\s*\n\s*", " ", page_text)
        sentences = split_sentences(page_text)
        for sent in sentences:
            rows.append({
                "chunk_id": chunk_id,
                "page": page_num,
                "sentence": sent
            })
            chunk_id += 1

    # 3) Save to CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["chunk_id", "page", "sentence"])
        writer.writeheader()
        writer.writerows(rows)

    # 4) Quick preview
    print(f"Saved {len(rows)} sentence chunks -> {out_csv}")
    for r in rows[:5]:
        print(f"[{r['chunk_id']:>3}] (p{r['page']}): {r['sentence'][:100]}{'...' if len(r['sentence'])>100 else ''}")

if __name__ == "__main__":
    main(PDF_PATH, OUT_CSV)
