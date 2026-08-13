import os
import glob
import fitz  # PyMuPDF
import unicodedata
import config
from graph_db import GraphDB
from ner import extract_triples, get_nlp
import re


def load_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    # 1. Normalize Unicode characters (canonical decomposition/recomposition)
    text = unicodedata.normalize("NFKC", text)

    # 2. Strip soft hyphens explicitly
    text = re.sub(r"\xad\s*", "", text)

    # Hard hyphen at a line-wrap: "hyphen-\nated" -> "hyphenated"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Remaining newlines are just line wraps, not paragraph breaks — join with a space
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text


def chunk_text(text: str, chunk_size: int, overlap_sentences: int = 1):
    sentences = [s.text.strip() for s in get_nlp()(text).sents if s.text.strip()]
    chunks, current, current_len = [], [], 0
    for sent in sentences:
        if current and current_len + len(sent) > chunk_size:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:]
            current_len = sum(len(s) for s in current)
        current.append(sent)
        current_len += len(sent)
    if current:
        chunks.append(" ".join(current))
    return chunks

# Define canonical mappings for domain-specific concepts
CANONICAL_ENTITIES = {
    "vr": "virtual reality",
}

def normalize(name: str) -> str:
    norm = " ".join(name.strip().lower().split())
    
    # 2. Map directly to canonical node if a variant exists
    if norm in CANONICAL_ENTITIES:
        return CANONICAL_ENTITIES[norm]

    return norm


def main():
    db = GraphDB()
    db.setup_indexes()

    pdf_files = sorted(glob.glob(os.path.join(config.PDF_DIR, "*.pdf")))
    if not pdf_files:
        print(f"No PDFs found in {config.PDF_DIR}/. Add PDFs there and re-run.")
        db.close()
        return

    total_triples = 0
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"\nProcessing {filename}...")
        text = load_pdf_text(pdf_path)
        chunks = chunk_text(text, config.CHUNK_SIZE)
        print(f"  {len(chunks)} chunks")

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{filename}_{idx}"
            triples = extract_triples(chunk)

            for t in triples:
                subject = normalize(t.get("subject", ""))
                relation = t.get("relation", "")
                obj = normalize(t.get("object", ""))
                if not subject or not relation or not obj:
                    continue
                db.write_triple(subject, relation, obj, chunk_id, chunk, filename)
                total_triples += 1

            if triples:
                print(f"  chunk {idx}: {len(triples)} triples extracted")

    print(f"\nDone. {total_triples} triples written to Neo4j.")
    print("Browse the graph at http://localhost:7474")
    db.close()


if __name__ == "__main__":
    main()
