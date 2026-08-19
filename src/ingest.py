import os
import glob
import fitz  # PyMuPDF
import unicodedata
import config
from graph_db import GraphDB
from ner import extract_triples, get_nlp
import re


def remove_inline_citations_safely(text: str) -> str:
    """
    Remove citation markers glued directly to a word (superscript-style,
    no space before) while leaving real numbers, decimals, and p-values
    (which are always preceded by a space or a digit) untouched.
 
    Handles: "neoplasia3" -> "neoplasia"
             "trauma,4"   -> "trauma,"      (keeps punctuation)
             "philosophers.6–8" -> "philosophers."   (ranges)
             "hemiplegia.9 10"  -> "hemiplegia."      (space-separated lists)
             "0.05" / "p<0.05"  -> untouched          (preceded by a digit)
             "16 patients"      -> untouched          (preceded by a space)
    """
    # [1], [1-3], [1, 2], [1, 3, 5]
    text = re.sub(
        r"\[\s*\d+(?:\s*[-–]\s*\d+)?"
        r"(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*\s*\]",
        "",
        text,
    )
 
    # Superscript-style citation clusters glued directly to the end of a word
    # (letter or '%'), with no space in between. The lookbehind requires the
    # character right before the optional punctuation to be a LETTER, not a
    # digit -- that's what protects "0.05" (preceded by '0') while still
    # catching "trauma,4" (preceded by 'a'). Because it never depends on
    # what follows, it also catches citations before lowercase continuation
    # words, which the capital-letter-lookahead version was missing.
    # Each digit group in the cluster also requires a negative lookahead for
    # a following letter -- otherwise a trailing marker like "Malhotra1" and
    # an unrelated leading marker like "1Centre" (affiliation superscript on
    # the next line) get merged into one cluster and swallow the space
    # between them.
    text = re.sub(
        r"(?<=[A-Za-z%\)])"        # glued to a word ending in a letter, % or )
        r"(?P<lead>[.,;]?)"        # optional punctuation -- kept in output
        r"(?P<nums>\d{1,3}(?![A-Za-z])"
        r"(?:(?:,\s*|\s+|–|-)\d{1,3}(?![A-Za-z]))*)",
        lambda m: m.group("lead"),
        text,
    )
 
    # Leading affiliation-style markers glued to the FRONT of a capitalized
    # word with a space before them, e.g. " 1Centre for..." -> " Centre for...".
    # Restricted to a preceding whitespace/start-of-string so it can't eat a
    # digit that's part of a real word.
    text = re.sub(r"(?<=\s)\d{1,2}(?=[A-Z][a-z])", "", text)
    text = re.sub(r"^\d{1,2}(?=[A-Z][a-z])", "", text)
 
    return text
 
 
def load_pdf_text(path: str) -> str:
    doc = fitz.open(path)
 
    junk_patterns = [
        re.compile(
            r"by guest|Downloaded from|practneurol|Protected by copyright",
            re.IGNORECASE,
        ),
        re.compile(
            r"first published as|http:\/\/pn\.bmj\.com|Pract Neurol",
            re.IGNORECASE,
        ),
        re.compile(r"^\s*To cite\s*:.*", re.IGNORECASE),
        re.compile(r"^\s*Correspondence to\b.*", re.IGNORECASE),
        re.compile(r"^\s*(Accepted|Published Online First)\b.*", re.IGNORECASE),
        re.compile(r"^\s*REVIEW\s*$|^\s*\d{1,4}\s*$", re.IGNORECASE),
        re.compile(
            r"^(Centre for Restorative Neuroscience|Department of Neurology|Division of Brain Sciences)",
            re.IGNORECASE,
        ),
        re.compile(r"Li\s+K.*Pract\s+Neurol", re.IGNORECASE),
        re.compile(r"doi:\s*10\.\d{4,9}/", re.IGNORECASE),
        # Split into smaller fragments too: this sidebar box gets broken into
        # separate blocks by the embedded QR-code image, so matching only the
        # full phrase let fragments like "free content" slip through.
        re.compile(r"Open\s+Access", re.IGNORECASE),
        re.compile(r"Scan\s+to\s+access\s+more", re.IGNORECASE),
        re.compile(r"^\s*free\s+content\s*$", re.IGNORECASE),
    ]
 
    figure_pattern = re.compile(
        r"^(Figure \d+|Table \d+|Box \d+|The patient tends to omit)",
        re.IGNORECASE,
    )
    box_bullet_pattern = re.compile(r"^[▸–]\s")
 
    body_blocks = []
    figure_blocks = []  # kept separate so they never interrupt a sentence
                         # that's split across a column/page boundary
                         # (a figure box sits at a fixed point on the page,
                         # not at the point a reader would encounter it in
                         # the text's actual reading order)
 
    for page_idx, page in enumerate(doc):
        rect = page.rect
        crop_box = fitz.Rect(
            rect.x0 + 42,
            rect.y0 + (35 if page_idx > 0 else 0),
            rect.x1 - 30,
            rect.y1 - 45,
        )
 
        raw_blocks = page.get_text("blocks", clip=crop_box)
        valid_page_blocks = []
 
        for b in raw_blocks:
            x0, y0, x1, y1, text, block_no, block_type = b
            if block_type != 0:
                continue
 
            cleaned_block_text = unicodedata.normalize("NFKC", text).strip()
            if any(p.search(cleaned_block_text) for p in junk_patterns):
                continue
 
            # Strip soft hyphens / zero-width artifacts before line processing
            cleaned_block_text = re.sub(
                r'[\u200b\u200c\u200d\u200e\u200f\uFEFF\u00ad\x07]', '', cleaned_block_text
            )
 
            # De-hyphenate line-wraps HERE, while the block still has its
            # original newlines -- doing this after lines are joined with
            # spaces (as in the previous version) meant the safe \n-anchored
            # pattern never fired, and only a much riskier space-anchored
            # fallback ran, which could wrongly merge things like
            # "pre- and post-test" into "preand post-test".
            cleaned_block_text = re.sub(
                r"(\w)-\n\s*(\w)", r"\1\2", cleaned_block_text
            )
 
            clean_lines = []
            for line in cleaned_block_text.splitlines():
                line_str = line.strip()
                if not line_str:
                    continue
                if any(p.search(line_str) for p in junk_patterns):
                    continue
                clean_lines.append(line_str)
 
            if not clean_lines:
                continue
 
            block_text = " ".join(clean_lines)
 
            valid_page_blocks.append((x0, y0, x1, y1, block_text))
 
        midpoint_x = rect.width / 2.0
        col1 = [b for b in valid_page_blocks if b[0] < midpoint_x]
        col2 = [b for b in valid_page_blocks if b[0] >= midpoint_x]
 
        col1.sort(key=lambda item: item[1])
        col2.sort(key=lambda item: item[1])
 
        for b in col1 + col2:
            block_text = b[4]
            if figure_pattern.match(block_text) or box_bullet_pattern.match(block_text):
                figure_blocks.append(block_text)
            else:
                body_blocks.append(block_text)
 
    full_text = "\n\n".join(body_blocks)
 
    ref_idx = full_text.find("REFERENCES")
    if ref_idx != -1:
        full_text = full_text[:ref_idx]
 
    # Any hyphen-space wraps left over from justified text (rare, once
    # de-hyphenation runs at the block stage above) -- kept narrow to actual
    # line-wrap remnants, not a blanket "letter-space-letter" merge.
    full_text = re.sub(r"(\w)-\n\s*(\w)", r"\1\2", full_text)
 
    full_text = remove_inline_citations_safely(full_text)
 
    stitched_paragraphs = []
    current_p = ""
    for block in full_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if not current_p:
            current_p = block
        else:
            ends_with_connector = bool(
                re.search(
                    r"\b(and|or|the|to|of|in|that|with|as|for|by|from|because|more|less)\s*$",
                    current_p,
                    re.IGNORECASE,
                )
            )
            is_mid_sentence = bool(
                re.search(r"[a-z0-9,]\s*$", current_p)
                and not re.search(r"[\.\?\!:]\s*$", current_p)
            )
            starts_with_lowercase = bool(re.match(r"^[a-z]", block))
 
            if ends_with_connector or is_mid_sentence or starts_with_lowercase:
                current_p += " " + block
            else:
                stitched_paragraphs.append(current_p)
                current_p = block
 
    if current_p:
        stitched_paragraphs.append(current_p)
 
    final_output = "\n\n".join(stitched_paragraphs)
    final_output = re.sub(r"[ \t]+", " ", final_output)
    final_output = re.sub(r"\n{3,}", "\n\n", final_output).strip()
 
    if figure_blocks:
        # Merge bullet-fragment blocks (e.g. Box 1's "▸ Core deficit" lines)
        # back onto the preceding Figure/Table/Box heading so each entry
        # reads as one unit instead of scattered one-line fragments.
        merged_figures = []
        for fb in figure_blocks:
            if box_bullet_pattern.match(fb) and merged_figures:
                merged_figures[-1] += " " + fb
            else:
                merged_figures.append(fb)
 
        figures_section = "\n\n".join(merged_figures)
        figures_section = re.sub(r"[ \t]+", " ", figures_section)
        # citation numbers appear in figure captions too (e.g. "test.27")
        figures_section = remove_inline_citations_safely(figures_section)
        final_output += "\n\n--- Figures, Tables, and Boxes ---\n\n" + figures_section
 
    with open("pdf_to_text.txt", "w", encoding="utf-8") as f:
        f.write(final_output)
 
    return final_output


"""def load_pdf_text(path: str) -> str:
    doc = fitz.open(path)
 
    page_texts = []
    for page in doc:
        page_h = page.rect.height
 
        # Header/footer band cutoffs, as a fraction of page height rather than
        # hardcoded points — generalizes across different page sizes.
        header_cutoff = page_h * 0.06   # top ~6% of page
        footer_cutoff = page_h * 0.955  # bottom ~4.5% of page
 
        kept_blocks = []
        for b in page.get_text("blocks"):
            x0, y0, x1, y1, block_text = b[0], b[1], b[2], b[3], b[4]
            block_text = block_text.strip()
            if not block_text:
                continue
            if y1 < header_cutoff:
                continue  # running header (e.g. "REVIEW")
            if y0 > footer_cutoff:
                continue  # running footer (page number, doi, journal name)
            # narrow tall block hugging the right margin = rotated watermark strip
            if x0 > page.rect.width * 0.93 and (x1 - x0) < 40:
                continue
            kept_blocks.append(block_text)
 
        page_texts.append("\n".join(kept_blocks))
 
    doc.close()
    text = "\n".join(page_texts)
 
    # Belt-and-braces: strip the recurring boilerplate phrase by content too,
    # in case a PDF's geometry doesn't match the band cutoffs above.
    text = re.sub(
        r"Protected by copyright,?\s*including for uses related to text and data mining,?\s*"
        r"AI training,?\s*and similar technologies\.?\s*",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\.?\s*by guest on [A-Za-z0-9,\s]+?http\S+\s*Downloaded from\s*",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\d{1,2}\s+\w+\s+\d{4}\.\s*10\.\d{4}/\S+\s*on\s*.+?first published as\s*",
        " ",
        text,
        flags=re.IGNORECASE,
    )
 
    # 1. Normalize Unicode characters (canonical decomposition/recomposition)
    text = unicodedata.normalize("NFKC", text)
 
    # 2. Strip soft hyphens explicitly
    text = re.sub(r"\xad\s*", "", text)
    # Strip hidden Unicode artifacts: Zero-Width Space, soft hyphens, etc.
    text = re.sub(r'[\u200b\u200c\u200d\u200e\u200f\uFEFF\u00ad\u00a0\x07]', ' ', text)
 
    # Hard hyphen at a line-wrap: "hyphen-\nated" -> "hyphenated"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
 
    # Remaining newlines are just line wraps, not paragraph breaks — join with a space
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
 
    # 3. Fix citation superscript numbers glued directly to adjacent words,
    # e.g. "neoplasia3 and" -> "neoplasia [3] and", "1Centre" -> "[1] Centre",
    # "Stroke2009" -> "Stroke 2009"
    text = re.sub(r'([a-zA-Z\.\,])(\d{1,3})(\s+[A-Za-z])', r'\1 [\2]\3', text)
    text = re.sub(r'(?<![\w])(\d{1,2})([A-Z][a-z]{2,})', r'[\1] \2', text)
    text = re.sub(r'([a-zA-Z])(\d{4};)', r'\1 \2', text)
 
    # 4. Strip the references section: cut everything from the "REFERENCES"
    # heading to the end of the document.
    text = re.split(r"\bREFERENCES\b\s*(?=\[?\d+\]?\s+[A-Z])", text, maxsplit=1)[0].strip()
 
    with open("pdf_to_text.txt", "w", encoding="utf-8") as f:
        f.write(text)
 
    return text"""


def chunk_text(text: str, chunk_size: int, overlap_sentences: int = 1):
    nlp = get_nlp()
    # Disable heavy NER during sentence segmentation for speed and memory safety
    disable_pipes = [p for p in ["ner", "entity_ruler"] if p in nlp.pipe_names]
    with nlp.select_pipes(disable=disable_pipes):
        doc = nlp(text)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

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
    "rct": "randomized controlled trial",
    "usn": "unilateral spatial neglect",
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
