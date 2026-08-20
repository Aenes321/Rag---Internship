import os
import glob
import fitz  # PyMuPDF
import unicodedata
import config
from graph_db import GraphDB
from ner import extract_triples, get_nlp
import re


def fix_pdf_ligatures(text: str) -> str:
    """
    Fixes words split by PDF font ligature/spacing artifacts (e.g. 'signifi cant',
    'diff erences', 'modifi ed', 'Th e', 'eff ect').
    """
    # 1. Fix "Th " kerning splits at word start (e.g., "Th e" -> "The", "Th is" -> "This", "Th ese" -> "These")
    text = re.sub(
        r"\b(Th|th)\s+(e|is|ese|ose|ere|us|ree|ird|rough|roughout)\b",
        r"\1\2",
        text,
    )

    # 2. Fix "ff" / "ffi" / "ffe" splits where prefix is a known non-word artifact
    # (e.g., "eff ect" -> "effect", "aff ect" -> "affect", "diff erence" -> "difference")
    text = re.sub(
        r"\b(eff|aff|diff|suff|offi)\s+([a-z]+)\b",
        r"\1\2",
        text,
    )

    # 3. Fix words split around "fi" or "fl" ligatures
    # (e.g., "signifi cant" -> "significant", "modifi ed" -> "modified", "speci fic" -> "specific")
    split_prefix_pattern = re.compile(
        r"\b("
        r"signifi|modifi|speci|classi|identi|bene|satis|quali|quanti|veri|clari|simpli|"
        r"ampli|recti|notifi|signi|personi|intensi|digni|heavi|easi|beauti|glori|terri|horri"
        r")\s+([a-z]+)\b",
        re.IGNORECASE,
    )
    text = split_prefix_pattern.sub(r"\1\2", text)

    # 4. Fix 3-part split ligatures (e.g., "signi fi cant" -> "significant", "bene fi cial" -> "beneficial")
    three_part_pattern = re.compile(
        r"\b(signi|bene|de|in|con|per|satis|identi|modi|speci|classi|clari|quali|quanti)\s+(fi|fl|ff)\s+([a-z]+)\b",
        re.IGNORECASE,
    )
    text = three_part_pattern.sub(r"\1\2\3", text)

    # 5. Targeted replacements for other common PDF split words
    common_splits = [
        (r"\bde\s+ficit(s?)\b", r"deficit\1"),
        (r"\bin\s+fluence(s?)\b", r"influence\1"),
        (r"\bin\s+flammation\b", r"inflammation"),
        (r"\bper\s+form(ed|ance|s|ing)?\b", r"perform\1"),
        (r"\bcon\s+fidence\b", r"confidence"),
        (r"\bcon\s+firm(ed|ation|s)?\b", r"confirm\1"),
        (r"\bman\s+ifest(ed|ation|s)?\b", r"manifest\1"),
        (r"\bsigni\s+ficant(ly)?\b", r"significant\1"),
        (r"\bdi\s+fficult(y|ies)?\b", r"difficult\1"),
        (r"\bpre\s+ference(s?)\b", r"preference\1"),
        (r"\bre\s+ference(s?)\b", r"reference\1"),
        (r"\bspeci\s+fic(ally|ation)?\b", r"specific\1"),
        (r"\baf\s+fected\b", r"affected"),
        (r"\bef\s+fect(ive|s)?\b", r"effect\1"),
    ]
    for pattern, replacement in common_splits:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def remove_inline_citations_safely(text: str) -> str:
    """
    Remove citation markers glued directly to a word (superscript-style,
    no space before) while leaving real numbers, decimals, and p-values untouched.
    """
    # [1], [1-3], [1, 2], [1, 3, 5]
    text = re.sub(
        r"\[\s*\d+(?:\s*[-–]\s*\d+)?"
        r"(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*\s*\]",
        "",
        text,
    )

    # Superscript-style citation clusters glued directly to the end of a word
    text = re.sub(
        r"(?<=[A-Za-z%\)])"        # glued to a word ending in a letter, % or )
        r"(?P<lead>[.,;]?)"        # optional punctuation -- kept in output
        r"(?P<nums>\d{1,3}(?![A-Za-z])"
        r"(?:(?:,\s*|\s+|–|-)\d{1,3}(?![A-Za-z]))*)",
        lambda m: m.group("lead"),
        text,
    )

    # Leading affiliation-style markers
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
    figure_blocks = []

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

            # De-hyphenate line-wraps
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

    full_text = re.sub(r"(\w)-\n\s*(\w)", r"\1\2", full_text)

    # Recombine words split by PDF font ligature/kerning artifacts
    full_text = fix_pdf_ligatures(full_text)

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
        merged_figures = []
        for fb in figure_blocks:
            if box_bullet_pattern.match(fb) and merged_figures:
                merged_figures[-1] += " " + fb
            else:
                merged_figures.append(fb)

        figures_section = "\n\n".join(merged_figures)
        figures_section = re.sub(r"[ \t]+", " ", figures_section)
        
        # Apply ligature fix to figure section as well
        figures_section = fix_pdf_ligatures(figures_section)
        figures_section = remove_inline_citations_safely(figures_section)
        
        final_output += "\n\n--- Figures, Tables, and Boxes ---\n\n" + figures_section

    with open("pdf_to_text.txt", "w", encoding="utf-8") as f:
        f.write(final_output)

    return final_output

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
