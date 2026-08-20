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
    Systematically removes false spaces created by PDF font ligatures (fi, ff, fl, ffi, ffl)
    and kerning artifacts (Th).
    """
    th_words = r"(e|is|ese|ose|ere|us|ree|ird|rough|roughout|en|an)"
    text = re.sub(r"\bTh\s+" + th_words + r"\b", r"Th\1", text)
    text = re.sub(r"\bth\s+" + th_words + r"\b", r"th\1", text)

    # Universal fix for words ending in 'fl', 'ffi', 'ffl' followed by space + lowercase
    text = re.sub(
        r"\b([A-Za-z]*(?:fl|ffi|ffl))\s+([a-z]+)\b",
        r"\1\2",
        text,
    )

    # Universal fix for words ending in or starting with 'fi' followed by space + lowercase
    def fix_fi_match(m):
        prefix, suffix = m.group(1), m.group(2)
        if prefix.lower() in ("sci", "hi", "wi") and suffix.lower() in ("movie", "system", "sound", "set"):
            return m.group(0)
        return prefix + suffix

    text = re.sub(
        r"\b([A-Za-z]*fi)\s+([a-z]+)\b",
        fix_fi_match,
        text,
        flags=re.IGNORECASE,
    )

    # Universal fix for words with 'ff' followed by space + lowercase
    real_ff_words = {
        "off", "staff", "cliff", "stiff", "buff", "fluff", "cuff", "puff",
        "sheriff", "muff", "stuff", "scuff", "sniff", "tariff", "bailiff",
        "plaintiff", "bluff", "turnoff", "payoff", "kickoff", "cutoff"
    }

    def fix_ff_match(m):
        prefix, suffix = m.group(1), m.group(2)
        if prefix.lower() not in real_ff_words:
            return prefix + suffix
        if suffix.lower() in ("ect", "ects", "ective", "erence", "erences", "erent", "icult", "iculty", "icient", "iciency", "ice", "icer"):
            return prefix + suffix
        return m.group(0)

    text = re.sub(
        r"\b([A-Za-z]*ff)\s+([a-z]+)\b",
        fix_ff_match,
        text,
        flags=re.IGNORECASE,
    )

    # Handle 3-part split ligatures (e.g. "bene fi cial" -> "beneficial")
    text = re.sub(
        r"\b([a-z]{2,})\s+(fi|fl|ff)\s+([a-z]{2,})\b",
        r"\1\2\3",
        text,
        flags=re.IGNORECASE,
    )

    return text


def remove_inline_citations_safely(text: str) -> str:
    """
    Removes inline citation numbers (superscripts, bracketed [1-3], or numbers
    following period + Capital word or 'et al.') while strictly preserving real
    clinical metrics, counts (375, 26, 5, 7, 14), decimals (0.05, 64.7), years (2022),
    and measurements.
    """
    # 1. Bracketed citations: [1], [1-3], [1, 2], [1, 3, 5]
    text = re.sub(r"\[\s*\d+(?:\s*[-–,]\s*\d+)*\s*\]", "", text)

    # 2. Citations following 'et al.' (e.g., "Glover et al. 16,28 applied")
    text = re.sub(
        r"(?<=et al\.)\s*\d{1,3}(?:\s*[,–-]\s*\d{1,3})*(?=\s+[A-Za-z]|\s*[.,;]|$)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # 3. Citations following sentence-ending period before a CAPITALIZED word
    # e.g., "space. 17 Castiello", "hygiene. 4,5 Previous", "ganglia. 1-3 Patients"
    # Followed by space + Capitalized word (A-Z) so numbers before lowercase words ("Class I, 7 as") are protected.
    text = re.sub(
        r"(?<=[a-zA-Z]\.)\s*\d{1,3}(?:\s*[,–-]\s*\d{1,3})*(?=\s+[A-Z][a-z])",
        "",
        text,
    )

    # 4. Citations glued directly to the end of a word (e.g. "hemiplegia9", "paretic side11")
    glued_citation = re.compile(
        r"(?P<prefix>[A-Za-z%\)])"
        r"(?P<lead>[.,;]?)"
        r"\d{1,3}(?:\s*[,–-]\s*\d{1,3})*"
        r"(?!\s*(?:mg|g|kg|ml|cm|mm|m|hz|min|mins|sec|secs|hours|hrs|days|weeks|months|years|patients|subjects|controls|%|version|points|score|scores|group|table|figure|box)\b)"
    )
    text = glued_citation.sub(r"\g<prefix>\g<lead>", text)

    # 5. Collapse double spaces left behind
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text

def load_pdf_text(path: str) -> str:
    doc = fitz.open(path)

    junk_patterns = [
        re.compile(r"by guest|Downloaded from|practneurol|Protected by copyright", re.IGNORECASE),
        re.compile(r"first published as|http:\/\/pn\.bmj\.com|Pract Neurol", re.IGNORECASE),
        re.compile(r"^\s*To cite\s*:.*", re.IGNORECASE),
        re.compile(r"^\s*Correspondence to\b.*", re.IGNORECASE),
        re.compile(r"^\s*(Accepted|Published Online First)\b.*", re.IGNORECASE),
        re.compile(r"^\s*REVIEW\s*$|^\s*\d{1,4}\s*$", re.IGNORECASE),
        re.compile(r"^(Centre for Restorative Neuroscience|Department of Neurology|Division of Brain Sciences)", re.IGNORECASE),
        re.compile(r"Li\s+K.*Pract\s+Neurol", re.IGNORECASE),
        re.compile(r"doi:\s*10\.\d{4,9}/", re.IGNORECASE),
        re.compile(r"Open\s+Access", re.IGNORECASE),
        re.compile(r"Scan\s+to\s+access\s+more", re.IGNORECASE),
        re.compile(r"^\s*free\s+content\s*$", re.IGNORECASE),
    ]

    figure_pattern = re.compile(r"^(Figure \d+|Table \d+|Box \d+|The patient tends to omit)", re.IGNORECASE)
    box_bullet_pattern = re.compile(r"^[▸–]\s")

    body_blocks = []
    figure_blocks = []

    for page_idx, page in enumerate(doc):
        rect = page.rect
        crop_box = fitz.Rect(
            rect.x0 + 10,
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

            cleaned_block_text = re.sub(r'[\u200b\u200c\u200d\u200e\u200f\uFEFF\u00ad\x07]', '', cleaned_block_text)
            cleaned_block_text = re.sub(r"(\w)-\n\s*(\w)", r"\1\2", cleaned_block_text)

            clean_lines = []
            for line in cleaned_block_text.splitlines():
                line_str = line.strip()
                if not line_str or any(p.search(line_str) for p in junk_patterns):
                    continue
                if re.match(r"^(REFERENCES|References)\b", line_str):
                    line_str = "\n\n__REF_START__\n\n" + line_str
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

    ref_match = re.search(r"__REF_START__", full_text)
    if ref_match and ref_match.start() > len(full_text) * 0.5:
        full_text = full_text[:ref_match.start()]

    full_text = re.sub(r"(\w)-\n\s*(\w)", r"\1\2", full_text)
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
            ends_with_connector = bool(re.search(r"\b(and|or|the|to|of|in|that|with|as|for|by|from|because|more|less)\s*$", current_p, re.IGNORECASE))
            is_mid_sentence = bool(re.search(r"[a-z0-9,]\s*$", current_p) and not re.search(r"[\.\?\!:]\s*$", current_p))
            starts_with_lowercase = bool(re.match(r"^[a-z]", block))

            if ends_with_connector or is_mid_sentence or starts_with_lowercase:
                current_p += " " + block
            else:
                stitched_paragraphs.append(current_p)
                current_p = block

    if current_p:
        stitched_paragraphs.append(current_p)

    if figure_blocks:
        merged_figures = []
        for fb in figure_blocks:
            if box_bullet_pattern.match(fb) and merged_figures:
                merged_figures[-1] += " " + fb
            else:
                merged_figures.append(fb)

        merged_figures = [fix_pdf_ligatures(fb) for fb in merged_figures]
        merged_figures = [remove_inline_citations_safely(fb) for fb in merged_figures]

        reference_key_pattern = re.compile(r"^(Figure|Table|Box)\s+(\d+)", re.IGNORECASE)
        figure_lookup = {}
        for fb in merged_figures:
            m = reference_key_pattern.match(fb)
            if m:
                key = (m.group(1).lower(), m.group(2))
                figure_lookup[key] = fb
            else:
                figure_lookup[("_unmatched", str(len(figure_lookup)))] = fb

        mention_pattern = re.compile(r"\b(figure|table|box)\s*(\d+)\b", re.IGNORECASE)
        inserted_keys = set()
        final_paragraphs = []

        for para in stitched_paragraphs:
            final_paragraphs.append(para)
            for m in mention_pattern.finditer(para):
                key = (m.group(1).lower(), m.group(2))
                if key in figure_lookup and key not in inserted_keys:
                    final_paragraphs.append(figure_lookup[key])
                    inserted_keys.add(key)

        leftover = [fb for key, fb in figure_lookup.items() if key not in inserted_keys]
        final_paragraphs.extend(leftover)
        stitched_paragraphs = final_paragraphs

    final_output = "\n\n".join(stitched_paragraphs)
    final_output = re.sub(r"[ \t]+", " ", final_output)
    final_output = re.sub(r"\n{3,}", "\n\n", final_output).strip()

    with open("pdf_to_text.txt", "w", encoding="utf-8") as f:
        f.write(final_output)

    return final_output

def chunk_text(text: str, chunk_size: int, overlap_sentences: int = 1):
    nlp = get_nlp()
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

CANONICAL_ENTITIES = {
    "vr": "virtual reality",
    "rct": "randomized controlled trial",
    "usn": "unilateral spatial neglect",
}

def normalize(name: str) -> str:
    norm = " ".join(name.strip().lower().split())
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