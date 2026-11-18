import re
from itertools import combinations
import spacy
import config
import unicodedata

_nlp = None

PRONOUNS = {
    "it", "its", "they", "them", "their", "theirs", "he", "him", "his", 
    "she", "her", "hers", "we", "us", "our", "ours", "you", "your", "yours", 
    "i", "me", "my", "mine", "this", "that", "these", "those", "which", 
    "who", "whom", "whose", "what", "whatever", "whichever", "whoever", 
    "myself", "yourself", "himself", "herself", "itself", "ourselves", "themselves"
}

LEADING_STRIP_WORDS = PRONOUNS | {"the", "a", "an", "some", "any", "such", "no","is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "how", "why", "when", "where"}

# Regex patterns for citations (e.g., "Choi et al.", "Choi et al. (2021)", "Choi et al., 2021")
CITATION_PATTERN = re.compile(
    r'\b[A-Z][a-zA-Z\-]+(?:\s+et\s+al\.?|\s+and\s+[A-Z][a-zA-Z\-]+)?(?:\s*,?\s*\(\d{4}\)|\s*,?\s*\d{4})?(?=\s|[.,;:!\?\)]|$)'
)

DIGIT_PATTERN = re.compile(r'\b\d{3,}\b')

# 1. Regex to strip PDF layout noise, page/box labels, and special symbols
PDF_ARTIFACT_PATTERN = re.compile(r"[▸•*■▪│\|–—\t]|\b(box|page|fig|figure|table)\s*\d+\b", re.IGNORECASE)

# Capitalized words that often precede a year but are not author names
# (e.g. "In 2016", "Since 2015", "Between 2014 and 2015").
NON_AUTHOR_WORDS = {
    "in", "since", "by", "from", "during", "after", "before", "until", "between",
    "of", "the", "a", "an", "at", "on", "for", "to", "as", "with", "through",
    "figure", "table", "box", "fig", "year", "years", "and", "or", "but", "this",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "spring", "summer", "autumn", "winter",
}

def extract_citations(text: str) -> list[str]:
    """Find author/year citation strings in text."""
    clean_citations = []
    for match in CITATION_PATTERN.finditer(text):
        raw = match.group(0)
        norm = _normalize(re.sub(r'[\(\)]', '', raw))
        if not norm:
            continue
        if "et al" in norm:
            clean_citations.append(norm)
            continue
        # Year-based citations must be formatted like a citation:
        # "Smith (2020)", "Smith, 2020" or "Smith and Jones 2020".
        has_year = any(char.isdigit() for char in norm)
        looks_like_citation = "(" in raw or "," in raw or " and " in raw
        if has_year and looks_like_citation and norm.split()[0] not in NON_AUTHOR_WORDS:
            clean_citations.append(norm)
    return clean_citations

CANONICAL_TOKENS = {
    "vr": "virtual reality",
    "ivr": "immersive virtual reality",
    "nivr": "non-immersive virtual reality",
    "sivr": "semi-immersive virtual reality",
    "vpa": "virtual prism adaptation",
    "vpat": "virtual prism adaptation therapy",
    "pa": "prism adaptation",
    "usn": "unilateral spatial neglect",
    "sn": "spatial neglect",
    "et": "eye tracking",
    "cb": "computer-based",
    "pnp": "paper-and-pencil",
    "pca": "posterior cortical atrophy",
    "fnirs": "functional near-infrared spectroscopy",
    "slf": "superior longitudinal fasciculus",
}

def normalize_entity(name: str) -> str:
    """Lowercase, strip parenthetical duplicates, collapse whitespace, and expand acronyms."""
    # Remove parenthetical acronyms attached to a phrase, e.g. "virtual reality (vr)" -> "virtual reality"
    cleaned = re.sub(r"\s*\([a-zA-Z0-9\-/]+\)\s*", " ", name)
    cleaned = re.sub(r"[()\[\]]", " ", cleaned)
    tokens = _normalize(cleaned).split()
    expanded = " ".join(CANONICAL_TOKENS.get(t, t) for t in tokens)
    # Deduplicate consecutive repeated phrases if acronym expanded next to full phrase
    return " ".join(dict.fromkeys(expanded.split())) if expanded == " ".join(expanded.split()[:len(set(expanded.split()))]) else expanded

PATTERNS = [{"label": "CONDITION", "pattern": [{"LOWER": "spatial"}, {"LOWER": "neglect"}]}]

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(config.SPACY_MODEL)
        if "entity_ruler" not in _nlp.pipe_names:
            ruler = _nlp.add_pipe(
                "entity_ruler",
                after="ner",
                config={"overwrite_ents": True},
            )
            ruler.add_patterns(PATTERNS)
    return _nlp


# Order matters: first matching keyword set wins. Keep specific relations
# before the generic fallback. Keywords must describe the *relation* (verb
# phrases), not entities: bare nouns such as "parietal", "hemisphere" or
# "egocentric" appear in almost every neglect sentence and previously caused
# nearly all triples to be labelled AFFECTS_REGION / SUBTYPE_OF.
RELATION_KEYWORDS = [
    # Captures non-significant / comparable outcomes
    ("NO_DIFFERENCE_IN", ["no significant difference", "did not differ", "no difference", 
        "comparable to", "similar improvement", "no statistical difference","remained similar", "did not show significant","no change in", 
        "statistically insignificant", "equivalent to"]),
    
    # Captures significant improvements/outcomes (direction-less phrases such as
    # "statistically significant" / "showed significant" were removed because
    # they also match worsening results)
    ("SHOWED_IMPROVEMENT_IN", ["significantly higher", "significant improvement", "greater improvement",
        "superior to", "outperformed", "showed higher", "showed greater", "improved more", "significantly improved"]),

    # Core Clinical Relations
    ("TREATS", ["treatment for", "treated with", "therapy for", "rehabilitation of", "alleviate", "relieve"]),
    ("PREVENTS", ["prevent", "reduces the risk", "reduce the risk", "protect against","prevented", "prevention of", "protects against"]),
    ("CAUSES", ["cause", "causes", "leads to", "lead to", "results in", "result in", "triggers", "secondary to","induced by", "caused by",
        "consequence of", "contributes to", "contributing to", "contributed to"]),
    ("RISK_FACTOR_FOR", ["risk factor", "increases the risk", "increase the risk", "predispose to", "predisposing to", "predisposed to", "predisposition to"]),
    ("COMPLICATION_OF", ["complication of", "sequela of","resulting from"]),
    ("ASSESSED_BY", ["evaluated by", "measured by", "assessed using", "assessed by"]),
    ("DIAGNOSED_BY", ["diagnosed by", "diagnosed with", "screened for", "screen for", "detected by", "confirmed by", "identified by"]),
    ("HAS_SYMPTOM", ["characterized by", "manifested by", "symptom of", "presenting with", "presents with", "deficit in", "impairment in", "inattention to"]),

    # Spatial Neglect Specific Relations
    ("AFFECTS_REGION", ["lesion in", "lesions in", "lesioned in", "damage to", "pathology in", "structural damage to",
        "localized to", "atrophy of", "atrophy in", "degeneration of"]),
    ("SUBTYPE_OF", ["subtype of", "classified as", "type of neglect", "form of neglect", "variant of"]),

    # Behavioral/task performance keywords
    ("MANIFESTS_AS", ["recite", "recites", "presented with", "demonstrate", "displays"]),
    ("INTERACTS_WITH", ["interact", "interaction", "contraindicat", "contraindicated", "synergistic", "synergy", "combined with"]),
    # Generic fallback relation
    ("ASSOCIATED_WITH", ["associated with", "related to", "co-occur", "co-occurred", "co-occurring", "correlated with", "linked to", "coupled with", "co-exist", "co-existed", "co-existing", "co-occurring", "co-occurred with", "co-occurring with", "co-exist with", "co-existed with", "co-existing with"]),
]

COMPILED_RELATION_KEYWORDS = [
    (
        relation,
        [re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE) for kw in keywords]
    )
    for relation, keywords in RELATION_KEYWORDS
]

# Relations whose keywords already express a negative result ("did not differ",
# "did not show significant"); these must not get an extra DOES_NOT_ prefix.
NEGATION_ENCODED = {"NO_DIFFERENCE_IN"}

def classify_relation(sentence_text: str) -> str:
    """Classifies a sentence into a relation label using word-boundary matching."""
    for relation, patterns in COMPILED_RELATION_KEYWORDS:
        if any(pattern.search(sentence_text) for pattern in patterns):
            return relation
    return None

def _normalize(name: str) -> str:
    clean = re.sub(r'[\u200b\u200c\u200d\u200e\u200f\uFEFF\u00ad\u00a0]', ' ', name)
    return " ".join(clean.strip().lower().split())

def is_generic_entity(entity_text: str) -> bool:
    """Return True if the entity is an exact stop entity or consists entirely of stop words."""
    norm = entity_text.strip().lower()
    if not norm or norm in PRONOUNS or norm in STOP_ENTITIES or len(norm) <= 1:
        return True
    
    # Split into individual words to check if ALL tokens are generic words
    words = norm.split()
    if all(w in STOP_ENTITIES or w in PRONOUNS for w in words):
        return True
        
    return False

STOP_ENTITIES = {
    # Question & Meta-concept artifacts
    "specific outcome measurements", "outcome measurements", "outcome measures", 
    "outcomes", "significant improvement", "significant difference", "significantly higher", 
    "improvements", "improvement", "randomized controlled trial", "the randomized controlled trial", 
    "controlled trial", "rct", "conventional therapy", "conventional occupational therapy", 
    "conventional neglect therapy", "effectiveness", "efficacy", "feasibility", 
    "utility", "validity", "reliability", "scales", "scale", "score", "scores", 
    "testing", "test", "tests", "measure", "measures", "measurement", "measurements",

    # Prompt & Question Artifacts
    "physically entail", "originally introduced", "describe how", "explain the", 
    "what is", "who introduced", "how does", "according to", "defined as",
    "main purpose", "primary goal", "key features", "general overview", "specific tasks", "part of their", "conducted by", "performed",

    # Generic Interventional & Clinical Terms
    "intervention", "interventions", "treatment", "treatments", "therapy", 
    "therapies", "rehabilitation", "strategy", "strategies", "protocol", 
    "protocols", "method", "methods", "approach", "approaches", "procedure", 
    "procedures", "management", "care", "exposure", "regimen", "tasks", "task", "control group",
    "reading", "text", "string", "strings", "numerical string", 
    "evaluation", "assessment", "recite", "reciting",
    "reading/writing", "full-text", "heading", "training", "training session", "sessions", "session", "practice", "practices",

    # Broad Academic & Study Terminology
    "study", "studies", "trial", "trials", "paper", "author", "authors", 
    "researcher", "researchers", "finding", "findings", "result", "results", 
    "effect", "effects", "impact", "evidence", "data", "analysis", 
    "group", "groups", "cohort", "subject", "subjects", "patient", "patients",
    "percentage", "percentages", "proportion", "rate", "rates", 
    "number", "numbers", "differences", "total", "cases", "figure", "figures",

    # Abstract / Categorical Concepts
    "bottom-up", "top-down", "mechanism", "mechanisms", "process", "processes", 
    "factor", "factors", "level", "levels", "type", "types", "form", "forms", 
    "role", "roles", "condition", "conditions", "system", "systems"
}

def _clean_entity_text(text: str) -> str:
    """Strips PDF artifacts, normalizes spaces, and enforces a max token length."""
    # Remove OCR symbols, bullets, and PDF layout labels
    text = PDF_ARTIFACT_PATTERN.sub(" ", text)
    # Strip trailing possessive markers ("patients' ", "USN's ") so possessive
    # noun-chunk spans normalize to the same node as their bare form.
    text = re.sub(r"^[^\w]+|[^\w]+$", "", text)
    text = re.sub(r"(\w)['’]s\b", r"\1", text)
    text = re.sub(r"(\w)s['’]\s", r"\1s ", text)
    # Normalize whitespaces
    norm = _normalize(text)
    
    # Cap entities to max 8 words to avoid absorbing entire paragraphs/headers
    words = norm.split()
    while words and words[0] in LEADING_STRIP_WORDS:
        words = words[1:]
    norm = " ".join(words)

    if len(words) > 8:
        norm = " ".join(words[:8])
        
    return norm

def _get_bounded_phrase(token) -> str:
    """Extract phrase while blocking pronoun tokens from being included."""
    if token.pos_ == "PRON" or token.text.lower() in PRONOUNS:
        return ""

    # NOTE: "conj" is deliberately excluded. Including it glued coordinated
    # nouns together ("stroke and neglect" -> "stroke neglect"); conjuncts are
    # handled separately by the callers in extract_triples().
    allowed_deps = {"compound", "amod", "flat", "nmod", "nummod", "poss"}
    
    phrase_tokens = [
        child for child in token.children
        if child.dep_ in allowed_deps and child.pos_ != "PRON" and child.text.lower() not in PRONOUNS]
    phrase_tokens.append(token)
    phrase_tokens.sort(key=lambda x: x.i)
    
    raw_phrase = " ".join([t.text for t in phrase_tokens])
    return _clean_entity_text(raw_phrase)

def extract_entities(text: str):
    """Return deduplicated entity strings, splitting parenthetical acronyms into separate entities."""
    # Pre-extract parenthetical acronyms/terms before spaCy parses them as one giant span
    parentheticals = re.findall(r"\(([A-Za-z0-9\-]{2,15})\)", text)
    text_no_parens = re.sub(r"\s*\([A-Za-z0-9\-]{2,15}\)", "", text)

    doc = get_nlp()(text_no_parens)
    seen = set()
    result = []

    def _add(candidate: str):
        norm = _clean_entity_text(candidate)
        if len(norm) > 1 and norm not in seen and not is_generic_entity(norm):
            seen.add(norm)
            result.append(norm)

    for cite in extract_citations(text):
        _add(cite)

    for p in parentheticals:
        _add(p)

    for ent in doc.ents:
        _add(ent.text)

    for chunk in doc.noun_chunks:
        chunk_clean = _clean_entity_text(chunk.text)
        if not chunk_clean or chunk_clean in seen or is_generic_entity(chunk_clean):
            continue
        words = chunk_clean.split()
        if len(words) > 1:
            _add(chunk_clean)
            # Also add the head noun if the chunk is long (> 3 words) so specific targets aren't lost
            if len(words) > 3 and chunk.root.pos_ in ("NOUN", "PROPN") and len(chunk.root.text) >= 4:
                _add(chunk.root.text)
        else:
            root = chunk.root
            if root.pos_ in ("NOUN", "PROPN") and len(chunk_clean) >= 3:
                _add(chunk_clean)

    return result


def extract_triples(chunk_text: str):
    clean_text = unicodedata.normalize("NFKC", chunk_text).replace("\xad", "").replace("\u00ad", "")
    doc = get_nlp()(clean_text)
    triples = []
    seen_triples = set()

    # Pronoun-resolution heuristic: the subject of the most recent ROOT verb.
    last_valid_subject = None

    for sent in doc.sents:
        for token in sent:
            if token.pos_ in ("VERB", "AUX") or token.dep_ == "ROOT":
                # Classify on this verb's own clause, not the whole sentence, so a
                # keyword in one clause doesn't relabel every verb in the sentence.
                clause = doc[token.left_edge.i : token.right_edge.i + 1]
                taxonomy_rel = classify_relation(clause.text)
                relation_label = taxonomy_rel if taxonomy_rel else token.lemma_.upper()
                is_negated = any(child.dep_ == "neg" or child.text.lower() in ("not", "never", "no", "fails",
                 "failed") for child in token.children)
                # Labels in NEGATION_ENCODED already express the negation
                # (e.g. "did not differ" -> NO_DIFFERENCE_IN); prefixing them
                # again produced double negatives like DOES_NOT_NO_DIFFERENCE_IN.
                if is_negated and relation_label not in NEGATION_ENCODED:
                    relation_label = f"DOES_NOT_{relation_label}"
                if not taxonomy_rel and token.lemma_ in ("be", "have", "do", "use"):
                    continue

                has_pronoun_subject = any(
                    child.dep_ in ("nsubj", "nsubjpass")
                    and (child.pos_ == "PRON" or child.text.lower() in PRONOUNS)
                    for child in token.children
                )

                subjects = []
                objects = []

                for child in token.children:
                    # Ignore pronoun children directly
                    if child.pos_ == "PRON" or child.text.lower() in PRONOUNS:
                        continue

                    # Extract nominal subjects
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        sub_phrase = _get_bounded_phrase(child)
                        if sub_phrase and not is_generic_entity(sub_phrase):
                            subjects.append(sub_phrase)
                            for conj in child.children:
                                if conj.dep_ == "conj":
                                    conj_phrase = _get_bounded_phrase(conj)
                                    if conj_phrase and not is_generic_entity(conj_phrase):
                                        subjects.append(conj_phrase)

                    # Extract objects, complements, and passive agents
                    elif child.dep_ in ("dobj", "attr", "acomp", "oprd", "agent"):
                        obj_phrase = _get_bounded_phrase(child)
                        if obj_phrase and not is_generic_entity(obj_phrase):
                            objects.append(obj_phrase)
                        for conj in child.children:
                            if conj.dep_ == "conj":
                                cphrase = _get_bounded_phrase(conj)
                                if cphrase and not is_generic_entity(cphrase):
                                    objects.append(cphrase)

                    # Prepositional objects
                    elif child.dep_ == "prep":
                        for pchild in child.children:
                            if pchild.dep_ in ("pobj", "pcomp") and pchild.pos_ != "PRON" and pchild.text.lower() not in PRONOUNS:
                                pobj_phrase = _get_bounded_phrase(pchild)
                                if pobj_phrase and not is_generic_entity(pobj_phrase):
                                    objects.append(pobj_phrase)
                                    for conj in pchild.children:
                                        if conj.dep_ == "conj":
                                            cphrase = _get_bounded_phrase(conj)
                                            if cphrase and not is_generic_entity(cphrase):
                                                objects.append(cphrase)

                # Pronoun resolution: only when this verb's subject is a pronoun
                # ("It improved ...") do we substitute the last ROOT subject.
                if not subjects and objects and has_pronoun_subject and last_valid_subject:
                    subjects = [last_valid_subject]
                elif subjects and token.dep_ == "ROOT":
                    last_valid_subject = subjects[0]

                # Pair valid non-pronoun subjects and objects
                for subj in subjects:
                    for obj in objects:
                        if subj != obj and len(subj) > 1 and len(obj) > 1:
                            triple_key = (subj, relation_label, obj)
                            if triple_key not in seen_triples:
                                seen_triples.add(triple_key)
                                triples.append({
                                    "subject": subj,
                                    "relation": relation_label,
                                    "object": obj
                                })

    return triples