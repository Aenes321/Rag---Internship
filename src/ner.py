import re
from itertools import combinations
import spacy
import config
import unicodedata
_nlp = None

# Regex patterns for citations (e.g., "Choi et al.", "Choi et al. (2021)", "Choi et al., 2021")
CITATION_PATTERN = re.compile(
    r'\b[A-Z][a-zA-Z\-]+(?:\s+et\s+al\.?|\s+and\s+[A-Z][a-zA-Z\-]+)?(?:\s*\(?\d{4}\)?)?\b'
)

DIGIT_PATTERN = re.compile(r'\b\d{3,}\b')

# 1. Regex to strip PDF layout noise, page/box labels, and special symbols
PDF_ARTIFACT_PATTERN = re.compile(r"[▸•*■▪│\|–—\t]|\b(box|page|fig|figure|table)\s*\d+\b", re.IGNORECASE)

def extract_citations(text: str) -> list[str]:
    """Find author/year citation strings in text."""
    matches = CITATION_PATTERN.findall(text)
    clean_citations = []
    for m in matches:
        norm = _normalize(re.sub(r'[\(\)]', '', m))
        # Ensure it actually contains an author-like indicator
        if "et al" in norm or any(char.isdigit() for char in norm):
            clean_citations.append(norm)
    return clean_citations

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(config.SPACY_MODEL)
    return _nlp


# Order matters: first matching keyword set wins. Keep specific relations
# before the generic fallback.
RELATION_KEYWORDS = [
    # Captures non-significant / comparable outcomes
    ("NO_DIFFERENCE_IN", ["no significant difference", "did not differ", "no difference", 
        "comparable to", "similar improvement", "no statistical difference","remained similar", "did not show significant"]),
    
    # Captures significant improvements/outcomes
    ("SHOWED_IMPROVEMENT_IN", ["significantly higher", "significant improvement", "greater improvement",
        "superior to", "outperformed", "showed higher", "showed greater", "showed significant", "improved more",]),

    # Behavioral/task performance keywords
    ("MANIFESTS_AS", ["recite", "recites", "read", "reads", "omit", "omits", "presented with", "demonstrate", "displays"]),
    # Spatial Neglect Specific Relations
    ("ASSESSED_BY", ["assess", "evaluated by", "measured by", "score", "scale", "test", "cancellation", "bisection"]),
    ("AFFECTS_REGION", ["lesion", "damage to", "parietal", "cortex", "hemisphere", "lobe", "pathology in"]),
    ("SUBTYPE_OF", ["subtype", "manifests as", "classified as", "peripersonal", "extrapersonal", "personal neglect"]),

    # Core Clinical Relations
    ("TREATS", ["treat", "therapy", "treatment", "rehabilitation", "manage", "relieve", "alleviate", "adaptation", "stimulation"]),
    ("CAUSES", ["cause", "causes", "leads to", "lead to", "results in", "result in", "triggers", "secondary to"]),
    ("HAS_SYMPTOM", ["symptom", "present with", "presents with", "sign of", "characterized by", "manifested by", "deficit"]),
    ("RISK_FACTOR_FOR", ["risk factor", "increases the risk", "increase the risk", "predispose"]),
    ("INTERACTS_WITH", ["interact", "interaction", "contraindicat"]),
    ("DIAGNOSED_BY", ["diagnos", "screen for", "detect", "confirmed by"]),
    ("COMPLICATION_OF", ["complication", "sequela"]),
    ("PREVENTS", ["prevent", "reduces the risk", "reduce the risk", "protect against"]),
    # Generic fallback relation
    ("ASSOCIATED_WITH", ["associated with", "related to", "co-occur", "co-occurred", "co-occurring", "correlated with"]),
]


def classify_relation(sentence_text: str) -> str:
    s = sentence_text.lower()
    for relation, keywords in RELATION_KEYWORDS:
        if any(kw in s for kw in keywords):
            return relation
    return None  # fallback: entities co-occur but no clear trigger word


def _normalize(name: str) -> str:
    return " ".join(name.strip().lower().split())

def is_generic_entity(entity_text: str) -> bool:
    """Return True if the entity is an exact stop entity or consists entirely of stop words."""
    norm = entity_text.strip().lower()
    if norm in STOP_ENTITIES:
        return True
    
    # Split into individual words to check if ALL tokens are generic words
    words = norm.split()
    if all(w in STOP_ENTITIES for w in words):
        return True
        
    return False

STOP_ENTITIES = {
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
    "evaluation", "assessment", "task", "tasks", "recite", "reciting",
    "reading/writing", "full-text", "heading", "training", "training session", "sessions", "session", "practice", "practices",

    # Broad Academic & Study Terminology
    "study", "studies", "trial", "trials", "paper", "author", "authors", 
    "researcher", "researchers", "finding", "findings", "result", "results", 
    "effect", "effects", "impact", "evidence", "data", "analysis", 
    "group", "groups", "cohort", "subject", "subjects", "patient", "patients",

    # Abstract / Categorical Concepts
    "bottom-up", "top-down", "mechanism", "mechanisms", "process", "processes", 
    "factor", "factors", "level", "levels", "type", "types", "form", "forms", 
    "role", "roles", "condition", "conditions", "system", "systems"
}

def _clean_entity_text(text: str) -> str:
    """Strips PDF artifacts, normalizes spaces, and enforces a max token length."""
    # Remove OCR symbols, bullets, and PDF layout labels
    text = PDF_ARTIFACT_PATTERN.sub(" ", text)
    # Remove floating digits/punctuation at start or end
    text = re.sub(r"^[^\w]+|[^\w]+$", "", text)
    # Normalize whitespaces
    norm = _normalize(text)
    
    # Cap entities to max 5 words to avoid absorbing entire paragraphs/headers
    words = norm.split()
    if len(words) > 5:
        norm = " ".join(words[:5])
        
    return norm

def _get_bounded_phrase(token) -> str:
    """Collects only immediate modifiers (compounds, adjectives, numbers) 
    instead of traversing the full subtree.
    """
    allowed_deps = {"compound", "amod", "flat", "nmod", "nummod", "poss"}
    
    phrase_tokens = [
        child for child in token.children if child.dep_ in allowed_deps
    ]
    phrase_tokens.append(token)
    
    # Sort tokens in their natural sentence order
    phrase_tokens.sort(key=lambda x: x.i)
    
    raw_phrase = " ".join([t.text for t in phrase_tokens])
    return _clean_entity_text(raw_phrase)

def extract_entities(text: str):
    """Return deduplicated entity strings found in text (used at query time)."""
    doc = get_nlp()(text)
    seen = set()
    result = []

    for num in DIGIT_PATTERN.findall(text):
        if num not in seen:
            seen.add(num)
            result.append(num)

    citations = extract_citations(text)
    for cite in citations:
        if cite not in seen and cite not in STOP_ENTITIES:
            seen.add(cite)
            result.append(cite)

    for ent in doc.ents:
        name = _normalize(ent.text)
        if len(name) > 1 and name not in seen and not is_generic_entity(name):
            seen.add(name)
            result.append(name)
    return result



def extract_triples(chunk_text: str):
    """Dependency-driven triple extraction with bounded entity lengths and PDF noise filtering."""
    # Clean unicode artifacts
    clean_text = unicodedata.normalize("NFKC", chunk_text).replace("\xad", "").replace("\u00ad", "")
    doc = get_nlp()(clean_text)
    triples = []

    for sent in doc.sents:
        classified_rel = classify_relation(sent.text)
        if not classified_rel:
            continue

        for token in sent:
            if token.pos_ in ("VERB", "AUX") or token.dep_ == "ROOT":
                subj = None
                direct_obj = None
                prep_objs = []

                for child in token.children:
                    # Capture Subjects
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        candidate = _get_bounded_phrase(child)
                        if candidate and not is_generic_entity(candidate):
                            subj = candidate

                    # Capture Direct Objects / Predicates
                    elif child.dep_ in ("dobj", "attr", "acomp"):
                        candidate = _get_bounded_phrase(child)
                        if candidate and not is_generic_entity(candidate):
                            direct_obj = candidate

                    # Capture Prepositional Objects (e.g., "in VR")
                    elif child.dep_ == "prep":
                        for pchild in child.children:
                            if pchild.dep_ in ("pobj", "pcomp"):
                                candidate = _get_bounded_phrase(pchild)
                                if candidate and not is_generic_entity(candidate):
                                    prep_objs.append(candidate)

                # Form Triples with Length Safeguards
                if subj and direct_obj and len(subj) > 1 and len(direct_obj) > 1:
                    triples.append({
                        "subject": subj,
                        "relation": classified_rel,
                        "object": direct_obj
                    })

                if subj and prep_objs:
                    for pobj in prep_objs:
                        if len(pobj) > 1 and len(subj) > 1:
                            triples.append({
                                "subject": subj,
                                "relation": classified_rel,
                                "object": pobj
                            })

                elif direct_obj and prep_objs:
                    for pobj in prep_objs:
                        if len(pobj) > 1 and len(direct_obj) > 1:
                            triples.append({
                                "subject": direct_obj,
                                "relation": classified_rel,
                                "object": pobj
                            })

    return triples
