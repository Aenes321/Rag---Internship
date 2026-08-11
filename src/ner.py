import re
from itertools import combinations
import spacy
import config

_nlp = None

# Regex patterns for citations (e.g., "Choi et al.", "Choi et al. (2021)", "Choi et al., 2021")
CITATION_PATTERN = re.compile(
    r'\b[A-Z][a-zA-Z\-]+(?:\s+et\s+al\.?|\s+and\s+[A-Z][a-zA-Z\-]+)?(?:\s*\(?\d{4}\)?)?\b'
)

DIGIT_PATTERN = re.compile(r'\b\d{3,}\b')

def extract_citations(text: str) -> list[str]:
    """Find author/year citation strings in text."""
    matches = CITATION_PATTERN.findall(text)
    clean_citations = []
    for m in matches:
        norm = _normalize(m).rstrip("(").strip()
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
    return "ASSOCIATED_WITH"  # fallback: entities co-occur but no clear trigger word


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
    "reading/writing", "full-text", "heading"

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
    """Return a list of {subject, relation, object} dicts for a chunk,
    by pairing co-occurring entities within each sentence."""
    doc = get_nlp()(chunk_text)
    triples = []

    for sent in doc.sents:

        relation = classify_relation(sent.text)
        if not relation:
            continue

        seen = set()
        ents = []

        citations = extract_citations(sent.text)
        for cite in citations:
            if cite not in seen:
                seen.add(cite)
                ents.append(cite)

        for ent in sent.ents:
            name = _normalize(ent.text)
            if len(name) > 1 and name not in seen and not is_generic_entity(name):
                seen.add(name)
                ents.append(name)

        if len(ents) < 2:
            continue

        
        for i in range(len(ents) - 1):
            triples.append({
                "subject": ents[i],
                "relation": relation,
                "object": ents[i + 1]
            })

    return triples
