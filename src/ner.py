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

PATTERNS = [{"label": "CONDITION", "pattern": [{"LOWER": "spatial"}, {"LOWER": "neglect"}]}]

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(config.SPACY_MODEL)
        ruler = _nlp.add_pipe(
            "entity_ruler",
            after="ner",
            config={"overwrite_ents": True},
        )
        ruler.add_patterns(PATTERNS)
    return _nlp


# Order matters: first matching keyword set wins. Keep specific relations
# before the generic fallback.
RELATION_KEYWORDS = [
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
    
    # Captures non-significant / comparable outcomes
    ("NO_DIFFERENCE_IN", ["no significant difference", "did not differ", "no difference", 
        "comparable to", "similar improvement", "no statistical difference","remained similar", "did not show significant"]),
    
    # Captures significant improvements/outcomes
    ("SHOWED_IMPROVEMENT_IN", ["significantly higher", "significant improvement", "greater improvement",
        "superior to", "outperformed", "showed higher", "showed greater", "showed significant", "improved more",]),

    # Behavioral/task performance keywords
    ("MANIFESTS_AS", ["recite", "recites", "read", "reads", "omit", "omits", "presented with", "demonstrate", "displays"]),
    
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
    
    # Cap entities to max 8 words to avoid absorbing entire paragraphs/headers
    words = norm.split()
    if len(words) > 8:
        norm = " ".join(words[:8])
        
    return norm

def _get_bounded_phrase(token) -> str:
    """Extracts entity phrase including compounds, adjectives, and conjunctions."""
    allowed_deps = {"compound", "amod", "flat", "nmod", "nummod", "poss", "conj"}
    
    phrase_tokens = [child for child in token.children if child.dep_ in allowed_deps]
    phrase_tokens.append(token)
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
    clean_text = unicodedata.normalize("NFKC", chunk_text).replace("\xad", "").replace("\u00ad", "")
    doc = get_nlp()(clean_text)
    triples = []
    seen_triples = set()

    for sent in doc.sents:
        # 1. First preference: Schema-aligned keyword relation
        taxonomy_rel = classify_relation(sent.text)

        for token in sent:
            # Look at verbs, root predicates, or auxiliary verbs
            if token.pos_ in ("VERB", "AUX") or token.dep_ == "ROOT":
                
                # Determine relation label: Prefer Taxonomy Keyword -> Fallback to dynamic Verb
                relation_label = taxonomy_rel if taxonomy_rel else token.lemma_.upper()
                
                # Skip meaningless generic verbs when using dynamic fallback
                if not taxonomy_rel and token.lemma_ in ("be", "have", "do", "use"):
                    continue

                subjects = []
                objects = []

                # Find all subjects, direct objects, and prepositional targets
                for child in token.children:
                    # Nominal Subjects (Active or Passive)
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        sub_phrase = _get_bounded_phrase(child)
                        if sub_phrase and not is_generic_entity(sub_phrase):
                            subjects.append(sub_phrase)
                            # Handle conjunctions (e.g., "Aravind and Lamontagne")
                            for conj in child.children:
                                if conj.dep_ == "conj":
                                    conj_phrase = _get_bounded_phrase(conj)
                                    if conj_phrase and not is_generic_entity(conj_phrase):
                                        subjects.append(conj_phrase)

                    # Direct Objects, Attributes, Complements
                    elif child.dep_ in ("dobj", "attr", "acomp", "oprd"):
                        obj_phrase = _get_bounded_phrase(child)
                        if obj_phrase and not is_generic_entity(obj_phrase):
                            objects.append(obj_phrase)
                        for conj in child.children:
                            if conj.dep_ == "conj":
                                cphrase = _get_bounded_phrase(conj)
                                if cphrase and not is_generic_entity(cphrase):
                                    objects.append(cphrase)
                    # Prepositional Objects (e.g., "tested in VR")
                    elif child.dep_ == "prep":
                        for pchild in child.children:
                            if pchild.dep_ in ("pobj", "pcomp"):
                                pobj_phrase = _get_bounded_phrase(pchild)
                                if pobj_phrase and not is_generic_entity(pobj_phrase):
                                    objects.append(pobj_phrase)

                # Pair every subject with every object found around this verb
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