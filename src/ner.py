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

# Regex patterns for citations (e.g., "Choi et al.", "Choi et al. (2021)", "Choi et al., 2021")
CITATION_PATTERN = re.compile(
    r'\b[A-Z][a-zA-Z\-]+(?:\s+et\s+al\.?|\s+and\s+[A-Z][a-zA-Z\-]+)?(?:\s*,?\s*\(?\d{4}\)?)?\b'
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
        if "entity_ruler" not in _nlp.pipe_names:
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
    # Captures non-significant / comparable outcomes
    ("NO_DIFFERENCE_IN", ["no significant difference", "did not differ", "no difference", 
        "comparable to", "similar improvement", "no statistical difference","remained similar", "did not show significant","no change in", 
        "statistically insignificant", "equivalent to"]),
    
    # Captures significant improvements/outcomes
    ("SHOWED_IMPROVEMENT_IN", ["significantly higher", "significant improvement", "greater improvement",
        "superior to", "outperformed", "showed higher", "showed greater", "showed significant", "improved more","significantly reduced", 
        "significantly increased", "statistically significant"]),

    # Spatial Neglect Specific Relations
    
    ("AFFECTS_REGION", ["lesion in", "lesioned in", "damage to", "parietal", "cortex", "hemisphere", "lobe", "pathology in","structural damage", 
        "localized to", "brain region", "cortical", "subcortical", "frontal", "temporal", "occipital", "parietal", "insula", "basal ganglia"]),
    ("SUBTYPE_OF", ["subtype of", "classified as", "type of neglect", "peripersonal", "extrapersonal", "personal neglect", "egocentric", "allocentric", "form of neglect"]),
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
    # Behavioral/task performance keywords
    ("MANIFESTS_AS", ["recite", "recites", "read", "reads", "presented with", "demonstrate", "displays"]),
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
    """Extract phrase while blocking pronoun tokens from being included."""
    if token.pos_ == "PRON" or token.text.lower() in PRONOUNS:
        return ""

    allowed_deps = {"compound", "amod", "flat", "nmod", "nummod", "poss", "conj"}
    
    phrase_tokens = [
        child for child in token.children
        if child.dep_ in allowed_deps and child.pos_ != "PRON" and child.text.lower() not in PRONOUNS]
    phrase_tokens.append(token)
    phrase_tokens.sort(key=lambda x: x.i)
    
    raw_phrase = " ".join([t.text for t in phrase_tokens])
    return _clean_entity_text(raw_phrase)

def extract_entities(text: str):
    """Return deduplicated entity strings and meaningful noun chunks."""
    doc = get_nlp()(text)
    seen = set()
    result = []

    citations = extract_citations(text)
    for cite in citations:
        if cite not in seen and not is_generic_entity(cite):
            seen.add(cite)
            result.append(cite)

    for ent in doc.ents:
        name = _normalize(ent.text)
        if len(name) > 1 and name not in seen and not is_generic_entity(name):
            seen.add(name)
            result.append(name)

    # 3. Noun Chunks (Captures complex clinical & financial metrics)
    for chunk in doc.noun_chunks:
        chunk_clean = _clean_entity_text(chunk.text)
        if len(chunk_clean.split()) > 1 and chunk_clean not in seen and not is_generic_entity(chunk_clean):
            seen.add(chunk_clean)
            result.append(chunk_clean)

    return result


def extract_triples(chunk_text: str):
    clean_text = unicodedata.normalize("NFKC", chunk_text).replace("\xad", "").replace("\u00ad", "")
    doc = get_nlp()(clean_text)
    triples = []
    seen_triples = set()

    last_valid_subject = None  # Heuristic for resolving pronouns across sentences

    for sent in doc.sents:
        taxonomy_rel = classify_relation(sent.text)

        for token in sent:
            if token.pos_ in ("VERB", "AUX") or token.dep_ == "ROOT":
                relation_label = taxonomy_rel if taxonomy_rel else token.lemma_.upper()
                
                if not taxonomy_rel and token.lemma_ in ("be", "have", "do", "use"):
                    continue

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

                # Last-subject heuristic: Re-use previous valid subject if sentence used a pronoun
                if not subjects and objects and last_valid_subject:
                    subjects = [last_valid_subject]

                if subjects:
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