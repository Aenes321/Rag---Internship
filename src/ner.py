"""
Fast, local triple extraction — no LLM calls.

Uses scispaCy for biomedical named entity recognition, then classifies the
relation between co-occurring entities in a sentence using keyword triggers.
This trades some precision/nuance for a large speed gain over per-chunk LLM
extraction (milliseconds vs. ~20-30s per chunk).
"""
from itertools import combinations
import spacy
import config

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(config.SPACY_MODEL)
    return _nlp


# Order matters: first matching keyword set wins. Keep specific relations
# before the generic fallback.
RELATION_KEYWORDS = [
    ("TREATS", ["treat", "therapy", "treatment for", "manage", "relieve", "alleviate"]),
    ("CAUSES", ["cause", "causes", "leads to", "lead to", "results in", "result in", "triggers"]),
    ("HAS_SYMPTOM", ["symptom", "present with", "presents with", "sign of", "characterized by"]),
    ("RISK_FACTOR_FOR", ["risk factor", "increases the risk", "increase the risk", "predispose"]),
    ("INTERACTS_WITH", ["interact", "interaction", "contraindicat"]),
    ("DIAGNOSED_BY", ["diagnos", "screen for", "detect", "confirmed by", "test for"]),
    ("COMPLICATION_OF", ["complication"]),
    ("PREVENTS", ["prevent", "reduces the risk", "reduce the risk", "protect against"]),
]


def classify_relation(sentence_text: str) -> str:
    s = sentence_text.lower()
    for relation, keywords in RELATION_KEYWORDS:
        if any(kw in s for kw in keywords):
            return relation
    return "ASSOCIATED_WITH"  # fallback: entities co-occur but no clear trigger word


def _normalize(name: str) -> str:
    return " ".join(name.strip().lower().split())


def extract_entities(text: str):
    """Return deduplicated entity strings found in text (used at query time)."""
    doc = get_nlp()(text)
    seen = set()
    result = []
    for ent in doc.ents:
        name = _normalize(ent.text)
        if len(name) > 1 and name not in seen:
            seen.add(name)
            result.append(name)
    return result


def extract_triples(chunk_text: str):
    """Return a list of {subject, relation, object} dicts for a chunk,
    by pairing co-occurring entities within each sentence."""
    doc = get_nlp()(chunk_text)
    triples = []

    for sent in doc.sents:
        seen = set()
        ents = []
        for ent in sent.ents:
            name = _normalize(ent.text)
            if len(name) > 1 and name not in seen:
                seen.add(name)
                ents.append(name)

        if len(ents) < 2:
            continue

        relation = classify_relation(sent.text)
        for a, b in combinations(ents, 2):
            triples.append({"subject": a, "relation": relation, "object": b})

    return triples
