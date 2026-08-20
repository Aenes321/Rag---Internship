import json
import requests
import config

def _call_ollama(prompt: str, system: str = None, json_mode: bool = False) -> (str, dict):
    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system
    if json_mode:
        payload["format"] = "json"

    response = requests.post(f"{config.OLLAMA_URL}/api/generate", json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()

    # Ollama returns durations in nanoseconds (1 second = 1e9 nanoseconds)
    eval_count = data.get("eval_count", 0)
    eval_duration_sec = data.get("eval_duration", 0) / 1e9

    prompt_eval_count = data.get("prompt_eval_count", 0)
    prompt_eval_sec = data.get("prompt_eval_duration", 0) / 1e9

    total_duration_sec = data.get("total_duration", 0) / 1e9

    metrics = {
        "prompt_tokens": prompt_eval_count,
        "generated_tokens": eval_count,
        "generation_speed_tok_sec": round(eval_count / eval_duration_sec, 2) if eval_duration_sec > 0 else 0,
        "prompt_speed_tok_sec": round(prompt_eval_count / prompt_eval_sec, 2) if prompt_eval_sec > 0 else 0,
        "llm_total_sec": round(total_duration_sec, 2),
    }

    return data["response"], metrics


ENTITY_SYSTEM = """Extract the key medical entities (diseases, symptoms, drugs,
treatments, tests, risk factors) mentioned in the user's question.
Return ONLY valid JSON: {"entities": ["...", "..."]}
Keep each entity short (a few words)."""


def extract_query_entities(question: str):
    raw = _call_ollama(question, system=ENTITY_SYSTEM, json_mode=True)
    try:
        data = json.loads(raw)
        return data.get("entities", [])
    except (json.JSONDecodeError, AttributeError):
        return []
RELATION_TEMPLATES = {
    "NO_DIFFERENCE_IN": "showed no significant difference in",
    "SHOWED_IMPROVEMENT_IN": "showed significant improvement in",
    "MANIFESTS_AS": "manifests as",
    "ASSESSED_BY": "is assessed by",
    "AFFECTS_REGION": "affects",
    "SUBTYPE_OF": "is a subtype of",
    "TREATS": "is used to treat",
    "CAUSES": "causes",
    "HAS_SYMPTOM": "has the symptom of",
    "RISK_FACTOR_FOR": "is a risk factor for",
    "INTERACTS_WITH": "interacts with",
    "DIAGNOSED_BY": "is diagnosed by",
    "COMPLICATION_OF": "is a complication of",
    "PREVENTS": "helps prevent",
    "ASSOCIATED_WITH": "is associated with",
}

def humanize_relation(relation: str) -> str:
    """Convert a Cypher relation type (fixed taxonomy or dynamic verb-lemma
    fallback) into a natural-language phrase for prompting."""
    if relation in RELATION_TEMPLATES:
        return RELATION_TEMPLATES[relation]

    # Dynamic fallback relations from ner.py are token.lemma_.upper(), e.g.
    # "STIMULATE", "PREDICT", "SHARE" -> best-effort present-tense verb.
    words = relation.replace("_", " ").lower().split()
    if not words:
        return "is related to"
    if len(words) == 1:
        verb = words[0]
        if verb.endswith(("s", "sh", "ch", "x", "z")):
            verb += "es"
        elif verb.endswith("y") and len(verb) > 1 and verb[-2] not in "aeiou":
            verb = verb[:-1] + "ies"
        else:
            verb += "s"
        return verb
    return " ".join(words)

ANSWER_SYSTEM = """You are a medical information assistant. Answer the user's
question using ONLY the facts and source text provided below. If the
information isn't sufficient to answer, say so plainly. Be clear, concise,
and do not invent information that isn't supported by the provided context. 

CRITICAL FORMATTING INSTRUCTIONS:
1. Write in natural, fluid sentences and paragraphs.
2. NEVER output raw relationship tags or uppercase keywords (e.g., DO NOT write "ASSESSED_BY", "AFFECTS", or "STIMULATE" in capital letters).
3. Integrate entities and relations smoothly into grammatical prose.
4. If the information isn't sufficient to give a definitive answer, state that clearly."""


def generate_answer(question: str, facts: list, source_snippets: list) -> str:
    facts_str = "\n".join(f"- {f['source']} {humanize_relation(f['relation'])} {f['target']}" for f in facts) or "(none)"
    sources_str = "\n---\n".join(source_snippets) or "(none)"
    prompt = f"""Question: {question}

Graph facts:
{facts_str}

Source text excerpts:
{sources_str}

Answer the question using only the information above."""
    return _call_ollama(prompt, system=ANSWER_SYSTEM)
