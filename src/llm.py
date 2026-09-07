import json
import requests
import config
import re

def clean_llm_citations(text: str) -> str:
    """Removes parenthetical citations like (Author et al., 2020) or [1, 2] from answers."""
    # Remove bracketed numbers: [1], [1-3]
    text = re.sub(r"\[\s*\d+(?:\s*[-–,]\s*\d+)*\s*\]", "", text)
    # Remove author citations: (Author et al., 2020), (Smith, 2019; Jones, 2021)
    text = re.sub(r"\([A-Z][A-Za-z\s–\-]+,?\s*(?:et\s*al\.?,?)?\s*\d{4}[a-z]?(?:;[^\)]+)?\)", "", text)
    # Collapse double spaces and clean punctuation gaps
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def _call_ollama(prompt: str, system: str = None, json_mode: bool = False) -> tuple[str, dict]:
    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 500,   # Increased from 300 to eliminate cut-offs
        }
    }
    if system:
        payload["system"] = system
    if json_mode:
        payload["format"] = "json"

    timeout_limit = 180
    try:
        response = requests.post(f"{config.OLLAMA_URL}/api/generate", json=payload, timeout=timeout_limit)
        response.raise_for_status()
        data = response.json()

        eval_count = data.get("eval_count", 0)
        eval_duration_sec = data.get("eval_duration", 0) / 1e9
        prompt_eval_count = data.get("prompt_eval_count", 0)
        prompt_eval_sec = data.get("prompt_eval_duration", 0) / 1e9
        total_duration_sec = data.get("total_duration", 0) / 1e9

        metrics = {
            "prompt_eval_latency": round(prompt_eval_sec, 2),
            "prompt_tokens": prompt_eval_count,
            "prompt_speed_tok_sec": round(prompt_eval_count / prompt_eval_sec, 2) if prompt_eval_sec > 0 else 0,
            "generation_eval_latency": round(eval_duration_sec, 2),
            "generated_tokens": eval_count,
            "generation_speed_tok_sec": round(eval_count / eval_duration_sec, 2) if eval_duration_sec > 0 else 0,
            "llm_latency_sec": round(total_duration_sec, 2),
        }
        return data["response"].strip(), metrics

    except requests.exceptions.Timeout:
        return "Error: LLM response timed out.", {
            "prompt_tokens": 0, "generated_tokens": 0,
            "generation_speed_tok_sec": 0.0, "prompt_speed_tok_sec": 0.0,
            "llm_latency_sec": timeout_limit, "prompt_eval_latency": 0.0, "generation_eval_latency": 0.0
        }


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
    if relation in RELATION_TEMPLATES:
        return RELATION_TEMPLATES[relation]
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


ANSWER_SYSTEM = """You are a precise medical synthesizer. Provide compact, direct, and factually exact clinical answers strictly from the context."""

def generate_answer(question: str, facts: list, source_snippets: list) -> tuple[str, dict]:
    facts_str = "\n".join(f"- {f['source']} {humanize_relation(f['relation'])} {f['target']}" for f in facts) or "(none)"
    sources_str = "\n---\n".join(source_snippets) or "(none)"

    prompt = f"""CONTEXT:
{sources_str}

GRAPH FACTS:
{facts_str}

QUESTION:
{question}

CRITICAL RULES:
1. Answer directly in the very first sentence.
2. Keep the answer strictly within 2 to 3 concise sentences. Do not provide background introductions.
3. If comparing two things, only state contrasts explicitly confirmed in the context.
4. Do not include any author names or year citations. State only the facts.


ANSWER:"""

    raw_answer, metrics = _call_ollama(prompt, system=ANSWER_SYSTEM)
    cleaned_answer = clean_llm_citations(raw_answer)
    return cleaned_answer, metrics