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

def _error_metrics(latency: float) -> dict:
    return {
        "prompt_tokens": 0, "generated_tokens": 0,
        "generation_speed_tok_sec": 0.0, "prompt_speed_tok_sec": 0.0,
        "llm_latency_sec": round(latency, 2), "prompt_eval_latency": 0.0, "generation_eval_latency": 0.0,
    }


def _call_ollama(prompt: str, system: str = None, json_mode: bool = False) -> tuple[str, dict]:
    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 500,   # Increased from 300 to eliminate cut-offs
            # Ollama's default context window is small (2048/4096 depending on
            # version) and it silently drops the *start* of over-long prompts,
            # i.e. the retrieved CONTEXT. Prompts here reach ~3k tokens.
            "num_ctx": config.OLLAMA_NUM_CTX,
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
        return "Error: LLM response timed out.", _error_metrics(timeout_limit)
    except (requests.exceptions.RequestException, ValueError, KeyError) as exc:
        # Connection refused, HTTP errors, malformed JSON / missing "response".
        # Previously these propagated and aborted the whole evaluation loop.
        return f"Error: LLM request failed ({type(exc).__name__}: {exc}).", _error_metrics(0.0)


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

NEGATED_TEMPLATES = {
    "NO_DIFFERENCE_IN": "showed a significant difference in",
    "SHOWED_IMPROVEMENT_IN": "did not show significant improvement in",
    "MANIFESTS_AS": "does not manifest as",
    "ASSESSED_BY": "is not assessed by",
    "AFFECTS_REGION": "does not affect",
    "SUBTYPE_OF": "is not a subtype of",
    "TREATS": "does not treat",
    "CAUSES": "does not cause",
    "HAS_SYMPTOM": "does not have the symptom of",
    "RISK_FACTOR_FOR": "is not a risk factor for",
    "INTERACTS_WITH": "does not interact with",
    "DIAGNOSED_BY": "is not diagnosed by",
    "COMPLICATION_OF": "is not a complication of",
    "PREVENTS": "does not prevent",
    "ASSOCIATED_WITH": "is not associated with",
}

def humanize_relation(relation: str) -> str:
    if relation.startswith("DOES_NOT_"):
        base = relation[len("DOES_NOT_"):]
        if base in NEGATED_TEMPLATES:
            return NEGATED_TEMPLATES[base]
        # Lemma-based relations are already in base form ("IMPROVE", "REDUCE").
        return "does not " + (" ".join(base.replace("_", " ").lower().split()) or "relate to")
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


ANSWER_SYSTEM = (
    "You are an expert clinical and neuroscientific researcher. "
    "Provide comprehensive, factually exact answers based strictly on the provided context. "
    "Include all relevant clinical metrics, percentages, test names, anatomical structures, "
    "and subgroup comparisons mentioned in the context that address every part of the question."
)

def generate_answer(question: str, facts: list, source_snippets: list) -> tuple[str, dict]:
    facts_str = "\n".join(f"- {f['source']} {humanize_relation(f['relation'])} {f['target']}" for f in facts) or "(none)"
    sources_str = "\n---\n".join(source_snippets) or "(none)"

    prompt = f"""CONTEXT:
{sources_str}

SUPPORTING GRAPH FACTS:
{facts_str}

QUESTION:
{question}

CRITICAL RULES:
1. Answer directly and immediately in the first sentence without meta-introductions (e.g., do not say "Based on the context" or "According to the study").
2. Address EVERY part of the question thoroughly (typically 3 to 6 sentences depending on question complexity).
3. Explicitly include specific clinical details present in the context: exact percentages, sample sizes, dollar amounts, assessment scale names (and abbreviations), anatomical tracts/regions, and which outcomes were statistically significant versus non-significant.
4. Prioritize the exact wording and findings in CONTEXT over SUPPORTING GRAPH FACTS if a graph fact looks truncated.
5. Do not include inline author citations or publication years (e.g., omit "(Smith et al., 2020)" or "[1]").

ANSWER:"""

    raw_answer, metrics = _call_ollama(prompt, system=ANSWER_SYSTEM)
    cleaned_answer = clean_llm_citations(raw_answer)
    return cleaned_answer, metrics