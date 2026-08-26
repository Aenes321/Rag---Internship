import json
import requests
import config

def _call_ollama(prompt: str, system: str = None, json_mode: bool = False) -> tuple[str, dict]:
    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 300,   # Stop after 300 tokens (prevents endless generation)
        }
    }
    if system:
        payload["system"] = system
    if json_mode:
        payload["format"] = "json"
    timeout_limit = 180  # seconds
    try :
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
            "prompt_eval_latency": round(prompt_eval_sec, 2),
            "prompt_tokens": prompt_eval_count,
            "prompt_speed_tok_sec": round(prompt_eval_count / prompt_eval_sec, 2) if prompt_eval_sec > 0 else 0,
            "generation_eval_latency": round(eval_duration_sec, 2),
            "generated_tokens": eval_count,
            "generation_speed_tok_sec": round(eval_count / eval_duration_sec, 2) if eval_duration_sec > 0 else 0,
            
            "llm_latency_sec": round(total_duration_sec, 2),
        }

        return data["response"], metrics
    
    except requests.exceptions.Timeout:
        print("  [WARNING] Ollama request timed out.")
        return "Error: LLM response timed out.", {
            "prompt_tokens": 0, "generated_tokens": 0,
            "generation_speed_tok_sec": 0.0, "prompt_speed_tok_sec": 0.0,
            "llm_latency_sec": timeout_limit, "prompt_eval_latency": 0.0, "generation_eval_latency": 0.0
        }


ENTITY_SYSTEM = """You are an expert medical and neurorehabilitation entity extractor.
Extract all key concepts, medical entities, physiological measures, technologies, and study citations mentioned in the user's question.
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

ANSWER_SYSTEM = """You are an expert medical information synthesizer. Your task is to provide accurate, direct, and complete answers based strictly on the provided context and graph facts."""

def generate_answer(question: str, facts: list, source_snippets: list) -> str:
    facts_str = "\n".join(f"- {f['source']} {humanize_relation(f['relation'])} {f['target']}" for f in facts) or "(none)"
    sources_str = "\n---\n".join(source_snippets) or "(none)"
    prompt = f"""CONTEXT:
{sources_str}

Graph facts:
{facts_str}

Question: 
{question}

CORE REASONING & SYNTHESIS PRINCIPLES:
1. Direct Answering: Begin answering the user's specific question immediately in the first sentence. Never use filler introductions (e.g., "Spatial neglect and X are related...", "According to the provided text...", "In this study...").
2. Answer directly and concisely in 1 single paragraph (3 to 4 sentences max). Do not repeat yourself.
3. Comprehensive Multi-Part Coverage: If the user asks about multiple aspects (e.g., comparisons, mechanisms, metrics, or relationships), address each component thoroughly and accurately.
4. Faithful Contrasts & Nuances: Only contrast two concepts if the context explicitly states that contrast. Do not invent asymmetrical claims (e.g., do not write "Unlike X, Y does..." unless the text explicitly states X does not).
5. Strict Topic Relevance: Include ONLY facts and details that directly answer the query. Ignore background information, assessment scales, anatomical tracts, or unrelated technologies present in the context that were not asked about.
6. Natural, Clean Medical Prose: 
   - Write in fluid, professional clinical English.
   - Do not include robotic meta-language (e.g., avoid "The context states", "The provided sources do not mention").
   - Never output raw uppercase relationship tags (e.g., do not write "TREATS", "AFFECTS_REGION").
7. Do not add boilerplate conclusions (do not write "further research is needed").
    
DIRECT ANSWER:"""
    
    return _call_ollama(prompt, system=ANSWER_SYSTEM)