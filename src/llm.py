import json
import requests
import config


def _call_ollama(prompt: str, system: str = None, json_mode: bool = False) -> str:
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
    return response.json()["response"]


EXTRACTION_SYSTEM = """You are a medical information extraction system.
Extract factual relationships from the given text as a JSON list of triples.
Each triple must have: "subject", "relation", "object".
Use ONLY these relation types: TREATS, CAUSES, HAS_SYMPTOM, RISK_FACTOR_FOR,
INTERACTS_WITH, DIAGNOSED_BY, COMPLICATION_OF, PREVENTS, ASSOCIATED_WITH.
Keep subject/object names short (a few words), e.g. "metformin", "type 2 diabetes".
Return ONLY valid JSON in this exact format:
{"triples": [{"subject": "...", "relation": "...", "object": "..."}]}
If no relevant facts are found, return {"triples": []}.
Do not include any text outside the JSON."""


def extract_triples(chunk_text: str):
    prompt = f"Text:\n{chunk_text}\n\nExtract triples as JSON."
    raw = _call_ollama(prompt, system=EXTRACTION_SYSTEM, json_mode=True)
    try:
        data = json.loads(raw)
        return data.get("triples", [])
    except (json.JSONDecodeError, AttributeError):
        return []


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


ANSWER_SYSTEM = """You are a medical information assistant. Answer the user's
question using ONLY the facts and source text provided below. If the
information isn't sufficient to answer, say so plainly. Be clear, concise,
and do not invent information that isn't supported by the provided context."""


def generate_answer(question: str, facts: list, source_snippets: list) -> str:
    facts_str = "\n".join(f"- {f['source']} {f['relation']} {f['target']}" for f in facts) or "(none)"
    sources_str = "\n---\n".join(source_snippets) or "(none)"
    prompt = f"""Question: {question}

Graph facts:
{facts_str}

Source text excerpts:
{sources_str}

Answer the question using only the information above."""
    return _call_ollama(prompt, system=ANSWER_SYSTEM)
