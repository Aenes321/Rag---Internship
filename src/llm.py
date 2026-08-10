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
