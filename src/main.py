from graph_db import GraphDB
from ner import extract_entities, is_generic_entity, normalize_entity  # fast local extraction, no LLM calls
from llm import generate_answer          # LLM still used for final answer synthesis
import time
import re
from rank_bm25 import BM25Okapi

# Same normalization as ingest.py, so e.g. "VR" in a question maps to the
# "virtual reality" nodes created at ingestion time.
normalize = normalize_entity


def _is_acronym(entity: str, question: str) -> bool:
    """True if the entity appears in the question with acronym-style casing
    (e.g. "USN", "fNIRS"). Entities are lowercased by extraction, so the
    original casing has to be recovered from the question text."""
    m = re.search(r"\b" + re.escape(entity) + r"\b", question, re.IGNORECASE)
    return bool(m) and sum(c.isupper() for c in m.group(0)) >= 2

def score_snippet(text: str, query: str, entities: list) -> int:
    text_lower = text.lower()
    score = 0.0
    
    # Reward compound entity matches heavily
    for e in entities:
        e_lower = e.lower().strip()
        if len(e_lower) <= 2:
            continue

        count = len(re.findall(r'\b' + re.escape(e_lower) + r'\b', text_lower))
        if count == 0:
            continue
        count = min(count, 3)

        words = e_lower.split()
        if len(words) > 1:
            weight = 5
        elif is_generic_entity(e_lower):
            weight = 1          # generic single word — low signal
        else:
            weight = 3          # specific single word (e.g. "saccades", "usn") — closer to compound value

        score += count * weight

    if any(char.isdigit() for char in query):
        digits_in_snippet = len(re.findall(r'\b\d+(?:\.\d+)?\b', text))
        score += min(digits_in_snippet * 0.5, 3.0)

    return score

# Only filter true function/question words in BM25 — NEVER filter domain words like
# "meta-analysis", "randomized", "assessment", or "systematic"!
BM25_STOPWORDS = {
    "what", "which", "how", "why", "when", "where", "who", "does", "did", "the",
    "and", "are", "was", "were", "for", "with", "from", "this", "that", "these",
    "those", "their", "its", "into", "about", "based", "between", "can", "has",
    "have", "been", "being", "there", "they", "such", "also", "than", "then",
    "according", "specific", "particularly",
}

def clean_query_for_bm25(query: str) -> list[str]:
    tokens = re.findall(r"\w+", query.lower())
    clean_tokens = [t for t in tokens if t not in BM25_STOPWORDS and len(t) > 2]
    return clean_tokens if clean_tokens else tokens

def rank_snippets_bm25(snippets: list[str], query: str, top_k: int = 7) -> list[str]:
    if not snippets:
        return []
    
    tokenized_corpus = [re.findall(r"\w+", s.lower()) for s in snippets]
    tokenized_query = clean_query_for_bm25(query)

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    ranked_pairs = sorted(zip(scores, snippets), key=lambda x: x[0], reverse=True)
    return [snippet for score, snippet in ranked_pairs[:top_k] if score > 0] or snippets[:top_k]


# Weight of the supporting chunk text relative to the triple itself. The chunk
# is only a tie-breaker: when it dominated, every fact from the same chunk got
# an identical score and the ranking said nothing about the facts themselves.
FACT_SOURCE_TEXT_WEIGHT = 0.2

def rank_facts_bm25(facts: list[dict], query: str, top_k: int = 15) -> list[dict]:
    """Rank retrieved graph facts using BM25 against the question text."""
    if not facts:
        return []

    tokenized_query = clean_query_for_bm25(query)
    if not tokenized_query:
        return facts[:top_k]

    triple_corpus = [
        re.findall(r"\w+", f"{f.get('source', '')} {f.get('relation', '').replace('_', ' ')} {f.get('target', '')}".lower())
        for f in facts
    ]
    text_corpus = [re.findall(r"\w+", (f.get("source_text") or "").lower()) for f in facts]

    triple_scores = BM25Okapi(triple_corpus).get_scores(tokenized_query)
    if any(text_corpus):
        text_scores = BM25Okapi(text_corpus).get_scores(tokenized_query)
    else:
        text_scores = [0.0] * len(facts)

    combined = [t + FACT_SOURCE_TEXT_WEIGHT * s for t, s in zip(triple_scores, text_scores)]
    ranked_pairs = sorted(zip(combined, facts), key=lambda x: x[0], reverse=True)
    return [fact for score, fact in ranked_pairs[:top_k]]

def answer_question(db: GraphDB, question: str) -> tuple[str, dict]:
    start_total_time = time.perf_counter()

    start_ner = time.perf_counter()
    raw_entities = extract_entities(question)
    cleaned_entities = []
    for e in raw_entities:
        norm = normalize(re.sub(r'^(the|a|an|in|on|at|of|for|with|by)\s+', '', e.strip(), flags=re.IGNORECASE))
        if norm and not is_generic_entity(norm) and norm not in cleaned_entities:
            cleaned_entities.append(norm)

    compound_entities = [e for e in cleaned_entities if len(e.split()) > 1]
    single_entities = [e for e in cleaned_entities if len(e.split()) == 1]

    # Keep all compound entities AND all single entities (do not drop single entities
    # that appear inside compound entities, as the compound span may be too specific for Neo4j)
    search_pool = list(dict.fromkeys(compound_entities + single_entities))
    ner_latency = time.perf_counter() - start_ner

    default_metrics = {
        "prompt_eval_latency": 0.0,
        "generation_eval_latency": 0.0,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "generation_speed_tok_sec": 0.0,
        "prompt_speed_tok_sec": 0.0,
        "ner_latency_sec": round(ner_latency, 2),
        "neo4j_latency_sec": 0.0,
        "llm_latency_sec": 0.0,
        "total_rag_latency_sec": round(time.perf_counter() - start_total_time, 2),
        "retrieved_facts": [],
        "retrieved_snippets": [],
    }

    print(f"  [entities extracted ({ner_latency:.2f}s): {search_pool}]")
    if not search_pool:
        return "Couldn't identify relevant entities in the question.", default_metrics

    start_neo4j = time.perf_counter()
    matched_scores: dict[str, float] = {}
    seed_nodes: set[str] = set()

    for e in search_pool:
        results = db.search_entities(e, limit=3)
        if not results:
            continue
        best_score = max(score for _, score in results)
        for name, score in results:
            if score < best_score * 0.5:
                continue
            if name not in matched_scores or score > matched_scores[name]:
                matched_scores[name] = score

            # Include multi-word entities, acronyms (e.g., fNIRS), nouns >= 4 letters, or high Lucene matches
            if len(e.split()) > 1 or _is_acronym(e, question) or len(e) >= 4 or score >= 4.0:
                seed_nodes.add(name)

    print(f"  [matched graph nodes: {sorted(matched_scores.items(), key=lambda p: p[1], reverse=True)}]")

    if not matched_scores:
        default_metrics["neo4j_latency_sec"] = round(time.perf_counter() - start_neo4j, 2)
        default_metrics["total_rag_latency_sec"] = round(time.perf_counter() - start_total_time, 2)
        return "No matching information found in the graph.", default_metrics

    # Fallback to top-scoring node if none met seed thresholds
    if not seed_nodes:
        seed_nodes = {max(matched_scores, key=matched_scores.get)}

    seed_list = list(seed_nodes)
    print(f"  [traversal seed nodes: {seed_list}]")

    # Retrieve candidate chunks using IDF-weighted graph queries + fulltext chunk index
    cooccurring_chunks = db.get_cooccurring_chunks(seed_list, limit=25)
    direct_chunks = db.get_chunks_for_entities(seed_list, limit=30)
    fulltext_chunks = db.search_chunks_fulltext(question, limit=15)

    # Interconnected Subgraph Path Traversal
    facts = db.traverse_subgraph(seed_list, limit=30)
    neo4j_latency = time.perf_counter() - start_neo4j
    print(f"  [graph facts retrieved ({neo4j_latency:.2f}s): {len(facts)} facts]")

    fact_chunks = [f["source_text"] for f in facts if f.get("source_text")]
    combined_snippets = list(dict.fromkeys(
        cooccurring_chunks + direct_chunks + fulltext_chunks + fact_chunks
    ))

    if not facts and not combined_snippets:
        default_metrics["neo4j_latency_sec"] = round(neo4j_latency, 2)
        default_metrics["total_rag_latency_sec"] = round(time.perf_counter() - start_total_time, 2)
        return "No related information found in the graph.", default_metrics

    # Lexical BM25 Ranking over the full candidate pool
    prioritized_snippets = rank_snippets_bm25(combined_snippets, question, top_k=8)
    prioritized_facts = rank_facts_bm25(facts, question, top_k=15)

    # LLM Answer Generation
    start_llm = time.perf_counter()
    answer, llm_metrics = generate_answer(question, prioritized_facts, prioritized_snippets)
    llm_latency = time.perf_counter() - start_llm

    total_latency = time.perf_counter() - start_total_time
    llm_metrics["ner_latency_sec"] = round(ner_latency, 2)
    llm_metrics["neo4j_latency_sec"] = round(neo4j_latency, 2)
    llm_metrics["llm_latency_sec"] = round(llm_latency, 2)
    llm_metrics["total_rag_latency_sec"] = round(total_latency, 2)

    llm_metrics["retrieved_facts"] = [
        f"{f['source']} -> {f['relation']} -> {f['target']}" for f in prioritized_facts
    ]
    llm_metrics["retrieved_snippets"] = prioritized_snippets

    return answer, llm_metrics


def main():
    db = GraphDB()
    print("Graph RAG demo — ask a question about the disease (type 'exit' to quit)\n")

    try:
        while True:
            question = input("Q: ").strip()
            if question.lower() in ("exit", "quit"):
                break
            if not question:
                continue

            answer, metrics = answer_question(db, question)
            print(f"A: {answer}\n")

            if metrics:
                print("=" * 50)
                print("  PERFORMANCE METRICS:")
                print(f"  • NER Latency       : {metrics['ner_latency_sec']}s")
                print(f"  • Neo4j Latency     : {metrics['neo4j_latency_sec']}s")
                print(f"  • Prompt Evaluation Latency : {metrics['prompt_eval_latency']} s")
                print(f"  • LLM Generation Latency : {metrics['generation_eval_latency']} s")
                print(f"  • LLM Latency       : {metrics['llm_latency_sec']}s")
                print(f"  • Prompt Context    : {metrics['prompt_tokens']} tokens")
                print(f"  • Prompt Eval Speed : {metrics['prompt_speed_tok_sec']} tokens/sec")
                print(f"  • Generated Tokens  : {metrics['generated_tokens']} tokens")
                print(f"  • Generation Speed : {metrics['generation_speed_tok_sec']} tokens/sec")
                print(f"  • Total RAG Latency : {metrics['total_rag_latency_sec']}s")
                print("=" * 50 + "\n")
    finally:
        db.close()


if __name__ == "__main__":
    main()
