from graph_db import GraphDB
from ner import extract_entities, is_generic_entity         # fast local extraction, no LLM calls
from llm import generate_answer          # LLM still used for final answer synthesis
import time
import re

def normalize(name: str) -> str:
    return " ".join(name.strip().lower().split())

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

def answer_question(db: GraphDB, question: str) -> tuple[str, dict]:
    start_total_time = time.perf_counter()

    start_ner = time.perf_counter()
    raw_entities = extract_entities(question)
    # Clean and normalize
    cleaned_entities = []
    for e in raw_entities:
        norm = normalize(re.sub(r'^(the|a|an)\s+', '', e.strip(), flags=re.IGNORECASE))
        if norm and norm not in cleaned_entities:
            cleaned_entities.append(norm)
    compound_entities = [e for e in cleaned_entities if len(e.split()) > 1]
    single_entities = [e for e in cleaned_entities if len(e.split()) == 1]
    # Use compound phrases first; fall back to non-generic singles
    search_pool = compound_entities + [
    e for e in single_entities if not any(e in c.split() for c in compound_entities)]
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
    }

    print(f"  [entities extracted ({ner_latency:.2f}s): {search_pool}]")

    if not search_pool:
        return "Couldn't identify relevant entities in the question.", default_metrics

    start_neo4j = time.perf_counter()
    matched = []
    for e in search_pool:
        matched.extend(db.search_entities(e, limit=3))
    matched = list(set(matched))
    print(f"  [matched graph nodes: {matched}]")

    if not matched:
        default_metrics["neo4j_latency_sec"] = round(time.perf_counter() - start_neo4j, 2)
        default_metrics["total_rag_latency_sec"] = round(time.perf_counter() - start_total_time, 2)
        return "No matching information found in the graph.", default_metrics

    direct_chunks = db.get_chunks_for_entities(matched, limit=10)
    facts = db.traverse(matched)
    neo4j_latency = time.perf_counter() - start_neo4j
    print(f"  [graph facts retrieved ({neo4j_latency:.2f}s): {len(facts)} facts]")
    if not facts:
        default_metrics["neo4j_latency_sec"] = round(neo4j_latency, 2)
        default_metrics["total_rag_latency_sec"] = round(time.perf_counter() - start_total_time, 2)
        return "No related facts found in the graph.", default_metrics

    combined = direct_chunks + [f["source_text"] for f in facts if f.get("source_text")]
    snippets = list(dict.fromkeys(combined))  # dedup, order-preserving

    scored = [(s, score_snippet(s, question, compound_entities)) for s in snippets]
    relevant = [(s, sc) for s, sc in scored if sc > 0]

    if relevant:
        prioritized_snippets = [s for s, _ in sorted(relevant, key=lambda pair: pair[1], reverse=True)]
    else:
        # No snippet had keyword overlap — fall back to the first candidate,
        # in original (not scored) order, rather than an arbitrary one.
        prioritized_snippets = snippets[:1]

    # Pass the top 8 most relevant chunk texts to the LLM
    start_llm = time.perf_counter()
    answer, llm_metrics = generate_answer(question, facts, prioritized_snippets[:8])
    llm_latency = time.perf_counter() - start_llm

    total_latency = time.perf_counter() - start_total_time
    llm_metrics["ner_latency_sec"] = round(ner_latency, 2)
    llm_metrics["neo4j_latency_sec"] = round(neo4j_latency, 2)
    llm_metrics["llm_latency_sec"] = round(llm_latency, 2)
    llm_metrics["total_rag_latency_sec"] = round(total_latency, 2)

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
