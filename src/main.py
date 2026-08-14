from graph_db import GraphDB
from ner import extract_entities         # fast local extraction, no LLM calls
from llm import generate_answer          # LLM still used for final answer synthesis
import time

def normalize(name: str) -> str:
    return " ".join(name.strip().lower().split())

def score_snippet(text: str, entities: list) -> int:
    """Count total term occurrences to score snippet relevance."""
    text_lower = text.lower()
    return sum(text_lower.count(e) for e in entities if len(e) > 2)

def answer_question(db: GraphDB, question: str) -> str:
    start_total_time = time.perf_counter()

    t0 = time.perf_counter()
    entities = extract_entities(question)
    entities = [normalize(e) for e in entities if e.strip()]
    ner_time = time.perf_counter() - t0
    print(f"  [entities extracted ({ner_time:.2f}s): {entities}]")

    if not entities:
        return "Couldn't identify relevant entities in the question.", {}

    t1 = time.perf_counter()
    matched = []
    for e in entities:
        matched.extend(db.search_entities(e))
    matched = list(set(matched))
    print(f"  [matched graph nodes: {matched}]")

    if not matched:
        return "No matching information found in the graph.", {}

    facts = db.traverse(matched)
    neo4j_time = time.perf_counter() - t1
    print(f"  [graph facts retrieved ({neo4j_time:.2f}s): {len(facts)} facts]")
    if not facts:
        return "No related facts found in the graph."

    # Extract source text snippets attached to retrieved relationships
    snippets = []
    for f in facts:
        if f.get("source_text") and f["source_text"] not in snippets:
            snippets.append(f["source_text"])

    # Prioritize snippets that explicitly contain matched specific entities
    prioritized_snippets = sorted(
        snippets, 
        key=lambda text: score_snippet(text, entities),  
        reverse=True
    )
    # Pass the top 8 most relevant chunk texts to the LLM
    answer, llm_metrics = generate_answer(question, facts, prioritized_snippets[:5])

    total_rag_time = time.perf_counter() - start_total_time
    llm_metrics["total_rag_latency_sec"] = round(total_rag_time, 2)
    llm_metrics["neo4j_time_sec"] = round(neo4j_time, 2)

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
                print(f"  • Generation Speed : {metrics['generation_speed_tok_sec']} tokens/sec")
                print(f"  • Prompt Eval Speed : {metrics['prompt_speed_tok_sec']} tokens/sec")
                print(f"  • Generated Tokens  : {metrics['generated_tokens']} tokens")
                print(f"  • Prompt Context    : {metrics['prompt_tokens']} tokens")
                print(f"  • LLM Latency       : {metrics['llm_total_sec']}s")
                print(f"  • Neo4j Latency     : {metrics['neo4j_time_sec']}s")
                print(f"  • Total RAG Latency : {metrics['total_rag_latency_sec']}s")
                print("=" * 50 + "\n")
    finally:
        db.close()


if __name__ == "__main__":
    main()
