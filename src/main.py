from graph_db import GraphDB
from ner import extract_entities         # fast local extraction, no LLM calls
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
        if len(e.split()) > 1 and e in text_lower:
            score += 5.0
        elif e in text_lower:
            score += 1.0

    # Reward chunks containing numerical or monetary figures relevant to quantitative questions
    if any(char.isdigit() for char in query):
        digits_in_snippet = len(re.findall(r'\b\d+(?:\.\d+)?\b', text))
        score += min(digits_in_snippet * 0.5, 3.0)

    return score

def answer_question(db: GraphDB, question: str) -> (str, dict):
    start_total_time = time.perf_counter()

    t0 = time.perf_counter()
    raw_entities = extract_entities(question)

    # Clean and normalize
    cleaned_entities = []
    for e in raw_entities:
        norm = normalize(re.sub(r'^(the|a|an)\s+', '', e.strip(), flags=re.IGNORECASE))
        if norm and norm not in cleaned_entities:
            cleaned_entities.append(norm)

    compound_entities = [e for e in cleaned_entities if len(e.split()) > 1]
    
    # Use compound phrases first; fall back to non-generic singles
    search_pool = compound_entities if compound_entities else cleaned_entities

    ner_time = time.perf_counter() - t0
    print(f"  [entities extracted ({ner_time:.2f}s): {search_pool}]")

    if not search_pool:
        return "Couldn't identify relevant entities in the question.", {}

    t1 = time.perf_counter()
    matched = []
    for e in search_pool:
        matched.extend(db.search_entities(e, limit=3))
    matched = list(set(matched))
    print(f"  [matched graph nodes: {matched}]")

    if not matched:
        return "No matching information found in the graph.", {}

    direct_chunks = db.get_chunks_for_entities(matched, limit=10)
    facts = db.traverse(matched)
    neo4j_time = time.perf_counter() - t1
    print(f"  [graph facts retrieved ({neo4j_time:.2f}s): {len(facts)} facts]")
    if not facts:
        return "No related facts found in the graph."

    snippets = list(set(direct_chunks + [f["source_text"] for f in facts if f.get("source_text")]))
    
    # Prioritize snippets that explicitly contain matched specific entities
    prioritized_snippets = sorted(
        snippets, 
        key=lambda text: score_snippet(text, question, compound_entities),  
        reverse=True
    )
    # Pass the top 8 most relevant chunk texts to the LLM
    answer, llm_metrics = generate_answer(question, facts, prioritized_snippets[:8])

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
