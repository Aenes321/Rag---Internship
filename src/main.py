from graph_db import GraphDB
from llm import extract_query_entities, generate_answer


def normalize(name: str) -> str:
    return " ".join(name.strip().lower().split())


def answer_question(db: GraphDB, question: str) -> str:
    entities = extract_query_entities(question)
    entities = [normalize(e) for e in entities if e.strip()]
    print(f"  [entities extracted: {entities}]")

    if not entities:
        return "Couldn't identify relevant entities in the question."

    matched = []
    for e in entities:
        matched.extend(db.search_entities(e))
    matched = list(set(matched))
    print(f"  [matched graph nodes: {matched}]")

    if not matched:
        return "No matching information found in the graph."

    facts = db.traverse(matched)
    if not facts:
        return "No related facts found in the graph."

    source_snippets = list({f["source_text"] for f in facts if f.get("source_text")})[:8]
    return generate_answer(question, facts, source_snippets)


def main():
    db = GraphDB()
    print("Graph RAG demo — ask a question about the spatial neglect (type 'exit' to quit)\n")

    try:
        while True:
            question = input("Q: ").strip()
            if question.lower() in ("exit", "quit"):
                break
            if not question:
                continue

            answer = answer_question(db, question)
            print(f"A: {answer}\n")
    finally:
        db.close()


if __name__ == "__main__":
    main()
