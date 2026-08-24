import os
import json
import time
import pandas as pd
import json
import requests
from evaluate import RAGEvaluator

#import answer_question uncomment this line if answer_question is defined in another module

# Modify the system prompt to instruct the LLM to answer questions using only the provided facts and source text.
ANSWER_SYSTEM = """You are a medical information assistant. Answer the user's
question using ONLY the facts and source text provided below."""

def _call_ollama(prompt: str, system: str = None, json_mode: bool = False) -> (str, dict):
    payload = {
        "model": "mistral", #change this if you want to use a different model
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system
    if json_mode:
        payload["format"] = "json"

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
        "prompt_tokens": prompt_eval_count,
        "generated_tokens": eval_count,
        "generation_speed_tok_sec": round(eval_count / eval_duration_sec, 2) if eval_duration_sec > 0 else 0,
        "prompt_speed_tok_sec": round(prompt_eval_count / prompt_eval_sec, 2) if prompt_eval_sec > 0 else 0,
        "llm_total_sec": round(total_duration_sec, 2),
    }

    return data["response"], metrics

def generate_answer(question: str, retrieved_docs) -> str:
    
    prompt = f"""Question: {question}

                ***CHANGE HERE***
***Your relevant knowledge base facts and snippets will be here***



Answer the question using only the information above."""
    return _call_ollama(prompt, system=ANSWER_SYSTEM)


def answer_question(question: str) -> (str, dict):
    start_total = time.perf_counter()

    # ------------------
    # Embedding
    # ------------------
    start = time.perf_counter()
    query_embedding = embed(question) #***CHANGE HERE*** Replace with your embedding function
    embed_latency = time.perf_counter() - start

    # ------------------
    # Retrieval
    # ------------------
    start = time.perf_counter()
    retrieved_docs = retrieve(query_embedding) #***CHANGE HERE*** Replace with your retrieval function
    retrieve_latency = time.perf_counter() - start

    # ------------------
    # LLM generation
    # ------------------
    start = time.perf_counter()
    answer, llm_metrics = generate_answer( 
        question,
        retrieved_docs
    )
    llm_latency = time.perf_counter() - start

    # ------------------
    # Total
    # ------------------
    total_latency = time.perf_counter() - start_total

    llm_metrics["total_rag_latency_sec"] = round(total_latency, 2)
    llm_metrics["embed_latency_sec"] = round(embed_latency, 2)
    llm_metrics["retrieve_latency_sec"] = round(retrieve_latency, 2)
    llm_metrics["llm_latency_sec"] = round(llm_latency, 2)

    return answer, llm_metrics

def load_input_file(file_path: str) -> pd.DataFrame:
    """Load test questions and reference answers from CSV or JSON."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found at: {file_path}")

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".json"):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format. Please use a .csv or .json file.")

    # Validate required columns
    required_cols = {"question", "reference_answer"}
    if not required_cols.issubset(df.columns):
        raise KeyError(
            f"Input file must contain the following columns: {required_cols}"
        )

    return df


def run_rag_and_evaluate(
    input_file: str,
    output_results_csv: str = "evaluation_results.csv",
    output_summary_json: str = "evaluation_summary.json",
):
    """Run RAG generation and evaluate against ground truth reference answers."""
    
    # 1. Load Input Test Cases
    print(f"\n[1/4] Loading test cases from '{input_file}'...")
    test_df = load_input_file(input_file)
    print(f"  Loaded {len(test_df)} test samples.")

    # 2. Generate RAG Responses
    print("\n[2/4] Generating answers from RAG...")

    generated_answers = []
    rag_latencies = []

    try:
        for idx, row in test_df.iterrows():
            q = row["question"]
            print(f"  Processing Question {idx + 1}/{len(test_df)}: {q[:60]}...")
            
            t0 = time.perf_counter()
            gen_ans, metrics = answer_question(q)
            if metrics:
                print("=" * 50)
                print("  PERFORMANCE METRICS:")
                print(f"  • Generation Speed : {metrics['generation_speed_tok_sec']} tokens/sec")
                print(f"  • Prompt Eval Speed : {metrics['prompt_speed_tok_sec']} tokens/sec")
                print(f"  • Generated Tokens  : {metrics['generated_tokens']} tokens")
                print(f"  • Prompt Context    : {metrics['prompt_tokens']} tokens")
                print(f"  • Embed Latency     : {metrics['embed_latency_sec']}s")
                print(f"  • Retrieve Latency : {metrics["retrieve_latency_sec"]}s")
                print(f"  • LLM Latency       : {metrics['llm_total_sec']}s")
                print(f"  • Total RAG Latency : {metrics['total_rag_latency_sec']}s")
                print("=" * 50 + "\n")
            latency = time.perf_counter() - t0

            generated_answers.append(gen_ans)
            rag_latencies.append(round(latency, 2))
    except Exception as e:
        print(f"[ERROR] Exception occurred during RAG generation: {e}")
        print("Exiting evaluation.")
        return
    
    test_df["generated_answer"] = generated_answers
    test_df["rag_latency_sec"] = rag_latencies

    # 3. Compute BERTScore & Cosine Similarity Metrics
    print("\n[3/4] Computing evaluation metrics (BERTScore & Cosine Similarity)...")
    evaluator = RAGEvaluator(st_model_name="all-MiniLM-L6-v2", bert_lang="en")

    eval_results_df = evaluator.evaluate_batch(
        references=test_df["reference_answer"].tolist(),
        generated=test_df["generated_answer"].tolist(),
        questions=test_df["question"].tolist(),
    )

    # Merge RAG Latency back into evaluation results
    eval_results_df["rag_latency_sec"] = test_df["rag_latency_sec"]

    # 4. Save Results to Output Files
    print(f"\n[4/4] Writing output results to files...")
    
    # Save detailed CSV
    eval_results_df.to_csv(output_results_csv, index=False)
    print(f"  • Detailed per-question results saved to: '{output_results_csv}'")

    # Compute & Save Summary Metrics
    summary_metrics = evaluator.summarize_results(eval_results_df)
    summary_metrics["mean_rag_latency_sec"] = round(eval_results_df["rag_latency_sec"].mean(), 2)
    summary_metrics["total_samples_evaluated"] = len(eval_results_df)

    with open(output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=4)
    print(f"  • Overall summary metrics saved to: '{output_summary_json}'")

    # Print Summary to Console
    print("\n" + "=" * 55)
    print("                EVALUATION SUMMARY")
    print("=" * 55)
    for k, v in summary_metrics.items():
        print(f"  • {k:<28}: {v}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    # Specify input file and output file paths
    INPUT_FILE = "../test_set.csv"
    OUTPUT_CSV = "evaluation_results.csv"
    OUTPUT_JSON = "evaluation_summary.json"

    run_rag_and_evaluate(
        input_file=INPUT_FILE,
        output_results_csv=OUTPUT_CSV,
        output_summary_json=OUTPUT_JSON,
    )