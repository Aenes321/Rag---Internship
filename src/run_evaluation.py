import os
import json
import time
import pandas as pd
from graph_db import GraphDB
from main import answer_question
from evaluate import RAGEvaluator  # Imports evaluator from previous step


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

    # 2. Connect to GraphDB & Generate RAG Responses
    print("\n[2/4] Generating answers from Graph RAG...")
    db = GraphDB()

    generated_answers = []
    rag_latencies = []

    try:
        for idx, row in test_df.iterrows():
            q = row["question"]
            print(f"  Processing Question {idx + 1}/{len(test_df)}: {q[:60]}...")
            
            t0 = time.perf_counter()
            gen_ans, metrics = answer_question(db, q)
            latency = time.perf_counter() - t0

            generated_answers.append(gen_ans)
            rag_latencies.append(round(latency, 2))
    finally:
        db.close()

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