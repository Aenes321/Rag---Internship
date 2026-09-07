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
    metrics_list = []

    try:
        for idx, row in test_df.iterrows():
            q = row["question"]
            print(f"  Processing Question {idx + 1}/{len(test_df)}: {q[:60]}...")
            
            gen_ans, metrics = answer_question(db, q)

            if metrics:
                print("=" * 50)
                print("  PERFORMANCE METRICS:")
                print(f"  • Generation Speed    : {metrics.get('generation_speed_tok_sec', 0)} tokens/sec")
                print(f"  • Prompt Eval Speed   : {metrics.get('prompt_speed_tok_sec', 0)} tokens/sec")
                print(f"  • Generated Tokens    : {metrics.get('generated_tokens', 0)} tokens")
                print(f"  • Prompt Context      : {metrics.get('prompt_tokens', 0)} tokens")
                print(f"  • NER Latency         : {metrics.get('ner_latency_sec', 0)}s")
                print(f"  • Neo4j Latency       : {metrics.get('neo4j_latency_sec', 0)}s")
                print(f"  • Evaluation Latency  : {metrics.get('prompt_eval_latency', 0)}s")
                print(f"  • Generation Latency  : {metrics.get('generation_eval_latency', 0)}s")
                print(f"  • Total LLM Latency   : {metrics.get('llm_latency_sec', 0)}s")
                print(f"  • Total RAG Latency    : {metrics.get('total_rag_latency_sec', 0)}s")
                print("=" * 50 + "\n")

            generated_answers.append(gen_ans)
            metrics_list.append(metrics)
    finally:
        db.close()

    test_df["generated_answer"] = generated_answers
    metrics_df = pd.DataFrame(metrics_list)
    for col in metrics_df.columns:
        test_df[col] = metrics_df[col]
    # 3. Compute BERTScore & Cosine Similarity Metrics
    print("\n[3/4] Computing evaluation metrics (BERTScore & Cosine Similarity)...")
    evaluator = RAGEvaluator(st_model_name="all-MiniLM-L6-v2", bert_lang="en")

    eval_results_df = evaluator.evaluate_batch(
        references=test_df["reference_answer"].tolist(),
        generated=test_df["generated_answer"].tolist(),
        questions=test_df["question"].tolist(),
    )

    # Attach all granular latency and token metrics to final results
    for col in metrics_df.columns:
        eval_results_df[col] = test_df[col]

    # 4. Save Results to Output Files
    print(f"\n[4/4] Writing output results to files...")
    
    # Save detailed CSV
    eval_results_df.to_csv(output_results_csv, index=False)
    print(f"  • Detailed per-question results saved to: '{output_results_csv}'")

    # Compute & Save Summary Metrics
    summary_metrics = evaluator.summarize_results(eval_results_df)
    summary_metrics["mean_generation_speed_tok_sec"] = round(eval_results_df["generation_speed_tok_sec"].mean(), 2)
    summary_metrics["mean_prompt_speed_tok_sec"] = round(eval_results_df["prompt_speed_tok_sec"].mean(), 2)
    summary_metrics["mean_ner_latency_sec"] = round(eval_results_df["ner_latency_sec"].mean(), 2)
    summary_metrics["mean_neo4j_latency_sec"] = round(eval_results_df["neo4j_latency_sec"].mean(), 2)
    summary_metrics["mean_llm_latency_sec"] = round(eval_results_df["llm_latency_sec"].mean(), 2)
    summary_metrics["mean_total_rag_latency_sec"] = round(eval_results_df["total_rag_latency_sec"].mean(), 2)
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