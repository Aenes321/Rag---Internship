import os
from typing import List, Dict, Union, Tuple
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score_fn


class RAGEvaluator:
    """
    Computes:
    1. Sentence Transformer Cosine Similarity (Semantic Similarity)
    2. BERTScore (Precision, Recall, F1 via contextual embeddings)
    """

    def __init__(
        self,
        st_model_name: str = "all-MiniLM-L6-v2",
        bert_lang: str = "en",
        bert_model_type: str = None,
        device: str = None,
    ):
        """
        Initialize evaluation models.

        :param st_model_name: HuggingFace model name for SentenceTransformers.
        :param bert_lang: Language code for BERTScore (e.g., 'en').
        :param bert_model_type: Specific model for BERTScore (e.g., 'roberta-large').
        :param device: Device to run evaluation on ('cuda', 'cpu', or None for auto).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_lang = bert_lang
        self.bert_model_type = bert_model_type

        print(f"[Evaluator] Loading Sentence Transformer: {st_model_name} on {self.device}...")
        self.st_model = SentenceTransformer(st_model_name, device=self.device)

    def compute_cosine_similarity(
        self, references: List[str], generated: List[str]
    ) -> List[float]:
        """Compute pairwise cosine similarity using SentenceTransformer embeddings."""
        ref_embeddings = self.st_model.encode(
            references, convert_to_tensor=True, show_progress_bar=False
        )
        gen_embeddings = self.st_model.encode(
            generated, convert_to_tensor=True, show_progress_bar=False
        )

        # Compute element-wise pairwise cosine similarity
        cosine_sims = F.cosine_similarity(ref_embeddings, gen_embeddings, dim=1)
        return cosine_sims.cpu().tolist()

    def compute_bertscore(
        self, references: List[str], generated: List[str]
    ) -> Tuple[List[float], List[float], List[float]]:
        """Compute BERTScore (Precision, Recall, F1) for generated vs reference answers."""
        print("[Evaluator] Computing BERTScore...")
        P, R, F1 = bert_score_fn(
            cands=generated,
            refs=references,
            lang=self.bert_lang,
            model_type=self.bert_model_type,
            verbose=False,
            device=self.device,
        )
        return P.tolist(), R.tolist(), F1.tolist()

    def evaluate_batch(
        self,
        references: List[str],
        generated: List[str],
        questions: List[str] = None,
    ) -> pd.DataFrame:
        """
        Evaluate a batch of questions, reference answers, and generated RAG answers.

        :param references: Ground truth / reference answers.
        :param generated: RAG generated answers.
        :param questions: (Optional) List of question strings.
        :return: pandas DataFrame containing detailed per-item evaluation metrics.
        """
        if len(references) != len(generated):
            raise ValueError("References and Generated answers lists must have equal length.")

        # 1. Cosine Similarity
        cos_sims = self.compute_cosine_similarity(references, generated)

        # 2. BERTScore
        p_scores, r_scores, f1_scores = self.compute_bertscore(references, generated)

        # Build results DataFrame
        data = []
        for idx in range(len(references)):
            row = {
                "reference_answer": references[idx],
                "generated_answer": generated[idx],
                "st_cosine_sim": round(cos_sims[idx], 4),
                "bertscore_precision": round(p_scores[idx], 4),
                "bertscore_recall": round(r_scores[idx], 4),
                "bertscore_f1": round(f1_scores[idx], 4),
            }
            if questions and idx < len(questions):
                row = {"question": questions[idx], **row}
            data.append(row)

        df = pd.DataFrame(data)
        return df

    def summarize_results(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute average performance metrics across all test samples."""
        summary = {
            "mean_cosine_similarity": round(df["st_cosine_sim"].mean(), 4),
            "mean_bertscore_precision": round(df["bertscore_precision"].mean(), 4),
            "mean_bertscore_recall": round(df["bertscore_recall"].mean(), 4),
            "mean_bertscore_f1": round(df["bertscore_f1"].mean(), 4),
        }
        return summary