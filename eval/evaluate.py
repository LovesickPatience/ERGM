import re

import torch
import numpy as np
import math
from tqdm import tqdm

from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from typing import Optional

# 尝试多种 BERTScore 后端
try:
    import evaluate
except Exception:
    evaluate = None

try:
    from bert_score import BERTScorer
except Exception:
    BERTScorer = None


class Evaluator:
    def __init__(
        self,
        device: Optional[str] = None,
        bert_model_path: Optional[str] = None,
        baseline_path: Optional[str] = None,
        num_layers: Optional[int] = None,
        rescale_with_baseline: bool = True,
    ):
        """
        Initializes models and metrics for evaluation.
        `bert_model_path` 支持传入本地 roberta-large 路径或 huggingface 名称。
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Evaluator initialized on device: {self.device}")

        self.bertscore_eval = None
        self.bertscore_scorer = None

        # 优先使用本地 BERTScorer（适合离线 roberta-large）
        if BERTScorer is not None:
            try:
                print("Loading BERTScore via bert_score.BERTScorer...")
                self.bertscore_scorer = BERTScorer(
                    model_type=bert_model_path or "roberta-large",
                    rescale_with_baseline=rescale_with_baseline,
                    device=self.device,
                    num_layers=num_layers,
                    lang="en",
                    baseline_path=baseline_path,
                )
            except Exception as e:
                print(f"Failed to init BERTScorer, will try evaluate backend: {e}")

        if self.bertscore_scorer is None and evaluate is not None:
            try:
                print("Loading BERTScore via evaluate...")
                self.bertscore_eval = evaluate.load("bertscore")
            except Exception as e:
                print(f"Failed to load evaluate bertscore: {e}")

        if self.bertscore_scorer is None and self.bertscore_eval is None:
            raise RuntimeError("BERTScore backend is not available (bert_score or evaluate).")

    def calculate_distinct(self, sentences):
        
        if not sentences:
            return 0.0, 0.0

        total_words = 0
        total_bigrams = 0
        unique_words = set()
        unique_bigrams = set()

        def simple_tokenize(text):
            if text is None:
                text = ""
            text = str(text)
            return re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)

        for sent in tqdm(sentences, desc="Calculating Distinct Scores"):
            # tokens = word_tokenize(sent.lower())
            tokens = simple_tokenize(sent)
            total_words += len(tokens)
            unique_words.update(tokens)
            
            bigrams = list(ngrams(tokens, 2))
            total_bigrams += len(bigrams)
            unique_bigrams.update(bigrams)

        dist_1 = len(unique_words) / total_words if total_words > 0 else 0.0
        dist_2 = len(unique_bigrams) / total_bigrams if total_bigrams > 0 else 0.0
        
        return dist_1, dist_2

    def calculate_bertscore(self, hypotheses, references):
        
        if not hypotheses or not references:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            
        print("Calculating BERTScore...")
        if self.bertscore_scorer is not None:
            # 使用本地 BERTScorer
            P, R, F1 = self.bertscore_scorer.score(
                cands=hypotheses,
                refs=references,
                verbose=True,
            )
            return {
                "precision": P.mean().item(),
                "recall": R.mean().item(),
                "f1": F1.mean().item(),
            }
        elif self.bertscore_eval is not None:
            results = self.bertscore_eval.compute(
                predictions=hypotheses, 
                references=references, 
                lang="en",
                verbose=True,
                device=self.device
            )
            return {
                "precision": np.mean(results["precision"]),
                "recall": np.mean(results["recall"]),
                "f1": np.mean(results["f1"]),
            }
        else:
            raise RuntimeError("BERTScore backend is not initialized.")

    def evaluate_all(self, hypotheses, references):
        
        results = {}
        
        dist_1, dist_2 = self.calculate_distinct(hypotheses)
        print("Dist_1: ", dist_1)
        print("Dist_2: ", dist_2)
        results['dist_1'] = dist_1
        results['dist_2'] = dist_2
        
        bertscore_results = self.calculate_bertscore(hypotheses, references)
        print("BERTScore: ", bertscore_results)
        results.update(bertscore_results) 
        
        return results