import torch
import numpy as np
import math
from tqdm import tqdm

from nltk.tokenize import word_tokenize
from nltk.util import ngrams

import evaluate

class Evaluator:
    def __init__(self, device=None):
        """
        Initializes models and metrics for evaluation.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Evaluator initialized on device: {self.device}")

        print("Loading BERTScore model...")
        self.bertscore = evaluate.load("bertscore")

    def calculate_distinct(self, sentences):
        
        if not sentences:
            return 0.0, 0.0

        total_words = 0
        total_bigrams = 0
        unique_words = set()
        unique_bigrams = set()

        for sent in tqdm(sentences, desc="Calculating Distinct Scores"):
            tokens = word_tokenize(sent.lower())
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
        results = self.bertscore.compute(
            predictions=hypotheses, 
            references=references, 
            lang="en",
            verbose=True,
            device=self.device
        )
        
        # Return the average scores
        return {
            "bs_precision": np.mean(results["precision"]),
            "bs_recall": np.mean(results["recall"]),
            "bs_f1": np.mean(results["f1"]),
        }

    def evaluate_all(self, hypotheses, references):
        
        results = {}
        
        dist_1, dist_2 = self.calculate_distinct(hypotheses)
        results['dist_1'] = dist_1
        results['dist_2'] = dist_2
        
        bertscore_results = self.calculate_bertscore(hypotheses, references)
        results.update(bertscore_results) 
        
        return results