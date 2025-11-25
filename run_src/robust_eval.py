import numpy as np
from typing import List, Dict
import statistics

class RobustEvaluator:
    def __init__(self, io_system, args) -> None:
        self.io = io_system
        self.args = args
        self.sample_size = getattr(args, 'robust_sample_size', 3)
        self.aggregation_method = getattr(args, 'robust_aggregation', 'median') # median, mean, trimmed_mean

    def verify_support(self, question: str, answer: str, context_chunk: str) -> float:
        """
        Verifies if the context supports the answer to the question.
        Returns a score between 0.0 and 1.0.
        """
        # Prompt engineering for verification
        prompt = (
            "You are a strict judge evaluating whether a retrieved document supports a proposed answer.\n"
            f"Question: {question}\n"
            f"Proposed Answer: {answer}\n"
            f"Retrieved Document: {context_chunk}\n\n"
            "Does the Retrieved Document provide sufficient evidence to support the Proposed Answer?\n"
            "Respond with a score from 0.0 (No support/Contradicts) to 1.0 (Full support).\n"
            "Output ONLY the numeric score."
        )
        
        try:
            # We use a small number of tokens as we only expect a number
            response_list = self.io.generate(
                model_input=prompt,
                max_tokens=10,
                num_return=1,
                stop_tokens=["\n"]
            )
            response = response_list[0].strip()
            score = float(response)
            return max(0.0, min(1.0, score)) # Clip to [0, 1]
        except Exception as e:
            print(f"Error in verify_support: {e}")
            return 0.0 # Default to no support on error

    def robust_score(self, question: str, answer: str, retrieved_docs: List[str]) -> float:
        """
        Calculates a robust score for the answer based on retrieved documents.
        Uses Isolate-then-Aggregate strategy.
        """
        if not retrieved_docs:
            return 0.0

        # 1. Isolate: Verify against each doc independently
        # Limit to sample_size to save compute
        docs_to_check = retrieved_docs[:self.sample_size]
        scores = []
        
        for doc in docs_to_check:
            score = self.verify_support(question, answer, doc)
            scores.append(score)

        # 2. Aggregate: Use robust statistics
        if not scores:
            return 0.0
            
        if self.aggregation_method == 'median':
            final_score = statistics.median(scores)
        elif self.aggregation_method == 'mean':
            final_score = statistics.mean(scores)
        elif self.aggregation_method == 'trimmed_mean':
            # Simple trimmed mean: remove min and max if len > 2
            if len(scores) > 2:
                scores.remove(min(scores))
                scores.remove(max(scores))
            final_score = statistics.mean(scores)
        else:
            final_score = statistics.median(scores)

        return final_score
