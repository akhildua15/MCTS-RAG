import unittest
from unittest.mock import MagicMock
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from run_src.robust_eval import RobustEvaluator

class TestRobustEvaluator(unittest.TestCase):
    def setUp(self):
        self.mock_io = MagicMock()
        self.mock_args = MagicMock()
        self.mock_args.robust_sample_size = 3
        self.mock_args.robust_aggregation = 'median'
        
        self.evaluator = RobustEvaluator(self.mock_io, self.mock_args)

    def test_verify_support(self):
        # Mock LLM response
        self.mock_io.generate.return_value = ["0.9"]
        
        score = self.evaluator.verify_support("Q", "A", "Doc")
        self.assertEqual(score, 0.9)
        
        # Test clipping
        self.mock_io.generate.return_value = ["1.5"]
        score = self.evaluator.verify_support("Q", "A", "Doc")
        self.assertEqual(score, 1.0)
        
        self.mock_io.generate.return_value = ["-0.5"]
        score = self.evaluator.verify_support("Q", "A", "Doc")
        self.assertEqual(score, 0.0)

    def test_robust_score_median(self):
        # Mock verify_support to return specific scores for specific docs
        # We'll just mock verify_support directly for simplicity
        self.evaluator.verify_support = MagicMock(side_effect=[0.1, 0.9, 0.8])
        
        docs = ["Doc1", "Doc2", "Doc3"]
        score = self.evaluator.robust_score("Q", "A", docs)
        
        # Median of [0.1, 0.9, 0.8] is 0.8
        self.assertEqual(score, 0.8)
        self.assertEqual(self.evaluator.verify_support.call_count, 3)

    def test_robust_score_mean(self):
        self.evaluator.aggregation_method = 'mean'
        self.evaluator.verify_support = MagicMock(side_effect=[0.1, 0.9, 0.8])
        
        docs = ["Doc1", "Doc2", "Doc3"]
        score = self.evaluator.robust_score("Q", "A", docs)
        
        # Mean of [0.1, 0.9, 0.8] is 0.6
        self.assertAlmostEqual(score, 0.6)

    def test_robust_score_trimmed_mean(self):
        self.evaluator.aggregation_method = 'trimmed_mean'
        self.evaluator.verify_support = MagicMock(side_effect=[0.1, 0.9, 0.8, 0.5])
        self.evaluator.sample_size = 4
        
        docs = ["Doc1", "Doc2", "Doc3", "Doc4"]
        score = self.evaluator.robust_score("Q", "A", docs)
        
        # Trimmed mean of [0.1, 0.5, 0.8, 0.9] -> remove 0.1 and 0.9 -> mean of [0.5, 0.8] = 0.65
        self.assertAlmostEqual(score, 0.65)

if __name__ == '__main__':
    unittest.main()
