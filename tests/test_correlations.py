import unittest

import pandas as pd

from src.correlation_analysis import correlation_matrix, flagged_correlations


class CorrelationTests(unittest.TestCase):
    def test_matrix_and_unique_flagged_pairs(self):
        data = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [4, 3, 2, 1]})
        matrix = correlation_matrix(data)
        result = flagged_correlations(data, threshold=0.7, min_observations=3)
        self.assertEqual(matrix.loc["a", "b"], 1.0)
        self.assertEqual(len(result), 3)
        self.assertIn("correlation_coefficient", result.columns)
        self.assertIn("interpretation", result.columns)
        self.assertEqual(len({tuple(sorted(pair)) for pair in zip(result.indicator_a, result.indicator_b)}), len(result))

    def test_constant_and_insufficient_pairs_are_ignored(self):
        data = pd.DataFrame({"a": [1, 1], "b": [2, 3]})
        self.assertTrue(flagged_correlations(data, min_observations=3).empty)

    def test_correlation_threshold_changes_flagged_pairs(self):
        data = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [4, 3, 2, 1]})
        self.assertEqual(len(flagged_correlations(data, threshold=0.99)), 3)
        self.assertEqual(len(flagged_correlations(data, threshold=1.01)), 0)


if __name__ == "__main__":
    unittest.main()
