import unittest
from io import StringIO

import pandas as pd

from src.data_loader import clean_data, load_csv, prepare_analysis_data, validate_dataset


class DataLoaderTests(unittest.TestCase):
    def test_empty_and_malformed_sources_are_reported(self):
        data, error = load_csv(StringIO(""))
        self.assertIsNone(data)
        self.assertIn("empty", error.lower())
        data, error = load_csv(StringIO('month,district\n"unterminated,A\n'))
        self.assertIsNone(data)
        self.assertTrue(error)

    def test_missing_columns_invalid_month_and_invalid_numeric_are_reported(self):
        data = pd.DataFrame({"month": ["not-a-month"], "district": ["A"], "anc_coverage": ["bad"]})
        validation = validate_dataset(data)
        self.assertFalse(validation["valid"])
        self.assertTrue(validation["missing_columns"])
        self.assertEqual(validation["invalid_month_count"], 1)
        self.assertIn("anc_coverage", validation["non_numeric_indicators"])

    def test_duplicate_keys_and_out_of_range_values_are_reported(self):
        data = pd.DataFrame(
            {
                "month": ["2026-01", "2026-01"],
                "district": ["A", "A"],
                "anc_coverage": [101, 50],
                "institutional_delivery": [80, 80],
                "immunization": [90, 90],
                "high_risk_cases": [-1, 2],
            }
        )
        validation = validate_dataset(data)
        self.assertFalse(validation["valid"])
        self.assertEqual(validation["duplicate_key_count"], 1)
        self.assertEqual(validation["invalid_value_counts"]["anc_coverage"], 1)
        self.assertEqual(validation["invalid_value_counts"]["high_risk_cases"], 1)

    def test_cleaning_does_not_mutate_original_and_handles_duplicates(self):
        original = pd.DataFrame({"month": ["2026-01", "2026-01"], "district": ["A", "A"], "metric": [None, None]})
        snapshot = original.copy(deep=True)
        cleaned, report = clean_data(original)
        self.assertTrue(original.equals(snapshot))
        self.assertEqual(report["duplicates_removed"], 1)
        self.assertFalse(cleaned.isna().any().any())
        self.assertEqual(len(prepare_analysis_data(cleaned)), 1)


if __name__ == "__main__":
    unittest.main()
