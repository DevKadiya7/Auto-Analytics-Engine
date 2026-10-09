import unittest

import pandas as pd

from src.trend_detection import detect_trends


class TrendTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "month": ["2026-08", "2026-07", "2026-08", "2026-07"],
                "district": ["Ahmedabad", "Ahmedabad", "Other", "Other"],
                "anc_coverage": [69, 85, 20, 0],
            }
        )

    def test_sorts_months_and_calculates_expected_change(self):
        result = detect_trends(self.data, threshold=10, indicators=["anc_coverage"])
        ahmedabad = result[result["district"] == "Ahmedabad"].iloc[0]
        self.assertAlmostEqual(ahmedabad["change_pct"], -18.823529, places=5)
        self.assertEqual(ahmedabad["previous_month"], "2026-07")
        self.assertEqual(ahmedabad["threshold"], 10)
        self.assertEqual(ahmedabad["direction"], "Decrease")
        self.assertTrue(ahmedabad["is_significant"])

    def test_zero_previous_value_does_not_raise(self):
        result = detect_trends(self.data, threshold=10, indicators=["anc_coverage"])
        other = result[result["district"] == "Other"].iloc[0]
        self.assertTrue(pd.isna(other["change_pct"]))
        self.assertFalse(other["is_significant"])

    def test_threshold_changes_significance(self):
        result = detect_trends(self.data, threshold=20, indicators=["anc_coverage"])
        ahmedabad = result[result["district"] == "Ahmedabad"].iloc[0]
        self.assertFalse(ahmedabad["is_significant"])


if __name__ == "__main__":
    unittest.main()
