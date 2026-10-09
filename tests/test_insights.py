import unittest

import pandas as pd

from src.correlation_analysis import flagged_correlations
from src.insight_generator import INSIGHT_COLUMNS, add_threshold_breaches, generate_insights
from src.outlier_detection import detect_outliers
from src.trend_detection import detect_trends


class InsightTests(unittest.TestCase):
    def test_all_insight_fields_and_severity_values(self):
        data = pd.DataFrame(
            {
                "month": ["2026-07", "2026-08", "2026-07", "2026-08"],
                "district": ["A", "A", "B", "B"],
                "metric_one": [85, 69, 80, 81],
                "metric_two": [10, 13, 10, 11],
            }
        )
        trends = detect_trends(data, indicators=["metric_one", "metric_two"])
        outliers = detect_outliers(data, indicators=["metric_one", "metric_two"])
        correlations = flagged_correlations(data, threshold=0.7, min_observations=3)
        insights = generate_insights(trends, outliers, correlations)
        insights = add_threshold_breaches(insights, data, "metric_one", 70)
        self.assertEqual(list(insights.columns), INSIGHT_COLUMNS)
        self.assertTrue(set(insights["severity"]).issubset({"Low", "Medium", "High"}))
        self.assertTrue((insights["type"] == "threshold_breach").any())


if __name__ == "__main__":
    unittest.main()
