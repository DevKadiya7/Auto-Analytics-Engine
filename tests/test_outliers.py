import unittest

import pandas as pd

from src.data_loader import prepare_analysis_data
from src.outlier_detection import detect_outliers


class OutlierTests(unittest.TestCase):
    def test_mehsana_anc_is_detected_by_iqr(self):
        data = prepare_analysis_data(pd.read_csv("data/healthcare_performance.csv"))
        result = detect_outliers(data, method="IQR", indicators=["anc_coverage"])
        mehsana = result[(result["district"] == "Mehsana") & (result["value"] == 42)]
        self.assertEqual(len(mehsana), 1)
        self.assertEqual(mehsana.iloc[0]["method"], "IQR")

    def test_constant_column_has_no_zscore_outliers(self):
        data = pd.DataFrame({"district": ["A", "B"], "month": ["2026-01", "2026-02"], "metric": [5, 5]})
        self.assertTrue(detect_outliers(data, method="Z-score", indicators=["metric"]).empty)


if __name__ == "__main__":
    unittest.main()
