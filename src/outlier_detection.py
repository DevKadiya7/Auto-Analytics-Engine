from __future__ import annotations

import pandas as pd


def detect_outliers(
    data: pd.DataFrame,
    method: str = "IQR",
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
    indicators: list[str] | None = None,
) -> pd.DataFrame:
    columns = ["district", "indicator", "month", "value", "method", "lower_bound", "upper_bound", "threshold"]
    if data.empty or not {"district", "month"}.issubset(data.columns):
        return pd.DataFrame(columns=columns)
    indicators = indicators or list(data.select_dtypes(include="number").columns)
    rows: list[dict[str, object]] = []

    for indicator in [column for column in indicators if column in data.columns]:
        series = pd.to_numeric(data[indicator], errors="coerce")
        valid = series.dropna()
        if len(valid) < 2:
            continue
        if method == "IQR":
            q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
            mask = (series < lower) | (series > upper)
            threshold = iqr_multiplier
        else:
            mean, std = valid.mean(), valid.std(ddof=0)
            if std == 0 or pd.isna(std):
                continue
            lower, upper = mean - zscore_threshold * std, mean + zscore_threshold * std
            mask = (series - mean).abs() > zscore_threshold * std
            threshold = zscore_threshold

        for index in data.index[mask.fillna(False)]:
            rows.append(
                {
                    "district": data.at[index, "district"],
                    "indicator": indicator,
                    "month": data.at[index, "month"],
                    "value": data.at[index, indicator],
                    "method": method,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "threshold": threshold,
                }
            )
    return pd.DataFrame(rows, columns=columns)
