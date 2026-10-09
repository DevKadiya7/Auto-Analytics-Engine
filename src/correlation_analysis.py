from __future__ import annotations

from itertools import combinations

import pandas as pd


def correlation_matrix(data: pd.DataFrame, indicators: list[str] | None = None) -> pd.DataFrame:
    available = indicators or list(data.select_dtypes(include="number").columns)
    available = [column for column in available if column in data.columns]
    return data[available].corr(method="pearson") if available else pd.DataFrame()


def flagged_correlations(
    data: pd.DataFrame,
    threshold: float = 0.70,
    indicators: list[str] | None = None,
    min_observations: int = 3,
) -> pd.DataFrame:
    columns = ["indicator_a", "indicator_b", "correlation", "correlation_coefficient", "observations", "threshold", "interpretation"]
    available = indicators or list(data.select_dtypes(include="number").columns)
    rows: list[dict[str, object]] = []
    for indicator_a, indicator_b in combinations([c for c in available if c in data.columns], 2):
        pair = data[[indicator_a, indicator_b]].dropna()
        if len(pair) < min_observations:
            continue
        coefficient = pair[indicator_a].corr(pair[indicator_b], method="pearson")
        if pd.notna(coefficient) and abs(coefficient) >= threshold:
            interpretation = "positive relationship" if coefficient > 0 else "negative relationship"
            rows.append(
                {
                    "indicator_a": indicator_a,
                    "indicator_b": indicator_b,
                    "correlation": coefficient,
                    "correlation_coefficient": coefficient,
                    "observations": len(pair),
                    "threshold": threshold,
                    "interpretation": interpretation,
                }
            )
    return pd.DataFrame(rows, columns=columns)
