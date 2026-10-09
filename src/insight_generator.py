from __future__ import annotations

from typing import Any

import pandas as pd

INSIGHT_COLUMNS = [
    "insight_id",
    "type",
    "indicator",
    "entity",
    "period",
    "value",
    "prev_value",
    "change_pct",
    "severity",
    "explanation",
]


def _severity_for_change(change_pct: float, threshold: float) -> str:
    magnitude = abs(change_pct)
    if magnitude >= 2 * threshold:
        return "High"
    return "Medium"


def _severity_for_outlier(value: float, lower: float, upper: float) -> str:
    distance = lower - value if value < lower else value - upper
    span = max(upper - lower, 1e-12)
    return "High" if distance >= span else "Medium"


def generate_insights(
    trends: pd.DataFrame,
    outliers: pd.DataFrame,
    correlations: pd.DataFrame,
    trend_threshold: float = 10.0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    significant = trends[trends["is_significant"]] if not trends.empty else trends
    for row in significant.itertuples(index=False):
        direction = "increased" if row.change_pct > 0 else "decreased"
        rows.append(
            {
                "type": "trend",
                "indicator": row.indicator,
                "entity": row.district,
                "period": row.month,
                "value": row.current_value,
                "prev_value": row.previous_value,
                "change_pct": row.change_pct,
                "severity": _severity_for_change(row.change_pct, trend_threshold),
                "explanation": (
                    f"{row.indicator} in {row.district} {direction} from {row.previous_value:.2f} "
                    f"to {row.current_value:.2f} ({row.change_pct:.1f}%), meeting the configured "
                    f"{trend_threshold:.1f}% trend threshold."
                ),
            }
        )

    for row in outliers.itertuples(index=False):
        severity = _severity_for_outlier(row.value, row.lower_bound, row.upper_bound)
        bound_text = f"below {row.lower_bound:.2f}" if row.value < row.lower_bound else f"above {row.upper_bound:.2f}"
        rows.append(
            {
                "type": "outlier",
                "indicator": row.indicator,
                "entity": row.district,
                "period": row.month,
                "value": row.value,
                "prev_value": None,
                "change_pct": None,
                "severity": severity,
                "explanation": (
                    f"{row.indicator} for {row.district} was {row.value:.2f} in {row.month}, "
                    f"{bound_text} the {row.method} detection bound."
                ),
            }
        )

    for row in correlations.itertuples(index=False):
        strength = abs(row.correlation)
        severity = "High" if strength >= 0.90 else "Medium" if strength >= 0.80 else "Low"
        direction = "positive" if row.correlation > 0 else "negative"
        rows.append(
            {
                "type": "correlation",
                "indicator": f"{row.indicator_a}:{row.indicator_b}",
                "entity": None,
                "period": None,
                "value": row.correlation,
                "prev_value": None,
                "change_pct": None,
                "severity": severity,
                "explanation": (
                    f"{row.indicator_a} and {row.indicator_b} have a {direction} Pearson correlation "
                    f"of {row.correlation:.2f} across {row.observations} observations."
                ),
            }
        )

    result = pd.DataFrame(rows, columns=[column for column in INSIGHT_COLUMNS if column != "insight_id"])
    if result.empty:
        return pd.DataFrame(columns=INSIGHT_COLUMNS)
    result.insert(0, "insight_id", [f"INS-{index:04d}" for index in range(1, len(result) + 1)])
    return result[INSIGHT_COLUMNS]


def add_threshold_breaches(
    insights: pd.DataFrame,
    data: pd.DataFrame,
    indicator: str,
    threshold: float,
) -> pd.DataFrame:
    """Append explicit lower-bound breaches; the threshold is supplied by the user in the UI."""
    if indicator not in data.columns or not {"district", "month"}.issubset(data.columns):
        return insights
    rows = []
    for row in data.loc[data[indicator] < threshold, ["district", "month", indicator]].itertuples(index=False):
        severity = "High" if row[2] < threshold * 0.75 else "Medium"
        rows.append(
            {
                "type": "threshold_breach",
                "indicator": indicator,
                "entity": row[0],
                "period": row[1],
                "value": row[2],
                "prev_value": None,
                "change_pct": None,
                "severity": severity,
                "explanation": f"{indicator} for {row[0]} was {row[2]:.2f}, below the configured threshold of {threshold:.2f}.",
            }
        )
    if rows:
        additions = pd.DataFrame(rows)
        records = rows if insights.empty else insights.to_dict("records") + rows
        result = pd.DataFrame(records, columns=[column for column in INSIGHT_COLUMNS if column != "insight_id"])
        result.insert(0, "insight_id", "")
        result["insight_id"] = [f"INS-{index:04d}" for index in range(1, len(result) + 1)]
        return result[INSIGHT_COLUMNS]
    return insights
