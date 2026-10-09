from __future__ import annotations

import pandas as pd


def detect_trends(
    data: pd.DataFrame,
    threshold: float = 10.0,
    indicators: list[str] | None = None,
) -> pd.DataFrame:
    columns = ["district", "indicator", "previous_month", "month", "previous_value", "current_value", "change_pct", "threshold", "direction", "is_significant"]
    if data.empty or not {"district", "month"}.issubset(data.columns):
        return pd.DataFrame(columns=columns)
    indicators = indicators or list(data.select_dtypes(include="number").columns)
    available = [column for column in indicators if column in data.columns]
    if not available:
        return pd.DataFrame(columns=columns)

    working = data.copy(deep=True)
    working["_month_date"] = pd.to_datetime(working["month"], format="%Y-%m", errors="coerce")
    working = working.dropna(subset=["_month_date"]).sort_values(["district", "_month_date"])
    long_data = working.melt(
        id_vars=["district", "month", "_month_date"],
        value_vars=available,
        var_name="indicator",
        value_name="current_value",
    ).sort_values(["district", "indicator", "_month_date"])
    long_data["previous_value"] = long_data.groupby(["district", "indicator"])["current_value"].shift(1)
    long_data["previous_month"] = long_data.groupby(["district", "indicator"])["month"].shift(1)
    long_data = long_data.dropna(subset=["previous_value", "current_value"])
    long_data["change_pct"] = long_data.apply(
        lambda row: ((row["current_value"] - row["previous_value"]) / row["previous_value"] * 100)
        if row["previous_value"] != 0 else float("nan"),
        axis=1,
    )
    long_data["is_significant"] = long_data["change_pct"].abs().ge(threshold).fillna(False)
    long_data["threshold"] = threshold
    long_data["direction"] = long_data["change_pct"].map(
        lambda value: "Increase" if value > 0 else "Decrease" if value < 0 else "No change"
    )
    return long_data[columns].reset_index(drop=True)
