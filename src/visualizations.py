from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def severity_bar_chart(insights: pd.DataFrame) -> plt.Figure:
    counts = insights["severity"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    figure, axis = plt.subplots(figsize=(7, 3.5))
    counts.plot(kind="bar", ax=axis, color=["#4c956c", "#f2c14e", "#d1495b"])
    axis.set_title("Insight counts by severity")
    axis.set_xlabel("Severity")
    axis.set_ylabel("Count")
    axis.tick_params(axis="x", rotation=0)
    figure.tight_layout()
    return figure


def correlation_heatmap(matrix: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7, 5))
    if matrix.empty:
        axis.text(0.5, 0.5, "No correlation data available", ha="center", va="center")
        axis.set_axis_off()
        return figure
    image = axis.imshow(matrix.fillna(0).values, cmap="RdBu_r", vmin=-1, vmax=1)
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            axis.text(column, row, f"{matrix.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8)
    axis.set_title("Pearson correlation matrix")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return figure


def district_line_chart(data: pd.DataFrame, district: str, indicator: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 4))
    subset = data[data["district"] == district].copy()
    subset["_month_date"] = pd.to_datetime(subset["month"], format="%Y-%m", errors="coerce")
    subset = subset.sort_values("_month_date")
    if subset.empty:
        axis.text(0.5, 0.5, "No rows match the selected district", ha="center", va="center")
    else:
        axis.plot(subset["month"], subset[indicator], marker="o", color="#176b87")
        axis.set_title(f"{indicator} over time: {district}")
        axis.set_xlabel("Month")
        axis.set_ylabel(indicator)
    figure.tight_layout()
    return figure
