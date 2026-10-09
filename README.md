# Healthcare Auto-Analytics Engine

A general-purpose Streamlit application for Assignment 4 district-level healthcare data. It validates the monthly schema, detects trends and statistical outliers, calculates Pearson correlations, and generates structured factual insights without an LLM API.

## Windows PowerShell setup

From the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
python -m unittest discover -s tests -v
streamlit run app.py
```

Open `http://localhost:8501`. The app starts with `data/healthcare_performance.csv`; use the sidebar uploader and sample toggle for another valid healthcare CSV.

## Architecture

- `app.py`: Streamlit UI, live filters, threshold controls, KPI summary, searchable insights, tabs, downloads, and output persistence.
- `src/data_loader.py`: CSV loading, schema/month validation, duplicate district/month checks, healthcare range validation, safe copy cleaning, and summary statistics.
- `src/trend_detection.py`: chronological district/indicator percentage changes.
- `src/outlier_detection.py`: configurable IQR or Z-score flags.
- `src/correlation_analysis.py`: Pearson matrix and unique thresholded pairs.
- `src/insight_generator.py`: structured, data-driven insight records and optional threshold breaches.
- `src/visualizations.py`: severity bar chart, correlation heatmap, and district line chart.

## Algorithms and severity rules

Trend percentage change is `(current - previous) / previous * 100`, calculated after sorting each district/indicator by valid month. A zero previous value produces an undefined percentage and is not flagged. A significant trend is Medium; a change at least twice the configured trend threshold is High.

IQR flags values below `Q1 - multiplier * IQR` or above `Q3 + multiplier * IQR`. Z-score flags values beyond the configured number of population standard deviations. Outliers are Medium when outside a bound and High when their distance beyond the bound is at least the full IQR detection interval. Constant and undersized series are skipped.

Correlation flags unique pairs with `abs(r) >= threshold`; constant columns and pairs with fewer than three observations are ignored. Correlation severity is Low for `0.70 <= |r| < 0.80`, Medium for `0.80 <= |r| < 0.90`, and High for `|r| >= 0.90`. The optional lower-bound threshold breach is user-configured in the UI and remains separate from trends and outliers.

The dashboard header is followed by five live KPIs: districts, records analyzed, generated insights, high-severity insights, and outliers. The sidebar controls the source, district/month/indicator filters, trend threshold, selected outlier method, and correlation threshold. The main area provides searchable/filterable insights and tabs for Trends, Outliers, Correlations, and District Comparison.

## Outputs and limitations

Running the app writes `outputs/insights.csv` and `outputs/correlation_matrix.csv`; the UI also provides downloads. The original DataFrame is never modified: cleaning and analysis use copies. Missing numeric values use the column median, text values use the mode or `Unknown`, and duplicate rows are removed from the analysis copy.

The supplied sample contains only 12 rows, two months per district, and six districts. Trend results are short-series screening signals. Pearson correlations are statistically fragile at this sample size and do not establish causation. Invalid months, missing required columns, malformed/empty files, duplicate district/month keys, out-of-range percentage/count values, missing values, constant indicators, and insufficient observations are reported or skipped gracefully. The Streamlit server console also receives the loaded `head()`, `info()`, and per-column missing-value report.

## Assignment checklist

- Part A: exact healthcare dataset, Pandas loading, preview, `info()`, missing counts, validation, and live district/month/indicator filters.
- Part B: configurable chronological percentage-change trends with zero-previous handling.
- Part C: configurable IQR/Z-score outlier detection with bounds and method in each row.
- Part D: complete Pearson matrix, unique thresholded pairs, CSV export, and limitation note.
- Part E: structured `INS-0001` insights with type, indicator, entity, period, value, previous value, change percentage, severity, and explanation.
- Part F: insight table, severity bar chart, heatmap, per-district line chart, trend/outlier tables, and downloads.

Tests verify the exact Ahmedabad `85 -> 69` change, Mehsana ANC outlier detection, threshold breaches, sorting, zero previous values, constant columns, insufficient observations, malformed/empty input, duplicate cleaning, threshold changes, and insight schema. Streamlit startup was verified with an HTTP 200 response; screenshots and full browser interaction should be captured manually using the instructions in `screenshots/README.md`.
