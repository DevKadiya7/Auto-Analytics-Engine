from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

from src.correlation_analysis import correlation_matrix, flagged_correlations
from src.data_loader import REQUIRED_COLUMNS, clean_data, load_csv, numeric_summary, prepare_analysis_data, validate_dataset
from src.insight_generator import add_threshold_breaches, generate_insights
from src.outlier_detection import detect_outliers
from src.trend_detection import detect_trends
from src.visualizations import correlation_heatmap, district_line_chart, severity_bar_chart


BASE_DIR = Path(__file__).parent
SAMPLE_PATH = BASE_DIR / "data" / "healthcare_performance.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


def _csv_bytes(data: pd.DataFrame) -> bytes:
    return data.to_csv(index=False).encode("utf-8")


def main() -> None:
    st.set_page_config(page_title="Automated Healthcare Insight Engine", page_icon="H", layout="wide")
    st.markdown(
        """
        <style>
        [data-testid="stMetric"] { border: 1px solid #dce5ea; border-radius: 8px; padding: 12px 14px; background: #ffffff; }
        [data-testid="stSidebar"] { background: #f4f8fa; }
        h1, h2, h3 { color: #153b50; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Automated Healthcare Insight Engine")
    st.caption("District-level performance monitoring, anomaly detection, and automated insights")

    with st.sidebar:
        st.header("Data source & controls")
        st.caption("Healthcare performance intelligence")
        uploaded = st.file_uploader("Upload healthcare CSV", type=["csv"])
        use_sample = st.checkbox("Use provided sample dataset", value=uploaded is None)
        trend_threshold = st.slider("Significant trend threshold (%)", 1.0, 100.0, 10.0, 1.0)
        method = st.selectbox("Outlier method", ["IQR", "Z-score"])
        iqr_multiplier = st.slider("IQR multiplier", 0.5, 3.0, 1.5, 0.1, disabled=method != "IQR")
        zscore_threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0, 0.1, disabled=method != "Z-score")
        correlation_threshold = st.slider("Absolute correlation threshold", 0.1, 1.0, 0.70, 0.05)
        st.divider()
        st.caption("The sample has only two months per district. Trends and correlations are exploratory, not causal evidence.")

    source = SAMPLE_PATH if use_sample else uploaded
    if source is None and uploaded is None:
        source = SAMPLE_PATH if SAMPLE_PATH.exists() else None
    if source is None:
        st.error("No CSV uploaded and the sample dataset is missing.")
        return
    if source == SAMPLE_PATH:
        st.info("Using the exact healthcare sample dataset. Upload a CSV and turn off the sample toggle to analyze another file.")

    data, error = load_csv(source)
    if error:
        st.error(error)
        return
    if data is None:
        st.error("No data was loaded.")
        return

    print("\n=== Healthcare dataset head ===")
    print(data.head())
    print("=== Healthcare dataset info ===")
    data.info()
    print("=== Missing values by column ===")
    print(data.isna().sum())

    validation = validate_dataset(data)
    st.caption(f"Source: {'provided sample' if source == SAMPLE_PATH else 'uploaded CSV'} | Active validation: {'PASS' if not validation['errors'] else 'REVIEW REQUIRED'}")

    with st.expander("Data loading and validation", expanded=True):
        st.subheader("Preview (head)")
        st.dataframe(data.head(), use_container_width=True)
        st.subheader("Data types and missing values")
        quality = pd.DataFrame({"dtype": data.dtypes.astype(str), "missing": data.isna().sum()})
        st.dataframe(quality, use_container_width=True)
        info_buffer = StringIO()
        data.info(buf=info_buffer)
        st.code(info_buffer.getvalue(), language="text")
        if validation["errors"]:
            for message in validation["errors"]:
                st.error(message)
            st.info("Fix the validation errors or upload a valid healthcare CSV to run analytics.")
            return
        st.success("Required schema, monthly dates, and numeric indicators are valid.")
        for warning in validation.get("warnings", []):
            st.warning(warning)
        if validation["duplicate_rows"] or int(data.isna().sum().sum()):
            st.warning(
                f"Cleaning will address {validation['duplicate_rows']} duplicate row(s) and "
                f"{int(data.isna().sum().sum())} missing value(s) in a separate analysis copy."
            )

    cleaned, cleaning_report = clean_data(data)
    analysis_data = prepare_analysis_data(cleaned)
    indicators = [column for column in REQUIRED_COLUMNS if column in analysis_data.columns and column not in {"month", "district"}]
    if not indicators:
        st.warning("No numeric healthcare indicators are available for analysis.")
        return

    with st.expander("Safe cleaning report"):
        st.write(
            f"Analysis uses a copy of the original data: filled {cleaning_report['missing_filled']} missing values "
            f"and removed {cleaning_report['duplicates_removed']} duplicate rows."
        )
        st.download_button("Download cleaned CSV", _csv_bytes(cleaned), "healthcare_performance_cleaned.csv", "text/csv")

    with st.sidebar:
        st.header("Live filters")
        district_options = sorted(analysis_data["district"].dropna().astype(str).unique())
        month_options = sorted(analysis_data["month"].dropna().astype(str).unique())
        selected_districts = st.multiselect("District", district_options, default=district_options)
        selected_months = st.multiselect("Month", month_options, default=month_options)
        selected_indicators = st.multiselect("Indicator", indicators, default=indicators)

    if not selected_districts or not selected_months or not selected_indicators:
        st.info("Select at least one district, month, and indicator to display live results.")
        return
    filtered = analysis_data[
        analysis_data["district"].astype(str).isin(selected_districts)
        & analysis_data["month"].astype(str).isin(selected_months)
    ].copy()

    trends = detect_trends(filtered, trend_threshold, selected_indicators)
    outliers = detect_outliers(filtered, method, iqr_multiplier, zscore_threshold, selected_indicators)
    matrix = correlation_matrix(filtered, selected_indicators)
    correlations = flagged_correlations(filtered, correlation_threshold, selected_indicators)
    insights = generate_insights(trends, outliers, correlations, trend_threshold)

    st.header("Key Automated Insights")
    threshold_indicator = st.selectbox("Optional lower-bound threshold indicator", ["None"] + selected_indicators)
    if threshold_indicator != "None":
        value_threshold = st.slider(f"{threshold_indicator} lower-bound threshold", 0.0, 100.0, 70.0, 1.0)
        insights = add_threshold_breaches(insights, filtered, threshold_indicator, value_threshold)
        st.caption("This is an operational screening threshold configured by the user, not a clinical standard.")
    insight_severity = st.multiselect("Severity", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    insight_types = st.multiselect("Insight type", ["trend", "outlier", "correlation", "threshold_breach"], default=["trend", "outlier", "correlation", "threshold_breach"])
    insight_search = st.text_input("Search insight explanations", placeholder="district, indicator, or phrase")
    visible_insights = insights[
        insights["severity"].isin(insight_severity) & insights["type"].isin(insight_types)
    ].copy()
    if insight_search:
        visible_insights = visible_insights[
            visible_insights.astype(str).apply(lambda column: column.str.contains(insight_search, case=False, na=False)).any(axis=1)
        ]

    OUTPUT_DIR.mkdir(exist_ok=True)
    insights.to_csv(OUTPUT_DIR / "insights.csv", index=False)
    st.download_button("Download insights CSV", _csv_bytes(insights), "insights.csv", "text/csv")

    kpis = st.columns(5)
    kpis[0].metric("Districts", f"{filtered['district'].nunique():,}")
    kpis[1].metric("Records analyzed", f"{len(filtered):,}")
    kpis[2].metric("Generated insights", f"{len(insights):,}")
    kpis[3].metric("High severity", f"{int((insights['severity'] == 'High').sum()):,}")
    kpis[4].metric("Outliers", f"{len(outliers):,}")

    if visible_insights.empty:
        st.info("No findings meet the current thresholds and filters.")
    else:
        st.dataframe(visible_insights, use_container_width=True, hide_index=True)
        st.pyplot(severity_bar_chart(insights), clear_figure=True)

    trend_tab, outlier_tab, correlation_tab, district_tab = st.tabs(["Trends", "Outliers", "Correlations", "District Comparison"])
    with trend_tab:
        st.subheader("Monthly percentage changes")
        st.caption(f"Significant when absolute change is at least {trend_threshold:.1f}%. Direction is calculated per district and indicator.")
        if trends.empty:
            st.info("No consecutive monthly observations are available under the current filters.")
        else:
            trend_display = trends.copy()
            trend_display["direction"] = trend_display["change_pct"].map(lambda value: "Increase" if value > 0 else "Decrease" if value < 0 else "No change")
            st.dataframe(trend_display, use_container_width=True, hide_index=True)
    with outlier_tab:
        st.subheader(f"{method} statistical flags")
        if outliers.empty:
            st.info("No statistical outliers detected with the current method and filters.")
        else:
            st.dataframe(outliers, use_container_width=True, hide_index=True)
            st.caption("Statistical flags support review; they are not clinical conclusions.")
    with correlation_tab:
        st.subheader("Pearson correlation analysis")
        st.caption("Correlations are fragile with only 12 rows and two months per district and do not establish causation.")
        if matrix.empty:
            st.info("At least two usable numeric indicators are needed for a correlation matrix.")
        else:
            st.dataframe(matrix.round(3), use_container_width=True)
            st.pyplot(correlation_heatmap(matrix), clear_figure=True)
            if correlations.empty:
                st.info("No indicator pairs meet the configured absolute correlation threshold.")
            else:
                st.dataframe(correlations, use_container_width=True, hide_index=True)
            OUTPUT_DIR.mkdir(exist_ok=True)
            matrix.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
            st.download_button("Download correlation matrix CSV", _csv_bytes(matrix.reset_index()), "correlation_matrix.csv", "text/csv")
    with district_tab:
        st.subheader("District performance over time")
        chart_columns = st.columns(2)
        chart_district = chart_columns[0].selectbox("Chart district", selected_districts)
        chart_indicator = chart_columns[1].selectbox("Chart indicator", selected_indicators)
        st.pyplot(district_line_chart(filtered, chart_district, chart_indicator), clear_figure=True)


if __name__ == "__main__":
    main()
