from __future__ import annotations

from typing import Any

import pandas as pd

REQUIRED_COLUMNS = [
    "month",
    "district",
    "anc_coverage",
    "institutional_delivery",
    "immunization",
    "high_risk_cases",
]
NUMERIC_INDICATORS = [
    "anc_coverage",
    "institutional_delivery",
    "immunization",
    "high_risk_cases",
]


def load_csv(source: Any) -> tuple[pd.DataFrame | None, str | None]:
    try:
        data = pd.read_csv(source)
    except pd.errors.EmptyDataError:
        return None, "The uploaded file is empty. Provide a CSV with headers and data rows."
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        return None, f"The CSV could not be parsed: {exc}"
    except Exception as exc:
        return None, f"Unable to read the CSV: {exc}"

    if data.empty and len(data.columns) == 0:
        return None, "The CSV contains no columns."
    return data, None


def validate_dataset(data: pd.DataFrame) -> dict[str, Any]:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    numeric_columns = [column for column in NUMERIC_INDICATORS if column in data.columns]
    non_numeric_indicators = [
        column for column in numeric_columns if not pd.api.types.is_numeric_dtype(data[column])
    ]
    invalid_month_count = 0
    duplicate_key_count = 0
    invalid_value_counts: dict[str, int] = {}
    if "month" in data.columns:
        invalid_month_count = int(pd.to_datetime(data["month"], format="%Y-%m", errors="coerce").isna().sum())
    if {"district", "month"}.issubset(data.columns):
        duplicate_key_count = int(data.duplicated(subset=["district", "month"]).sum())
    for column in ("anc_coverage", "institutional_delivery", "immunization"):
        if column in data.columns and pd.api.types.is_numeric_dtype(data[column]):
            invalid_value_counts[column] = int(((data[column] < 0) | (data[column] > 100)).sum())
    if "high_risk_cases" in data.columns and pd.api.types.is_numeric_dtype(data["high_risk_cases"]):
        invalid_value_counts["high_risk_cases"] = int((data["high_risk_cases"] < 0).sum())

    errors = []
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}.")
    if non_numeric_indicators:
        errors.append(f"Indicator columns must be numeric: {', '.join(non_numeric_indicators)}.")
    if invalid_month_count:
        errors.append(f"{invalid_month_count} month value(s) are invalid; expected YYYY-MM.")
    if duplicate_key_count:
        errors.append(f"{duplicate_key_count} duplicate district/month record(s) found.")
    warnings = []
    invalid_values = {column: count for column, count in invalid_value_counts.items() if count}
    if invalid_values:
        details = ", ".join(f"{column}: {count}" for column, count in invalid_values.items())
        warnings.append(f"Suspicious numeric value(s) found ({details}); review the source data.")

    return {
        "valid": not errors,
        "errors": errors,
        "missing_columns": missing_columns,
        "non_numeric_indicators": non_numeric_indicators,
        "invalid_month_count": invalid_month_count,
        "duplicate_key_count": duplicate_key_count,
        "invalid_value_counts": invalid_value_counts,
        "warnings": warnings,
        "missing_values": data.isna().sum(),
        "duplicate_rows": int(data.duplicated().sum()),
    }


def clean_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean a copy without mutating the input data."""
    cleaned = data.copy(deep=True)
    duplicate_count = int(cleaned.duplicated().sum())
    missing_count = int(cleaned.isna().sum().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    for column in cleaned.columns:
        if not cleaned[column].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            median = cleaned[column].median()
            cleaned[column] = cleaned[column].fillna(0 if pd.isna(median) else median)
        else:
            mode = cleaned[column].mode(dropna=True)
            cleaned[column] = cleaned[column].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    return cleaned, {"missing_filled": missing_count, "duplicates_removed": duplicate_count}


def prepare_analysis_data(data: pd.DataFrame) -> pd.DataFrame:
    """Keep valid monthly rows and expose a private datetime sort key."""
    prepared = data.copy(deep=True)
    if "month" in prepared.columns:
        prepared["_month_date"] = pd.to_datetime(prepared["month"], format="%Y-%m", errors="coerce")
        prepared = prepared.dropna(subset=["_month_date"])
    return prepared


def numeric_summary(data: pd.DataFrame) -> pd.DataFrame:
    numeric = data.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    return numeric.agg(["count", "mean", "median", "min", "max"]).T.round(2)
