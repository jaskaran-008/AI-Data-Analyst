"""
data_analysis.py
All FACTUAL calculations live here (Pandas only, no AI).
build_analysis_summary() creates the compact summary that is shown in the app
AND passed to the AI model.
"""

import re
import warnings

import pandas as pd

# Keywords used to guess the meaning of columns. Order matters:
# "profit" and "quantity" are checked before "sales" so that a generic
# keyword like "total" does not grab "Total Profit".
ROLE_KEYWORDS = {
    "profit": ["profit", "earnings"],
    "quantity": ["quantity", "qty", "units", "volume"],
    "sales": ["sales", "revenue", "amount", "turnover", "total"],
    "category": ["category", "segment", "department", "type", "class"],
    "region": ["region", "country", "state", "city", "area", "market", "location"],
    "product": ["product", "item", "sku"],
}

DATE_NAME_HINTS = ["date", "time", "timestamp", "day", "month", "period"]


# ---------------------------------------------------------------
# Loading
# ---------------------------------------------------------------

def load_csv(file):
    """Read a CSV into a DataFrame. Raises ValueError with a friendly message."""
    try:
        df = pd.read_csv(file)
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded CSV file is empty.")
    except UnicodeDecodeError:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding="latin-1")
        except Exception:
            raise ValueError("Could not decode the file. Please save it as a UTF-8 CSV.")
    except Exception as e:
        raise ValueError(f"Could not read this file as a CSV: {e}")

    if df.shape[1] == 0 or df.empty:
        raise ValueError("The CSV has no data rows or no columns.")

    df.columns = [str(c).strip() for c in df.columns]
    return df


def convert_date_columns(df):
    """Try to convert date-like text columns to datetime.
    Returns (new_df, list_of_date_columns)."""
    df = df.copy()
    date_cols = []

    for col in df.columns:
        series = df[col]

        if pd.api.types.is_datetime64_any_dtype(series):
            date_cols.append(col)
            continue

        is_text = pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
        if not is_text:
            continue

        sample = series.dropna().astype(str).head(200)
        if sample.empty:
            continue

        name_hint = any(h in col.lower() for h in DATE_NAME_HINTS)
        looks_like_date = sample.str.contains(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}").mean() > 0.8
        if not (name_hint or looks_like_date):
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed_sample = pd.to_datetime(sample, errors="coerce")
        if parsed_sample.notna().mean() >= 0.9:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df[col] = pd.to_datetime(df[col], errors="coerce")
            date_cols.append(col)

    return df, date_cols


# ---------------------------------------------------------------
# Basic dataset information
# ---------------------------------------------------------------

def is_id_column(name):
    """True for columns that look like identifiers (e.g. 'Order ID', 'customer_id')."""
    n = str(name)
    return bool(re.search(r"(^|[^a-z])id($|[^a-z])", n.lower())) or n.endswith("ID")


def get_dataset_shape(df):
    return df.shape


def get_column_names(df):
    return list(df.columns)


def get_data_types(df):
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def get_missing_values(df):
    """Return a list of columns that have missing values."""
    missing = df.isna().sum()
    rows = []
    for col, count in missing.items():
        if count > 0:
            rows.append({
                "column": col,
                "missing": int(count),
                "percent": round(count / len(df) * 100, 1),
            })
    return rows


def count_duplicates(df):
    return int(df.duplicated().sum())


def get_numeric_columns(df):
    """Numeric columns, ignoring ID-like columns."""
    return [c for c in df.select_dtypes(include="number").columns if not is_id_column(c)]


def get_categorical_columns(df, date_cols):
    """Text/category columns (not numeric, not dates)."""
    cols = []
    for c in df.columns:
        if c in date_cols or pd.api.types.is_numeric_dtype(df[c]):
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        cols.append(c)
    return cols


def get_basic_statistics(df, numeric_cols, max_cols=8):
    """describe() output for numeric columns, as a nested dict."""
    if not numeric_cols:
        return {}
    stats = df[numeric_cols[:max_cols]].describe().round(2)
    return stats.to_dict()


# ---------------------------------------------------------------
# Detecting common business columns
# ---------------------------------------------------------------

def detect_columns(df, date_cols):
    """Guess which columns are sales, profit, category, region, etc.
    Works even if nothing matches (values are then None)."""
    numeric_cols = get_numeric_columns(df)
    categorical_cols = [c for c in get_categorical_columns(df, date_cols) if not is_id_column(c)]
    used = set()

    def find(role, candidates, check=None):
        for keyword in ROLE_KEYWORDS[role]:
            for col in candidates:
                if col in used:
                    continue
                if keyword in col.lower() and (check is None or check(col)):
                    used.add(col)
                    return col
        return None

    def unique_between(low, high):
        return lambda col: low <= df[col].nunique() <= high

    detected = {
        "profit": find("profit", numeric_cols),
        "quantity": find("quantity", numeric_cols),
        "sales": find("sales", numeric_cols),
        "category": find("category", categorical_cols, unique_between(2, 50)),
        "region": find("region", categorical_cols, unique_between(2, 50)),
        "product": find("product", categorical_cols, unique_between(2, 10**9)),
        "date": date_cols[0] if date_cols else None,
    }

    # Main measure used for comparisons: sales > profit > quantity > first numeric column
    main_measure = None
    for role in ("sales", "profit", "quantity"):
        if detected[role]:
            main_measure = detected[role]
            break
    if main_measure is None and numeric_cols:
        main_measure = numeric_cols[0]
    detected["main_measure"] = main_measure

    # Columns to compare groups by
    group_columns = {}
    for role in ("category", "region", "product"):
        if detected[role]:
            group_columns[role] = detected[role]
    if not group_columns:  # fallback: any low-cardinality text column
        for col in categorical_cols:
            if 2 <= df[col].nunique() <= 20 and len(group_columns) < 2:
                group_columns[col] = col
    detected["group_columns"] = group_columns

    return detected


# ---------------------------------------------------------------
# Business metrics
# ---------------------------------------------------------------

def _make_kpi(df, col, label):
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if values.empty:
        return None
    return {
        "label": label,
        "column": col,
        "total": round(float(values.sum()), 2),
        "average": round(float(values.mean()), 2),
        "median": round(float(values.median()), 2),
        "min": round(float(values.min()), 2),
        "max": round(float(values.max()), 2),
    }


def calculate_kpis(df, detected, numeric_cols):
    """Total / average / median / min / max for sales, profit, quantity.
    If none were detected, use the first 3 numeric columns instead."""
    kpis = []
    for role in ("sales", "profit", "quantity"):
        col = detected[role]
        if col:
            kpi = _make_kpi(df, col, role.capitalize())
            if kpi:
                kpis.append(kpi)
    if not kpis:
        for col in numeric_cols[:3]:
            kpi = _make_kpi(df, col, col)
            if kpi:
                kpis.append(kpi)
    return kpis


def top_groups(df, group_col, measure_col=None, top_n=5):
    """Compare groups: sum of measure_col per group (or row counts if no measure)."""
    if measure_col:
        grouped = df.groupby(group_col)[measure_col].sum()
        measure_name = f"Sum of {measure_col}"
    else:
        grouped = df[group_col].value_counts()
        measure_name = "Row count"

    grouped = grouped.sort_values(ascending=False)
    if grouped.empty:
        return None

    total = grouped.sum()
    all_positive = bool((grouped >= 0).all())

    def to_rows(series):
        rows = []
        for name, value in series.items():
            row = {"group": str(name), "value": round(float(value), 2)}
            if total > 0 and all_positive:
                row["share_pct"] = round(float(value) / float(total) * 100, 1)
            rows.append(row)
        return rows

    return {
        "group_column": group_col,
        "measure": measure_name,
        "number_of_groups": int(len(grouped)),
        "top": to_rows(grouped.head(top_n)),
        "bottom": to_rows(grouped.tail(3)) if len(grouped) > top_n else [],
    }


def get_time_series(df, date_col, measure_col=None):
    """Aggregate the measure by month (or by day for short date ranges).
    Returns a Series (index = period text) or None if not enough data."""
    columns = [date_col] + ([measure_col] if measure_col else [])
    temp = df[columns].dropna(subset=[date_col])
    if temp.empty:
        return None

    span_days = (temp[date_col].max() - temp[date_col].min()).days
    if span_days > 60:
        period = temp[date_col].dt.to_period("M").astype(str)
    else:
        period = temp[date_col].dt.strftime("%Y-%m-%d")

    if measure_col:
        series = temp.groupby(period)[measure_col].sum()
    else:
        series = temp.groupby(period).size()

    series = series.sort_index()
    return series if len(series) >= 2 else None


def calculate_time_trend(df, date_col, measure_col=None):
    series = get_time_series(df, date_col, measure_col)
    if series is None:
        return None

    first, last = float(series.iloc[0]), float(series.iloc[-1])
    change_pct = round((last - first) / abs(first) * 100, 1) if first != 0 else None
    span_days = (df[date_col].max() - df[date_col].min()).days

    return {
        "date_column": date_col,
        "measure": measure_col if measure_col else "row count",
        "granularity": "monthly" if span_days > 60 else "daily",
        "number_of_periods": int(len(series)),
        "first_period": str(series.index[0]),
        "last_period": str(series.index[-1]),
        "first_value": round(first, 2),
        "last_value": round(last, 2),
        "change_pct": change_pct,
        "best_period": str(series.idxmax()),
        "worst_period": str(series.idxmin()),
        "average_per_period": round(float(series.mean()), 2),
    }


# ---------------------------------------------------------------
# Unusual values and relationships
# ---------------------------------------------------------------

def find_outliers(df, numeric_cols, max_cols=10, top_n=5):
    """IQR method: values far outside the middle 50% are flagged as outliers."""
    results = []
    for col in numeric_cols[:max_cols]:
        values = df[col].dropna()
        if len(values) < 8:
            continue
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = values[(values < lower) | (values > upper)]
        if len(outliers) > 0:
            results.append({
                "column": col,
                "outlier_count": int(len(outliers)),
                "percent_of_rows": round(len(outliers) / len(values) * 100, 1),
                "normal_range_low": round(float(lower), 2),
                "normal_range_high": round(float(upper), 2),
                "min_value": round(float(values.min()), 2),
                "max_value": round(float(values.max()), 2),
            })
    results.sort(key=lambda r: r["outlier_count"], reverse=True)
    return results[:top_n]


def find_correlations(df, numeric_cols, max_cols=10, min_abs=0.5, top_n=5):
    """Strongest correlations between pairs of numeric columns."""
    cols = numeric_cols[:max_cols]
    if len(cols) < 2:
        return []

    corr = df[cols].corr()
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if pd.notna(value) and abs(value) >= min_abs:
                pairs.append({
                    "column_1": cols[i],
                    "column_2": cols[j],
                    "correlation": round(float(value), 2),
                })
    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
    return pairs[:top_n]


# ---------------------------------------------------------------
# The compact summary (shown in the UI AND sent to the AI)
# ---------------------------------------------------------------

def build_analysis_summary(df, date_cols):
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df, date_cols)
    detected = detect_columns(df, date_cols)
    main = detected["main_measure"]

    # KPIs
    kpis = calculate_kpis(df, detected, numeric_cols)
    profit_margin = None
    if detected["sales"] and detected["profit"]:
        total_sales = pd.to_numeric(df[detected["sales"]], errors="coerce").sum()
        total_profit = pd.to_numeric(df[detected["profit"]], errors="coerce").sum()
        if total_sales != 0:
            profit_margin = round(float(total_profit / total_sales * 100), 1)

    # Group comparisons (max 3 group columns to keep the summary small)
    group_comparisons = []
    for label, col in list(detected["group_columns"].items())[:3]:
        result = top_groups(df, col, main)
        if result:
            group_comparisons.append(result)

    # Extra comparison: profit by the first group column (if profit is not the main measure)
    if detected["profit"] and detected["profit"] != main and detected["group_columns"]:
        first_group = list(detected["group_columns"].values())[0]
        result = top_groups(df, first_group, detected["profit"])
        if result:
            group_comparisons.append(result)

    # Time trend
    time_trend = None
    if detected["date"]:
        time_trend = calculate_time_trend(df, detected["date"], main)
        if time_trend:
            time_trend["note"] = "The first and last periods may be incomplete."

    constant_columns = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    missing_rows = get_missing_values(df)

    return {
        "overview": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": get_column_names(df),
            "data_types": get_data_types(df),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "date_columns": date_cols,
        },
        "data_quality": {
            "duplicate_rows": count_duplicates(df),
            "total_missing_cells": int(df.isna().sum().sum()),
            "missing_by_column": missing_rows,
            "constant_columns": constant_columns,
        },
        "detected_columns": detected,
        "kpis": kpis,
        "profit_margin_pct": profit_margin,
        "group_comparisons": group_comparisons,
        "time_trend": time_trend,
        "outliers": find_outliers(df, numeric_cols),
        "correlations": find_correlations(df, numeric_cols),
        "statistics": get_basic_statistics(df, numeric_cols),
    }