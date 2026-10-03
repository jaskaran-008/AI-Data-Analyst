"""
visualizations.py
Creates a few useful Matplotlib charts, depending on which columns exist.
Each chart is returned as (title, figure). The app displays them.
"""

import matplotlib.pyplot as plt

from data_analysis import get_time_series

MAX_CHARTS = 4


def _trend_chart(df, date_col, measure_col):
    series = get_time_series(df, date_col, measure_col)
    if series is None:
        return None

    labels = list(series.index)
    x = list(range(len(labels)))
    ylabel = measure_col if measure_col else "Number of rows"

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, series.values, marker="o")
    step = max(1, len(labels) // 8)  # avoid crowded x labels
    ax.set_xticks(x[::step])
    ax.set_xticklabels(labels[::step], rotation=45, ha="right")
    ax.set_title(f"{ylabel} over time")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return f"{ylabel} over time", fig


def _bar_chart(df, group_col, measure_col, top_n=10):
    if measure_col:
        data = df.groupby(group_col)[measure_col].sum().sort_values(ascending=False).head(top_n)
        xlabel = f"Total {measure_col}"
    else:
        data = df[group_col].value_counts().head(top_n)
        xlabel = "Number of rows"

    if data.empty:
        return None

    names = [str(n) for n in data.index][::-1]  # reversed so the biggest bar is on top
    values = list(data.values)[::-1]

    title = f"{xlabel} by {group_col}"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names, values)
    ax.set_title(title + (f" (top {top_n})" if len(data) == top_n else ""))
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    return title, fig


def _histogram(df, col):
    values = df[col].dropna()
    if values.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=20)
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return f"Distribution of {col}", fig


def create_visualizations(df, summary):
    """Return a list of (title, figure) with at most MAX_CHARTS charts."""
    detected = summary["detected_columns"]
    main = detected["main_measure"]
    date_col = detected["date"]
    groups = list(detected["group_columns"].values())
    charts = []

    def add(builder, *args):
        try:
            result = builder(*args)
            if result:
                charts.append(result)
        except Exception:
            pass  # skip a chart that fails instead of crashing the app

    # 1. Time trend
    if date_col:
        add(_trend_chart, df, date_col, main)

    # 2. Up to two group comparisons (category / region / product ...)
    for col in groups[:2]:
        add(_bar_chart, df, col, main)

    # 3. Profit comparison for the first group
    profit = detected["profit"]
    if profit and profit != main and groups:
        add(_bar_chart, df, groups[0], profit)

    # 4. Fallback: distribution of the main numeric column
    if not charts and main:
        add(_histogram, df, main)

    return charts[:MAX_CHARTS]