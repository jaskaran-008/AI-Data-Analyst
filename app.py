"""
app.py
Main Streamlit application.
Run with:  streamlit run app.py
"""

import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from ai_analysis import (
    MODEL_NAME,
    AIAnalysisError,
    answer_question,
    generate_ai_insights,
    get_api_key,
)
from data_analysis import build_analysis_summary, convert_date_columns, load_csv
from visualizations import create_visualizations

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

st.title("📊 AI Data Analyst")
st.write(
    "Upload any CSV file. Pandas analyzes it automatically, then an open-weight AI model "
    f"(**{MODEL_NAME}** via Groq) explains the verified findings in plain business language."
)


# ---------- helper functions ----------

@st.cache_data(show_spinner=False)
def run_analysis(file_bytes: bytes):
    """Load the CSV and run the full Pandas analysis (cached so reruns are fast)."""
    df = load_csv(io.BytesIO(file_bytes))
    df, date_cols = convert_date_columns(df)
    summary = build_analysis_summary(df, date_cols)
    return df, summary


def fmt(value):
    """Format numbers nicely for display."""
    try:
        return f"{value:,.2f}"
    except (TypeError, ValueError):
        return str(value)


# ---------- 1. Upload ----------

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded is None:
    st.info("👆 Upload a CSV file to begin. You can test with `sample_data/sales.csv`.")
    st.stop()

file_bytes = uploaded.getvalue()

# Reset stored AI results when a new file is uploaded
file_key = f"{uploaded.name}-{len(file_bytes)}"
if st.session_state.get("file_key") != file_key:
    st.session_state["file_key"] = file_key
    st.session_state["insights"] = None
    st.session_state["answer"] = None

try:
    with st.spinner("Analyzing your data with Pandas..."):
        df, summary = run_analysis(file_bytes)
except ValueError as e:  # friendly errors raised by load_csv
    st.error(str(e))
    st.stop()
except Exception as e:  # anything unexpected
    st.error(f"Something went wrong while analyzing this file: {e}")
    st.stop()

overview = summary["overview"]
quality = summary["data_quality"]
detected = summary["detected_columns"]

# ---------- 2. Dataset overview ----------

st.header("1. Dataset Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{overview['rows']:,}")
c2.metric("Columns", overview["columns"])
c3.metric("Numeric columns", len(overview["numeric_columns"]))

st.subheader("Preview")
st.dataframe(df.head(10), use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Columns and data types")
    types_df = pd.DataFrame(
        {"Column": list(overview["data_types"].keys()),
         "Type": list(overview["data_types"].values())}
    )
    st.dataframe(types_df, hide_index=True, use_container_width=True)
with right:
    st.subheader("Basic statistics (numeric columns)")
    if summary["statistics"]:
        st.dataframe(pd.DataFrame(summary["statistics"]), use_container_width=True)
    else:
        st.info("No numeric columns found.")

# ---------- 3. Data quality ----------

st.header("2. Data Quality")
q1, q2 = st.columns(2)
q1.metric("Duplicate rows", quality["duplicate_rows"])
q2.metric("Missing cells", quality["total_missing_cells"])

if quality["missing_by_column"]:
    st.write("Columns with missing values:")
    st.dataframe(pd.DataFrame(quality["missing_by_column"]), hide_index=True)
else:
    st.success("No missing values found.")

if quality["constant_columns"]:
    st.warning("Columns with only one unique value: " + ", ".join(quality["constant_columns"]))

# ---------- 4. Business metrics ----------

st.header("3. Business Metrics")

role_names = ["sales", "profit", "quantity", "product", "category", "region", "date"]
found = [f"{r} → `{detected[r]}`" for r in role_names if detected.get(r)]
if found:
    st.caption("Auto-detected columns: " + ", ".join(found))
else:
    st.caption(
        "No standard business columns (sales, profit, region...) were detected, "
        "so generic numeric and categorical columns are used instead."
    )

if summary["kpis"]:
    for kpi in summary["kpis"]:
        st.markdown(f"**{kpi['label']}** (`{kpi['column']}`)")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total", fmt(kpi["total"]))
        k2.metric("Average", fmt(kpi["average"]))
        k3.metric("Median", fmt(kpi["median"]))
        k4.metric("Max", fmt(kpi["max"]))
    if summary["profit_margin_pct"] is not None:
        st.metric("Overall profit margin", f"{summary['profit_margin_pct']}%")
else:
    st.info("No numeric columns available for KPI calculation.")

if summary["group_comparisons"]:
    st.subheader("Group comparisons")
    for g in summary["group_comparisons"]:
        st.markdown(f"**{g['measure']} by `{g['group_column']}`** ({g['number_of_groups']} groups)")
        a, b = st.columns(2)
        a.caption("Top groups")
        a.dataframe(pd.DataFrame(g["top"]), hide_index=True, use_container_width=True)
        if g["bottom"]:
            b.caption("Lowest groups")
            b.dataframe(pd.DataFrame(g["bottom"]), hide_index=True, use_container_width=True)

trend = summary["time_trend"]
if trend:
    st.subheader("Time trend")
    t1, t2, t3 = st.columns(3)
    t1.metric(f"First {trend['granularity']} period", fmt(trend["first_value"]), trend["first_period"])
    t2.metric(f"Last {trend['granularity']} period", fmt(trend["last_value"]), trend["last_period"])
    t3.metric("Change", f"{trend['change_pct']}%" if trend["change_pct"] is not None else "n/a")
    st.caption(
        f"Best period: {trend['best_period']} | Worst period: {trend['worst_period']}. "
        "The first and last periods may be incomplete."
    )

if summary["outliers"]:
    st.subheader("Unusual values (outliers, IQR method)")
    st.dataframe(pd.DataFrame(summary["outliers"]), hide_index=True, use_container_width=True)

if summary["correlations"]:
    st.subheader("Strong correlations between numeric columns")
    st.dataframe(pd.DataFrame(summary["correlations"]), hide_index=True, use_container_width=True)
    st.caption("Correlation does not prove that one column causes the other.")

# ---------- 5. Visualizations ----------

st.header("4. Visualizations")
try:
    charts = create_visualizations(df, summary)
except Exception as e:
    charts = []
    st.warning(f"Could not create charts: {e}")

if not charts:
    st.info("No suitable columns were found for automatic charts.")
else:
    cols = st.columns(2)
    for i, (title, fig) in enumerate(charts):
        with cols[i % 2]:
            st.pyplot(fig)
            plt.close(fig)

# ---------- 6. AI insights ----------

st.header("5. AI Business Insights")
st.caption(
    f"Pandas calculated every number above. {MODEL_NAME} only interprets those verified findings."
)

api_key = get_api_key()
if not api_key:
    st.warning(
        "GROQ_API_KEY is missing. Create a `.env` file (see `.env.example`) or add it to "
        "Streamlit secrets, then restart the app. The analysis above still works without it."
    )

if st.button("✨ Generate AI Insights", disabled=not api_key):
    try:
        with st.spinner("Asking the AI to interpret the findings..."):
            st.session_state["insights"] = generate_ai_insights(summary)
    except AIAnalysisError as e:
        st.error(str(e))

# AI Insights section
if st.session_state.get("insights"):
    st.markdown(st.session_state["insights"].replace("$", r"\$"))

# ---------- 7. Ask the data (simple optional feature) ----------

st.header("6. Ask the Data (optional)")
st.caption("Questions are answered only from the computed analysis summary, not from the raw rows.")
question = st.text_input("Ask a question, e.g. 'Which region performs best?'")

if st.button("Ask", disabled=not api_key) and question.strip():
    try:
        with st.spinner("Thinking..."):
            st.session_state["answer"] = answer_question(question, summary)
    except AIAnalysisError as e:
        st.error(str(e))

# Ask the Data section
if st.session_state.get("answer"):
    st.markdown(st.session_state["answer"].replace("$", r"\$"))