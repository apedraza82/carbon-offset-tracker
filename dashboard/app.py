"""Streamlit dashboard for Carbon Offset Tracker.

Run locally: streamlit run dashboard/app.py
Deploy: Connect GitHub repo to Streamlit Cloud (free tier)
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page config
st.set_page_config(
    page_title="Carbon Offset Tracker",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent.parent / "data"


@st.cache_data(ttl=3600)
def load_data():
    """Load matched retirements data."""
    parquet_path = DATA_DIR / "matched_retirements.parquet"
    csv_path = DATA_DIR / "matched_retirements.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        return None

    # Parse dates if available
    for col in ["ret_date", "retirementdate", "Retirement/Cancellation Date", "Retirement Date"]:
        if col in df.columns:
            df["retirement_date"] = pd.to_datetime(df[col], errors="coerce")
            df["retirement_year"] = df["retirement_date"].dt.year
            break

    return df


@st.cache_data(ttl=3600)
def load_stats():
    """Load summary statistics."""
    stats_path = DATA_DIR / "summary_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            return json.load(f)
    return None


def main():
    st.title("Carbon Offset Tracker")
    st.markdown(
        "**Corporate Carbon Offset Retirements Matched to Listed Firms** "
        "| [Paper](https://example.com) | [GitHub](https://github.com/carbon-offset-tracker)"
    )

    df = load_data()
    if df is None:
        st.error("No data found. Run the pipeline first: `python -m src.pipeline --skip-download --skip-llm`")
        return

    stats = load_stats()

    # --- Sidebar Filters ---
    st.sidebar.header("Filters")

    # Registry filter
    registries = sorted(df["registry"].dropna().unique()) if "registry" in df.columns else []
    selected_registries = st.sidebar.multiselect("Registry", registries, default=registries)

    # Year filter
    if "retirement_year" in df.columns:
        years = sorted(df["retirement_year"].dropna().unique())
        if years:
            year_range = st.sidebar.slider(
                "Retirement Year",
                min_value=int(min(years)),
                max_value=int(max(years)),
                value=(int(min(years)), int(max(years))),
            )
        else:
            year_range = None
    else:
        year_range = None

    # Match status filter
    match_filter = st.sidebar.radio("Match Status", ["All", "Matched only", "Unmatched only"])

    # Country filter
    country_col = next((c for c in ["country", "Country/Area", "Country"] if c in df.columns), None)
    if country_col:
        countries = sorted(df[country_col].dropna().unique())
        selected_countries = st.sidebar.multiselect("Country (project)", countries[:50], default=[])

    # Firm search
    firm_search = st.sidebar.text_input("Search firm name")

    # --- Apply Filters ---
    mask = pd.Series(True, index=df.index)

    if selected_registries and "registry" in df.columns:
        mask &= df["registry"].isin(selected_registries)

    if year_range and "retirement_year" in df.columns:
        mask &= df["retirement_year"].between(*year_range)

    if match_filter == "Matched only":
        mask &= df["factset_entity_id"].notna() & (df["factset_entity_id"] != "")
    elif match_filter == "Unmatched only":
        mask &= df["factset_entity_id"].isna() | (df["factset_entity_id"] == "")

    if country_col and selected_countries:
        mask &= df[country_col].isin(selected_countries)

    if firm_search:
        name_col = "matched_name" if "matched_name" in df.columns else "raw_beneficiary"
        mask &= df[name_col].str.contains(firm_search, case=False, na=False)

    filtered = df[mask]

    # --- Key Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Retirements", f"{len(filtered):,}")
    matched_count = (filtered["factset_entity_id"].notna() & (filtered["factset_entity_id"] != "")).sum()
    col2.metric("Matched to Firms", f"{matched_count:,}")
    col3.metric("Unique Firms", f"{filtered['factset_entity_id'].nunique():,}" if "factset_entity_id" in filtered.columns else "N/A")
    if "quantity" in filtered.columns:
        col4.metric("Total Credits", f"{filtered['quantity'].sum():,.0f}")
    elif "Quantity" in filtered.columns:
        col4.metric("Total Credits", f"{filtered['Quantity'].sum():,.0f}")

    if stats:
        st.caption(f"Last updated: {stats.get('last_updated', 'unknown')}")

    # --- Charts ---
    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)

    # Retirements over time
    with chart_col1:
        st.subheader("Retirements Over Time")
        if "retirement_year" in filtered.columns:
            yearly = filtered.groupby(["retirement_year", "registry"]).size().reset_index(name="count")
            fig = px.bar(yearly, x="retirement_year", y="count", color="registry",
                         labels={"retirement_year": "Year", "count": "Retirements"},
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # By registry
    with chart_col2:
        st.subheader("By Registry")
        if "registry" in filtered.columns:
            reg_counts = filtered["registry"].value_counts().reset_index()
            reg_counts.columns = ["Registry", "Count"]
            fig = px.pie(reg_counts, values="Count", names="Registry",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # Geographic distribution
    if country_col and len(filtered) > 0:
        st.subheader("Geographic Distribution (Top 20 Countries)")
        top_countries = filtered[country_col].value_counts().head(20).reset_index()
        top_countries.columns = ["Country", "Retirements"]
        fig = px.bar(top_countries, x="Country", y="Retirements",
                     color_discrete_sequence=["#2E86AB"])
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Top firms
    if matched_count > 0:
        st.subheader("Top 20 Firms by Retirement Count")
        name_col = "matched_name" if "matched_name" in filtered.columns else "raw_beneficiary"
        matched_df = filtered[filtered["factset_entity_id"].notna() & (filtered["factset_entity_id"] != "")]
        top_firms = matched_df[name_col].value_counts().head(20).reset_index()
        top_firms.columns = ["Firm", "Retirements"]
        fig = px.bar(top_firms, x="Retirements", y="Firm", orientation="h",
                     color_discrete_sequence=["#A23B72"])
        fig.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # --- Data Table ---
    st.markdown("---")
    st.subheader("Data Explorer")

    display_cols = [c for c in [
        "raw_beneficiary", "matched_name", "factset_entity_id", "registry",
        "retirement_year", "country", "Country/Area", "quantity", "Quantity",
        "match_confidence", "match_method",
    ] if c in filtered.columns]

    st.dataframe(filtered[display_cols].head(500), use_container_width=True, height=400)

    # --- Download ---
    st.markdown("---")
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        csv_data = filtered[display_cols].to_csv(index=False)
        st.download_button(
            label="Download Filtered Data (CSV)",
            data=csv_data,
            file_name="carbon_offset_retirements_filtered.csv",
            mime="text/csv",
        )

    with col_dl2:
        full_csv = df.to_csv(index=False)
        st.download_button(
            label="Download Full Dataset (CSV)",
            data=full_csv,
            file_name="carbon_offset_retirements_full.csv",
            mime="text/csv",
        )

    # --- Citation ---
    st.markdown("---")
    st.subheader("Citation")
    st.info(
        "If you use this dataset in your research, please cite:\n\n"
        "> Pedraza, A., Williams, T., & Zeni, F. (2025). "
        "\"Local Visibility vs. Global Integrity: Evidence from Corporate Carbon Offset Portfolios.\" "
        "Working Paper.\n\n"
        "Data source: [Berkeley Carbon Trading Project](https://gspp.berkeley.edu/research/osf-bctp/offsets-database). "
        "Firm matching via [FactSet](https://www.factset.com/)."
    )


if __name__ == "__main__":
    main()
