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

    # Filter to matched only
    df = df[df["factset_entity_id"].notna() & (df["factset_entity_id"] != "")].copy()

    return df


@st.cache_data(ttl=3600)
def load_stats():
    """Load summary statistics."""
    stats_path = DATA_DIR / "summary_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            return json.load(f)
    return None


def fmt_tonnes(t):
    """Format tonnes as human-readable string."""
    if t >= 1e6:
        return f"{t/1e6:.1f}M"
    if t >= 1e3:
        return f"{t/1e3:.0f}K"
    return f"{t:,.0f}"


def main():
    st.title("Carbon Offset Tracker")
    st.markdown(
        "**Corporate Carbon Offset Retirements Matched to Listed Firms** "
        "| [Paper](https://example.com) | [GitHub](https://github.com/apedraza82/carbon-offset-tracker)"
    )

    df = load_data()
    if df is None:
        st.error("No data found. Run the pipeline first: `python -m src.pipeline --skip-download --skip-llm`")
        return

    stats = load_stats()

    # Quantity column
    qty_col = "quantity" if "quantity" in df.columns else "Quantity" if "Quantity" in df.columns else None

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

    # Country filter (project)
    country_col = next((c for c in ["country", "Country/Area", "Country"] if c in df.columns), None)
    if country_col:
        countries = sorted(df[country_col].dropna().unique())
        selected_countries = st.sidebar.multiselect("Country (project)", countries, default=[])

    # HQ country filter
    selected_hq = []
    if "hq_country" in df.columns:
        hq_countries = sorted(df["hq_country"].dropna().unique())
        selected_hq = st.sidebar.multiselect("HQ Country (firm)", hq_countries, default=[])

    # Project type filter
    selected_ptypes = []
    if "projecttype" in df.columns:
        ptypes = sorted(df["projecttype"].dropna().unique())
        selected_ptypes = st.sidebar.multiselect("Project Type", ptypes, default=[])

    # Firm search
    firm_search = st.sidebar.text_input("Search firm name")

    # --- Apply Filters ---
    mask = pd.Series(True, index=df.index)

    if selected_registries and "registry" in df.columns:
        mask &= df["registry"].isin(selected_registries)

    if year_range and "retirement_year" in df.columns:
        mask &= df["retirement_year"].between(*year_range)

    if country_col and selected_countries:
        mask &= df[country_col].isin(selected_countries)

    if selected_hq and "hq_country" in df.columns:
        mask &= df["hq_country"].isin(selected_hq)

    if selected_ptypes and "projecttype" in df.columns:
        mask &= df["projecttype"].isin(selected_ptypes)

    if firm_search:
        name_col = "matched_name" if "matched_name" in df.columns else "raw_beneficiary"
        mask &= df[name_col].str.contains(firm_search, case=False, na=False)

    filtered = df[mask]

    # --- Key Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    total_qty = filtered[qty_col].sum() if qty_col else 0
    col1.metric("Total Quantity Retired", fmt_tonnes(total_qty) + " tCO2")
    col2.metric("Transactions", f"{len(filtered):,}")
    col3.metric("Unique Firms", f"{filtered['factset_entity_id'].nunique():,}")
    if country_col:
        col4.metric("Countries", f"{filtered[country_col].nunique():,}")

    if stats:
        st.caption(f"Last updated: {stats.get('last_updated', 'unknown')}")

    # --- Charts ---
    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)

    # Quantity retired over time
    with chart_col1:
        st.subheader("Quantity Retired Over Time")
        if "retirement_year" in filtered.columns and qty_col:
            yearly = filtered.groupby(["retirement_year", "registry"])[qty_col].sum().reset_index()
            yearly[qty_col] = yearly[qty_col] / 1e6  # Convert to MtCO2
            fig = px.bar(yearly, x="retirement_year", y=qty_col, color="registry",
                         labels={"retirement_year": "Year", qty_col: "MtCO2"},
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # By registry
    with chart_col2:
        st.subheader("Quantity by Registry")
        if "registry" in filtered.columns and qty_col:
            reg_qty = filtered.groupby("registry")[qty_col].sum().reset_index()
            reg_qty.columns = ["Registry", "Tonnes"]
            fig = px.pie(reg_qty, values="Tonnes", names="Registry",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # Geographic distribution
    if country_col and qty_col and len(filtered) > 0:
        st.subheader("Top 20 Countries by Quantity Retired (tCO2)")
        top_countries = filtered.groupby(country_col)[qty_col].sum().nlargest(20).reset_index()
        top_countries.columns = ["Country", "Tonnes"]
        fig = px.bar(top_countries, x="Country", y="Tonnes",
                     labels={"Tonnes": "tCO2"},
                     color_discrete_sequence=["#2E86AB"])
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Top firms
    if qty_col:
        st.subheader("Top 20 Firms by Quantity Retired (tCO2)")
        name_col = "matched_name" if "matched_name" in filtered.columns else "raw_beneficiary"
        top_firms = filtered.groupby(name_col)[qty_col].sum().nlargest(20).reset_index()
        top_firms.columns = ["Firm", "Tonnes"]
        fig = px.bar(top_firms, x="Tonnes", y="Firm", orientation="h",
                     labels={"Tonnes": "tCO2"},
                     color_discrete_sequence=["#A23B72"])
        fig.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # --- Data Table (matched only) ---
    st.markdown("---")
    st.subheader("Data Explorer")

    display_cols = [c for c in [
        "matched_name", "factset_entity_id", "hq_country", "registry",
        "retirement_year", "country", "Country/Area", "quantity", "Quantity",
        "projectname", "projecttype", "vintage",
    ] if c in filtered.columns]

    st.dataframe(filtered[display_cols].head(500), use_container_width=True, height=400)

    # --- Download (matched only, no MSCI) ---
    st.markdown("---")
    download_cols = [c for c in display_cols if c not in ("rating_msci", "numrating_msci")]

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        csv_data = filtered[download_cols].to_csv(index=False)
        st.download_button(
            label=f"Download Filtered Data ({len(filtered):,} rows)",
            data=csv_data,
            file_name="carbon_offset_retirements_filtered.csv",
            mime="text/csv",
        )

    with col_dl2:
        full_csv = df[download_cols].to_csv(index=False)
        st.download_button(
            label=f"Download Full Dataset ({len(df):,} rows)",
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
