"""
Nexus AI — Page 02: Dataset Ingestion & Telemetry
-------------------------------------------------
High-performance multi-format dataset ingestion, instant benchmark loader,
schema introspection, and telemetry initialization.
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.session_state import init_session_state, get_active_data, get_dataset_telemetry, reset_workflow, load_sample_dataset
from utils.ui import inject_global_css, render_hero, render_section_header, render_metric_grid, render_step_navigation, render_sidebar_status, render_html


def main():
    init_session_state()
    inject_global_css()
    render_sidebar_status()

    render_hero(
        badge="Data Ingestion",
        title="Dataset Ingestion & Schema Specification",
        subtitle="Connect your raw CSV or Excel dataset to initialize the Nexus AI end-to-end Machine Learning pipeline."
    )

    # Ingestion Container
    with st.container(border=True):
        render_section_header("01", "Data Source Connection")
        
        tab_upload, tab_sample = st.tabs(["Upload File (CSV / XLSX)", "Benchmark Datasets"])

        with tab_upload:
            uploaded_file = st.file_uploader(
                "Upload Raw Dataset",
                type=["csv", "xlsx", "xls"],
                help="Supported file extensions: .csv, .xlsx, .xls",
                label_visibility="collapsed"
            )

            if uploaded_file is not None:
                current_key = (uploaded_file.name, uploaded_file.size)
                if current_key != st.session_state.get("file_key"):
                    try:
                        with st.spinner("Parsing dataset and introspecting schema..."):
                            if uploaded_file.name.lower().endswith(".csv"):
                                df = pd.read_csv(uploaded_file)
                            else:
                                df = pd.read_excel(uploaded_file)

                            st.session_state.original_data = df.copy()
                            st.session_state.data = df.copy()
                            st.session_state.file_name = uploaded_file.name
                            st.session_state.dataset_name = uploaded_file.name.rsplit(".", 1)[0]
                            st.session_state.file_key = current_key
                            reset_workflow(keep_original_file=True)
                            st.success(f"Connected successfully: {uploaded_file.name} ({len(df):,} observations)")
                    except Exception as e:
                        st.error(f"Failed to parse uploaded file: {e}")
                else:
                    st.info(f"Active dataset: {uploaded_file.name}")

        with tab_sample:
            render_html(
                """
                <div style="color: var(--text-secondary); font-size: 13.5px; margin-bottom: 14px;">
                    Select a standard machine learning benchmark dataset for immediate evaluation:
                </div>
                """
            )
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.button("Iris Flowers (Classification)", use_container_width=True):
                    load_sample_dataset("iris")
                    st.success("Loaded Iris Dataset")
                    st.rerun()
            with c2:
                if st.button("Titanic Survival (Classification)", use_container_width=True):
                    load_sample_dataset("titanic")
                    st.success("Loaded Titanic Dataset")
                    st.rerun()
            with c3:
                if st.button("Wine Quality (Classification)", use_container_width=True):
                    load_sample_dataset("wine")
                    st.success("Loaded Wine Dataset")
                    st.rerun()
            with c4:
                if st.button("California Housing (Regression)", use_container_width=True):
                    load_sample_dataset("california")
                    st.success("Loaded California Housing Dataset")
                    st.rerun()

    # Active Dataset Telemetry & Previews
    df = get_active_data()
    if df is not None:
        t = get_dataset_telemetry(df)
        
        render_html("<br>")
        render_section_header("02", "Ingestion Telemetry Summary")
        
        metrics = [
            {"icon": "database", "label": "Total Observations", "value": f"{t['rows']:,}", "sub": "Total records"},
            {"icon": "table", "label": "Feature Space", "value": f"{t['cols']}", "sub": f"{t['numeric_count']} Numeric · {t['categorical_count']} Categorical"},
            {"icon": "alert-triangle", "label": "Missing Values", "value": f"{t['missing_count']:,}", "sub": f"{t['missing_pct']:.1f}% missing rate"},
            {"icon": "repeat", "label": "Duplicate Rows", "value": f"{t['duplicate_count']:,}", "sub": f"{t['duplicate_pct']:.1f}% duplicate rate"},
        ]
        render_metric_grid(metrics)

        # Data Inspection Tables
        with st.container(border=True):
            render_section_header("03", "Dataset Preview")
            st.dataframe(df.head(15), use_container_width=True, hide_index=False)

        with st.container(border=True):
            render_section_header("04", "Feature Schema & Types")
            schema_data = []
            for col in df.columns:
                n_null = int(df[col].isnull().sum())
                pct_null = (n_null / len(df)) * 100
                n_uniq = int(df[col].nunique(dropna=False))
                dtype_str = str(df[col].dtype)
                sample_val = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "None"
                schema_data.append({
                    "Feature Name": col,
                    "Data Type": dtype_str,
                    "Distinct Values": n_uniq,
                    "Missing Count": n_null,
                    "Missing Rate": f"{pct_null:.1f}%",
                    "Sample Value": sample_val,
                })
            schema_df = pd.DataFrame(schema_data)
            st.dataframe(schema_df, use_container_width=True, hide_index=True)

    # Navigation Footer
    render_step_navigation(
        prev_page="pages/01_Dashboard.py",
        next_page="pages/03_Data_Overview.py",
        prev_label="Dashboard",
        next_label="Data Overview"
    )


if __name__ == "__main__":
    main()
