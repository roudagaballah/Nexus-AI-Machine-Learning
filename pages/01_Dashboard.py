"""
Nexus AI — Page 01: Executive Dashboard & Command Center
--------------------------------------------------------
High-level telemetry, dataset health index, workflow progression matrix,
and instant stage navigation.
"""

import streamlit as st
import pandas as pd
from utils.session_state import init_session_state, get_active_data, get_dataset_telemetry, get_workflow_status, load_sample_dataset
from utils.ui import inject_global_css, render_hero, render_section_header, render_metric_grid, render_step_navigation, render_sidebar_status, render_html


def main():
    init_session_state()
    inject_global_css()
    render_sidebar_status()

    # Hero Banner
    render_hero(
        badge="Executive Overview",
        title="Machine Learning Studio & Analytics Hub",
        subtitle="End-to-end autonomous data intelligence, preprocessing telemetry, and model lifecycle management."
    )

    df = get_active_data()
    t = get_dataset_telemetry(df)
    stages = get_workflow_status()
    done_count = sum(1 for s in stages if s["done"])
    progress_pct = int((done_count / len(stages)) * 100)

    # Top KPI Metrics Grid
    if df is not None:
        metrics = [
            {"icon": "database", "label": "Total Observations", "value": f"{t['rows']:,}", "sub": "Active records in memory", "delta": "Active", "delta_type": "pos"},
            {"icon": "table", "label": "Feature Space", "value": f"{t['cols']:,}", "sub": f"{t['numeric_count']} Numeric · {t['categorical_count']} Categorical"},
            {"icon": "alert-triangle", "label": "Missing Value Rate", "value": f"{t['missing_pct']:.1f}%", "sub": f"{t['missing_count']:,} null cells", "delta_type": "neg" if t['missing_pct'] > 0 else "pos"},
            {"icon": "activity", "label": "Pipeline Progress", "value": f"{progress_pct}%", "sub": f"{done_count} of {len(stages)} stages completed", "delta": f"Step {done_count+1}", "delta_type": "pos"},
        ]
    else:
        metrics = [
            {"icon": "database", "label": "Dataset Connection", "value": "Awaiting Ingestion", "sub": "No dataset loaded in memory"},
            {"icon": "table", "label": "Feature Space", "value": "0", "sub": "Ready for upload"},
            {"icon": "shield", "label": "Quality Score", "value": "N/A", "sub": "Pending dataset upload"},
            {"icon": "activity", "label": "Pipeline Progress", "value": "0%", "sub": "Upload data to begin"},
        ]

    render_metric_grid(metrics)

    # Quick Start or Active Dataset Card
    if df is None:
        with st.container(border=True):
            render_section_header("01", "Fast-Start Benchmark Datasets")
            render_html(
                """
                <div style="color: var(--text-secondary); font-size: 13.5px; margin-bottom: 14px;">
                    No active dataset detected in memory. Upload your CSV/Excel file or initialize with an instant benchmark dataset:
                </div>
                """
            )
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.button("Iris Flowers (Classification)", use_container_width=True):
                    load_sample_dataset("iris")
                    st.rerun()
            with c2:
                if st.button("Titanic Survival (Classification)", use_container_width=True):
                    load_sample_dataset("titanic")
                    st.rerun()
            with c3:
                if st.button("Wine Quality (Classification)", use_container_width=True):
                    load_sample_dataset("wine")
                    st.rerun()
            with c4:
                if st.button("California Housing (Regression)", use_container_width=True):
                    load_sample_dataset("california")
                    st.rerun()

    else:
        # Dataset Health & Telemetry Breakdown
        with st.container(border=True):
            render_section_header("01", "Dataset Health & Readiness Index")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Health Score", f"{t['health_score']}/100", delta="Ready" if t['health_score'] >= 80 else "Needs Cleaning")
            c2.metric("Duplicate Rows", f"{t['duplicate_count']:,}", delta=f"{t['duplicate_pct']:.1f}% rate", delta_color="inverse")
            c3.metric("Memory Usage", f"{t['memory_mb']:.2f} MB")
            c4.metric("Active Model", st.session_state.get("model_name", "None"))

    # Interactive Pipeline Stages Grid
    render_html("<br>")
    with st.container(border=True):
        render_section_header("02", "Pipeline Stages & Workflow Progression")
        render_html(
            """
            <div style="color: var(--text-secondary); font-size: 13.5px; margin-bottom: 16px;">
                Status matrix across the 12 workflow stages. Click any stage to open its dedicated workspace:
            </div>
            """
        )

        cols = st.columns(4)
        for idx, stage in enumerate(stages):
            col = cols[idx % 4]
            with col:
                status_text = "Completed" if stage["done"] else "Pending"
                status_color = "var(--success)" if stage["done"] else "var(--text-muted)"
                status_bg = "var(--success-subtle)" if stage["done"] else "var(--bg-surface-elevated)"
                render_html(
                    f"""
                    <div style="background: var(--bg-surface-elevated); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 12px 14px; margin-bottom: 10px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-weight: 600; font-size: 13px; color: var(--text-primary);">{stage['name']}</span>
                            <span style="color: {status_color}; background: {status_bg}; padding: 1px 6px; border-radius: 4px; font-size: 10.5px; font-weight: 600; text-transform: uppercase;">{status_text}</span>
                        </div>
                    </div>
                    """
                )
                if st.button(f"Open {stage['name']}", key=f"btn_jump_{stage['id']}", use_container_width=True, type="secondary"):
                    st.switch_page(stage["page"])

    # Bottom Step Navigation
    render_step_navigation(
        prev_page=None,
        next_page="pages/02_Upload.py",
        next_label="Begin Ingestion"
    )


if __name__ == "__main__":
    main()
