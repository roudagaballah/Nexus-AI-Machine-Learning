"""
Nexus AI — Page 04: Data Cleaning & Duplicate Management
--------------------------------------------------------
Sanitize observations, purge duplicate rows, eliminate constant zero-variance features,
and remove redundant columns.
"""

import streamlit as st
import pandas as pd
from utils.session_state import init_session_state, get_active_data, set_active_data, get_dataset_telemetry
from utils.ui import inject_global_css, render_hero, render_section_header, render_metric_grid, render_before_after, render_step_navigation, render_sidebar_status, dataset_guard, render_html
from utils.preprocessing import remove_duplicates, drop_unwanted_columns, drop_constant_columns


def main():
    init_session_state()
    inject_global_css()
    render_sidebar_status()

    if not dataset_guard():
        return

    df = get_active_data()
    t = get_dataset_telemetry(df)

    render_hero(
        badge="Data Preprocessing",
        title="Data Cleaning & Deduplication",
        subtitle="Eliminate duplicate records, remove uninformative zero-variance features, and prune unwanted columns."
    )

    # Telemetry KPI Grid
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    metrics = [
        {"icon": "repeat", "label": "Identified Duplicates", "value": f"{t['duplicate_count']:,}", "sub": f"{t['duplicate_pct']:.1f}% duplicate rate", "delta_type": "neg" if t['duplicate_count'] > 0 else "pos"},
        {"icon": "alert-triangle", "label": "Constant Features", "value": f"{len(constant_cols)}", "sub": "Zero-variance columns"},
        {"icon": "database", "label": "Active Observations", "value": f"{t['rows']:,}", "sub": "Current row count"},
        {"icon": "activity", "label": "Cleaning Status", "value": "Cleaned" if st.session_state.get("cleaning_done") else "Ready", "sub": "Pipeline stage 04", "delta_type": "pos"},
    ]
    render_metric_grid(metrics)

    # 1. Duplicate Removal Section
    with st.container(border=True):
        render_section_header("01", "Duplicate Observations Removal")
        render_html(
            """
            <div style="color: var(--text-secondary); font-size: 13.5px; margin-bottom: 14px;">
                Identify and remove duplicate entries based on all features or a selected column subset:
            </div>
            """
        )

        col_dup1, col_dup2 = st.columns([1, 1])
        with col_dup1:
            keep_mode_label = st.radio(
                "Retention Strategy:",
                ["Keep First Occurrence", "Keep Last Occurrence", "Drop All Occurrences"],
                index=0,
                horizontal=False,
                key="dup_keep_mode"
            )
            keep_map = {
                "Keep First Occurrence": "first",
                "Keep Last Occurrence": "last",
                "Drop All Occurrences": False,
            }
            keep_val = keep_map[keep_mode_label]

        with col_dup2:
            subset_enabled = st.checkbox("Match on specific subset of columns only", value=False)
            selected_subset = []
            if subset_enabled:
                selected_subset = st.multiselect("Select columns for uniqueness matching:", df.columns.tolist(), default=df.columns.tolist()[:3])

        if t["duplicate_count"] > 0 or subset_enabled:
            if st.button("Remove Duplicates", use_container_width=True, key="btn_remove_duplicates"):
                df_cleaned, count_removed = remove_duplicates(df, subset=selected_subset if subset_enabled else None, keep=keep_val)
                st.session_state.duplicates_done = True
                st.session_state.cleaning_done = True
                set_active_data(df_cleaned, stage_name="Cleaning", log_entry=f"Purged {count_removed:,} duplicate rows (strategy: {keep_mode_label})")
                st.success(f"Successfully removed {count_removed:,} duplicate observations.")
                st.rerun()
        else:
            st.info("Zero duplicate observations detected in active dataset.")

    # 2. Column Pruning Section
    render_html("<br>")
    with st.container(border=True):
        render_section_header("02", "Feature Pruning & Constant Feature Drop")
        col_prune1, col_prune2 = st.columns(2)

        with col_prune1:
            st.markdown("##### Drop Unwanted Columns")
            cols_to_drop = st.multiselect("Select columns to permanently remove:", df.columns.tolist(), key="select_cols_to_drop")
            if cols_to_drop:
                if st.button(f"Drop {len(cols_to_drop)} Selected Column(s)", use_container_width=True):
                    df_dropped = drop_unwanted_columns(df, cols_to_drop)
                    st.session_state.cleaning_done = True
                    if st.session_state.get("preprocessing_pipeline"):
                        st.session_state.preprocessing_pipeline.record_dropped_columns(cols_to_drop)
                    set_active_data(df_dropped, stage_name="Cleaning", log_entry=f"Dropped columns: {', '.join(cols_to_drop)}")
                    st.success(f"Dropped columns: {', '.join(cols_to_drop)}")
                    st.rerun()

        with col_prune2:
            st.markdown("##### Constant Zero-Variance Features")
            if constant_cols:
                st.warning(f"Detected constant features: {', '.join(constant_cols)}")
                if st.button(f"Drop {len(constant_cols)} Constant Feature(s)", use_container_width=True):
                    df_const_cleaned, dropped_const = drop_constant_columns(df)
                    st.session_state.cleaning_done = True
                    if st.session_state.get("preprocessing_pipeline"):
                        st.session_state.preprocessing_pipeline.record_dropped_columns(dropped_const)
                    set_active_data(df_const_cleaned, stage_name="Cleaning", log_entry=f"Removed constant zero-variance features: {', '.join(dropped_const)}")
                    st.success(f"Removed constant features: {', '.join(dropped_const)}")
                    st.rerun()
            else:
                st.info("No constant zero-variance features detected.")

    # Before / After Matrix
    orig_df = st.session_state.get("original_data")
    if orig_df is not None:
        render_html("<br>")
        render_section_header("03", "Cleaning Audit Telemetry")
        render_before_after(orig_df, df, note="Comparing original ingestion dataset against current cleaned state.")

    # Navigation Footer
    render_step_navigation(
        prev_page="pages/03_Data_Overview.py",
        next_page="pages/05_Missing_Values.py",
        prev_label="Data Overview",
        next_label="Missing Values Lab"
    )


if __name__ == "__main__":
    main()
