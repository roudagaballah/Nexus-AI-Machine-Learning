"""
Nexus AI — Page 05: Missing Value Imputation Lab
-----------------------------------------------
Multi-strategy missing values resolution: Simple, KNN, Iterative MICE,
constant assignment, and row/column pruning.
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.session_state import init_session_state, get_active_data, set_active_data, get_dataset_telemetry
from utils.ui import inject_global_css, render_hero, render_section_header, render_metric_grid, render_before_after, render_step_navigation, render_sidebar_status, dataset_guard, render_html
from utils.preprocessing import drop_missing_values, impute_missing_values
from utils.visualization import plot_missing_heatmap, plot_missing_matrix


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
        title="Missing Value Imputation Lab",
        subtitle="Detect, analyze missingness patterns, and apply statistical or machine learning imputation models."
    )

    # Missing Stats
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0].index.tolist()

    metrics = [
        {"icon": "alert-triangle", "label": "Missing Cells", "value": f"{t['missing_count']:,}", "sub": f"{t['missing_pct']:.1f}% missing rate", "delta_type": "neg" if t['missing_count'] > 0 else "pos"},
        {"icon": "table", "label": "Affected Features", "value": f"{len(null_cols)}", "sub": f"Out of {t['cols']} features"},
        {"icon": "database", "label": "Complete Observations", "value": f"{len(df.dropna()):,}", "sub": f"{(len(df.dropna())/len(df)*100):.1f}% completeness"},
        {"icon": "activity", "label": "Imputation Status", "value": "Imputed" if st.session_state.get("missing_done") else ("Clean" if t['missing_count'] == 0 else "Pending"), "sub": "Pipeline stage 05", "delta_type": "pos"},
    ]
    render_metric_grid(metrics)

    # Visual Matrix & Imputation Lab
    if t["missing_count"] > 0:
        with st.container(border=True):
            render_section_header("01", "Missingness Distribution & Heatmap")
            st.plotly_chart(plot_missing_heatmap(df), use_container_width=True)

        # Imputation Lab
        with st.container(border=True):
            render_section_header("02", "Imputation Strategy Configuration")

            tab_global, tab_column, tab_drop = st.tabs(["Global Strategy", "Feature-Specific Strategy", "Prune Observations & Features"])

            with tab_global:
                st.markdown("##### Uniform Imputation Strategy")
                st.caption("Apply a standardized statistical or ML imputation across all affected features:")
                strat = st.selectbox(
                    "Select Technique:",
                    [
                        "Mean (Numerical) / Mode (Categorical)",
                        "Median (Numerical - Outlier Robust) / Mode (Categorical)",
                        "Most Frequent / Mode (All features)",
                        "KNN Imputer (K-Nearest Neighbors Multi-Feature Estimation)",
                        "Iterative MICE Imputer (Multivariate Chained Equations)",
                    ],
                    key="global_impute_strat"
                )

                k_val = 5
                m_iter = 10
                if "KNN" in strat:
                    k_val = st.slider("KNN Number of Neighbors (k):", 1, 15, 5)
                elif "Iterative" in strat:
                    m_iter = st.slider("Max MICE Iterations:", 5, 30, 10)

                if st.button("Apply Global Imputation", use_container_width=True, key="btn_apply_global_impute", type="primary"):
                    try:
                        num_null_cols = [c for c in null_cols if pd.api.types.is_numeric_dtype(df[c])]
                        cat_null_cols = [c for c in null_cols if not pd.api.types.is_numeric_dtype(df[c])]

                        df_res = df.copy()
                        all_imputers = {}

                        if "Mean" in strat:
                            if num_null_cols:
                                df_res, imp1 = impute_missing_values(df_res, num_null_cols, strategy="mean")
                                all_imputers.update(imp1)
                            if cat_null_cols:
                                df_res, imp2 = impute_missing_values(df_res, cat_null_cols, strategy="most_frequent")
                                all_imputers.update(imp2)
                        elif "Median" in strat:
                            if num_null_cols:
                                df_res, imp1 = impute_missing_values(df_res, num_null_cols, strategy="median")
                                all_imputers.update(imp1)
                            if cat_null_cols:
                                df_res, imp2 = impute_missing_values(df_res, cat_null_cols, strategy="most_frequent")
                                all_imputers.update(imp2)
                        elif "Most Frequent" in strat:
                            df_res, imp_all = impute_missing_values(df_res, null_cols, strategy="most_frequent")
                            all_imputers.update(imp_all)
                        elif "KNN" in strat:
                            if num_null_cols:
                                df_res, imp1 = impute_missing_values(df_res, num_null_cols, strategy="knn", n_neighbors=k_val)
                                all_imputers.update(imp1)
                            if cat_null_cols:
                                df_res, imp2 = impute_missing_values(df_res, cat_null_cols, strategy="most_frequent")
                                all_imputers.update(imp2)
                        elif "Iterative" in strat:
                            if num_null_cols:
                                df_res, imp1 = impute_missing_values(df_res, num_null_cols, strategy="iterative", max_iter=m_iter)
                                all_imputers.update(imp1)
                            if cat_null_cols:
                                df_res, imp2 = impute_missing_values(df_res, cat_null_cols, strategy="most_frequent")
                                all_imputers.update(imp2)

                        st.session_state.missing_done = True
                        if st.session_state.get("preprocessing_pipeline"):
                            st.session_state.preprocessing_pipeline.record_imputation(strat, null_cols, all_imputers)
                        set_active_data(df_res, stage_name="Missing Values", log_entry=f"Applied global imputation ({strat}) on {len(null_cols)} columns.")
                        st.success("Global imputation applied successfully.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Imputation error: {e}")

            with tab_column:
                st.markdown("##### Feature-Specific Imputation")
                st.caption("Configure custom imputation strategies per individual feature:")
                col_plans = {}
                cols_grid = st.columns(min(3, max(1, len(null_cols)))) if null_cols else [st.container()]
                for idx, col in enumerate(null_cols):
                    c = cols_grid[idx % len(cols_grid)]
                    with c:
                        is_num = pd.api.types.is_numeric_dtype(df[col])
                        opts = ["mean", "median", "most_frequent", "constant"] if is_num else ["most_frequent", "constant"]
                        sel_strat = st.selectbox(f"{col} ({'Numeric' if is_num else 'Categorical'} · {df[col].isnull().sum()} nulls):", opts, key=f"col_imp_{col}")
                        col_plans[col] = sel_strat

                if st.button("Apply Feature-Specific Imputation", use_container_width=True, key="btn_apply_col_impute", type="primary"):
                    try:
                        df_res = df.copy()
                        all_imputers = {}
                        for col, s in col_plans.items():
                            df_res, imp = impute_missing_values(df_res, [col], strategy=s, fill_value="Missing")
                            all_imputers.update(imp)
                        st.session_state.missing_done = True
                        if st.session_state.get("preprocessing_pipeline"):
                            st.session_state.preprocessing_pipeline.record_imputation("column_specific", list(col_plans.keys()), all_imputers)
                        set_active_data(df_res, stage_name="Missing Values", log_entry=f"Applied custom column imputation on {len(col_plans)} columns.")
                        st.success("Feature-specific imputation applied successfully.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Column imputation error: {e}")

            with tab_drop:
                st.markdown("##### Prune Incomplete Observations & High-Missing Features")
                c_drop1, c_drop2 = st.columns(2)
                with c_drop1:
                    st.markdown("**Drop Rows with Missing Values**")
                    drop_mode = st.radio("Condition:", ["Any value is missing", "All values are missing"], key="drop_mode_radio")
                    how_val = "any" if "Any" in drop_mode else "all"
                    if st.button("Drop Incomplete Rows", use_container_width=True):
                        df_dropped = drop_missing_values(df, axis=0, how=how_val)
                        removed = len(df) - len(df_dropped)
                        st.session_state.missing_done = True
                        set_active_data(df_dropped, stage_name="Missing Values", log_entry=f"Dropped {removed:,} incomplete rows (mode: {how_val}).")
                        st.success(f"Dropped {removed:,} incomplete rows.")
                        st.rerun()

                with c_drop2:
                    st.markdown("**Drop Features with High Missing Rate**")
                    pct_thresh = st.slider("Missing Threshold % to Drop Feature:", 10, 90, 50)
                    high_null_cols = [c for c in df.columns if (df[c].isnull().sum() / len(df) * 100) >= pct_thresh]
                    if high_null_cols:
                        st.warning(f"Features exceeding {pct_thresh}% missing rate: {', '.join(high_null_cols)}")
                        if st.button(f"Drop {len(high_null_cols)} High-Missing Features", use_container_width=True):
                            df_dropped_cols = df.drop(columns=high_null_cols).copy()
                            st.session_state.missing_done = True
                            if st.session_state.get("preprocessing_pipeline"):
                                st.session_state.preprocessing_pipeline.record_dropped_columns(high_null_cols)
                            set_active_data(df_dropped_cols, stage_name="Missing Values", log_entry=f"Dropped high-missing features (> {pct_thresh}%): {', '.join(high_null_cols)}")
                            st.success(f"Dropped {len(high_null_cols)} features.")
                            st.rerun()
                    else:
                        st.info(f"No features exceed the {pct_thresh}% threshold.")

    else:
        with st.container(border=True):
            render_section_header("01", "Dataset Completeness Audit")
            st.success("✓ Complete Data: No missing cells detected in the active dataset.")
            
            # Completeness details table
            comp_data = []
            for col in df.columns:
                comp_data.append({
                    "Feature": col,
                    "Data Type": str(df[col].dtype),
                    "Total Count": len(df[col]),
                    "Missing Count": 0,
                    "Completeness": "100.0%"
                })
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    # Before / After Matrix
    orig_df = st.session_state.get("original_data")
    if orig_df is not None:
        render_html("<br>")
        render_section_header("03", "Missing Values Resolution Audit")
        render_before_after(orig_df, df, note="Telemetry tracking of missing value remediation.")

    # Navigation Footer
    render_step_navigation(
        prev_page="pages/04_Cleaning.py",
        next_page="pages/06_Outliers.py",
        prev_label="Data Cleaning",
        next_label="Outlier Detection"
    )


if __name__ == "__main__":
    main()
