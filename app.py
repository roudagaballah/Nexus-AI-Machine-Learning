"""
========================================================================================
NEXUS AI — Enterprise Machine Learning & Data Intelligence Platform
========================================================================================
Main application entry point orchestrating multi-page workflow navigation,
global session state synchronization, and enterprise SaaS design architecture.
"""

import streamlit as st
from utils.session_state import init_session_state
from utils.ui import inject_global_css

# ============================================================
# GLOBAL APP CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Nexus AI — Machine Learning Platform",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Global Shared State & Inject Global CSS
init_session_state()
inject_global_css()

# ============================================================
# STRUCTURED WORKFLOW NAVIGATION HIERARCHY
# ============================================================
pages = {
    "Workspace Hub": [
        st.Page("pages/01_Dashboard.py", title="Executive Dashboard", icon=":material/dashboard:", default=True),
        st.Page("pages/02_Upload.py", title="Dataset Ingestion", icon=":material/upload_file:"),
        st.Page("pages/03_Data_Overview.py", title="Data Overview", icon=":material/table_chart:"),
    ],
    "Data Preprocessing": [
        st.Page("pages/04_Cleaning.py", title="Data Cleaning & Deduplication", icon=":material/auto_fix_high:"),
        st.Page("pages/05_Missing_Values.py", title="Missing Values Lab", icon=":material/find_replace:"),
        st.Page("pages/06_Outliers.py", title="Outlier Detection", icon=":material/crisis_alert:"),
    ],
    "Feature Engineering": [
        st.Page("pages/07_Encoding.py", title="Categorical Encoding", icon=":material/tag:"),
        st.Page("pages/08_Feature_Engineering.py", title="Feature Engineering & PCA", icon=":material/alt_route:"),
        st.Page("pages/09_Scaling.py", title="Feature Scaling", icon=":material/tune:"),
    ],
    "Visual Intelligence": [
        st.Page("pages/10_Visualization.py", title="Visual Analytics & 3D", icon=":material/scatter_plot:"),
    ],
    "Machine Learning": [
        st.Page("pages/11_Modeling.py", title="Model Studio & Training", icon=":material/psychology:"),
        st.Page("pages/12_Evaluation.py", title="Evaluation & Live Testing", icon=":material/check_circle:"),
    ],
    "Artifacts & Export": [
        st.Page("pages/14_Export.py", title="Export Center & Audit Report", icon=":material/download:"),
    ],
}

# Orchestrate and execute active page
pg = st.navigation(pages)
pg.run()
