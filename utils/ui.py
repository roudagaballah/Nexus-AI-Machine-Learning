"""
Nexus AI — Reusable Enterprise UI Components & Navigation System
-----------------------------------------------------------------
High-fidelity visual components, clean design tokens, KPI cards,
Before/After comparison matrices, and unified workflow navigation.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from .session_state import get_workflow_status, get_dataset_telemetry, reset_workflow


# Standardized Lucide-style SVG icons dictionary (16x16, stroke-width 2)
SVG_ICONS = {
    "database": '<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>',
    "table": '<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"></rect><path d="M3 9h18"></path><path d="M3 15h18"></path><path d="M9 3v18"></path><path d="M15 3v18"></path></svg>',
    "layers": '<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>',
    "activity": '<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>',
    "alert-triangle": '<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>',
    "check-circle": '<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>',
    "cpu": '<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><rect x="4" y="4" width="16" height="16" rx="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>',
    "target": '<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>',
    "bar-chart": '<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><line x1="12" y1="20" x2="12" y2="10"></line><line x1="18" y1="20" x2="18" y2="4"></line><line x1="6" y1="20" x2="6" y2="16"></line></svg>',
    "sliders": '<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line></svg>',
    "download": '<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>',
    "shield": '<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>',
    "compass": '<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"></polygon></svg>',
    "hash": '<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><line x1="4" y1="9" x2="20" y2="9"></line><line x1="4" y1="15" x2="20" y2="15"></line><line x1="10" y1="3" x2="8" y2="21"></line><line x1="16" y1="3" x2="14" y2="21"></line></svg>',
    "repeat": '<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><polyline points="17 1 21 5 17 9"></polyline><path d="M3 11V9a4 4 0 0 1 4-4h14"></path><polyline points="7 23 3 19 7 15"></polyline><path d="M21 13v2a4 4 0 0 1-4 4H3"></path></svg>',
    "maximize": '<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path></svg>',
}


def _resolve_icon_svg(icon_key: str) -> str:
    """Map legacy emoji or icon key to clean inline SVG."""
    if icon_key in SVG_ICONS:
        return SVG_ICONS[icon_key]

    mapping = {
        "🧬": "database",
        "📊": "bar-chart",
        "🩹": "alert-triangle",
        "⚡": "activity",
        "📡": "activity",
        "🔢": "hash",
        "🏷️": "layers",
        "🩺": "shield",
        "🔄": "repeat",
        "🛑": "alert-triangle",
        "🧹": "sliders",
        "🔤": "layers",
        "⚖️": "target",
        "⚠️": "alert-triangle",
        "📏": "sliders",
        "🌌": "maximize",
        "🤖": "cpu",
        "🎯": "target",
        "💾": "download",
        "📥": "database",
        "🔍": "compass",
    }
    resolved = mapping.get(icon_key, "activity")
    return SVG_ICONS.get(resolved, SVG_ICONS["activity"])


def render_html(html_content: str):
    """
    Safely render raw HTML in Streamlit without Markdown interpreting
    indented lines as code blocks (<pre><code>).
    Uses native st.html if available, or unindented st.markdown fallback.
    """
    if hasattr(st, "html"):
        st.html(html_content)
    else:
        lines = [line.strip() for line in html_content.strip().splitlines() if line.strip()]
        st.markdown("".join(lines), unsafe_allow_html=True)


def inject_global_css():
    """Inject the centralized Enterprise Design System CSS into the Streamlit app."""
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        render_html(f"<style>{css_content}</style>")


def render_hero(badge: str, title: str, subtitle: str):
    """Render a standardized enterprise SaaS hero header."""
    clean_badge = badge.replace("●", "").replace("⚠️", "").strip()
    html_content = f"""
    <div class="hero-header">
        <span class="hero-badge-tag">{clean_badge}</span>
        <h1 class="hero-title">{title}</h1>
        <p class="hero-subtitle">{subtitle}</p>
    </div>
    """
    render_html(html_content)


def render_section_header(number: str, title: str):
    """Render a clean structured section header."""
    clean_title = title.replace("🎯", "").replace("🤖", "").replace("🔧", "").replace("📂", "").replace("🧪", "").strip()
    html_content = f"""
    <div class="section-header-wrap">
        <span class="section-number-pill">{number}</span>
        <span class="section-title-text">{clean_title}</span>
        <div class="section-divider-line"></div>
    </div>
    """
    render_html(html_content)


def render_metric_grid(metrics: list):
    """
    Render a responsive grid of clean KPI cards.
    Each item in metrics is a dict with: icon, label, value, sub (optional), delta (optional, pos/neg).
    """
    cards_html = []
    for m in metrics:
        icon_raw = m.get("icon", "activity")
        icon_svg = _resolve_icon_svg(icon_raw)
        label = m.get("label", "Metric")
        val = m.get("value", "--")
        sub = m.get("sub", "")
        delta = m.get("delta", None)
        delta_type = m.get("delta_type", "pos")

        delta_html = f'<span class="metric-badge-delta {delta_type}">{delta}</span>' if delta is not None else ""
        sub_html = f'<div class="metric-sub-text">{sub}</div>' if sub else ""

        card_html = f"""
        <div class="metric-card-pro">
            <div class="metric-top-row">
                <span class="metric-label-text">{label}</span>
                <div class="metric-icon-wrap">{icon_svg}</div>
            </div>
            <div class="metric-value-text">{val}</div>
            {sub_html}
        </div>
        """
        cards_html.append(card_html)

    grid_html = f'<div class="metric-grid">{"".join(cards_html)}</div>'
    render_html(grid_html)


def render_before_after(df_before: pd.DataFrame, df_after: pd.DataFrame, note: str = None):
    """Render a side-by-side Before/After KPI comparative panel."""
    t_before = get_dataset_telemetry(df_before)
    t_after = get_dataset_telemetry(df_after)

    row_delta = t_after["rows"] - t_before["rows"]
    col_delta = t_after["cols"] - t_before["cols"]
    miss_delta = t_after["missing_count"] - t_before["missing_count"]
    dup_delta = t_after["duplicate_count"] - t_before["duplicate_count"]

    def fmt_delta(val, invert_color=False):
        if val == 0:
            return '<span style="color:var(--text-muted); font-size:11.5px;">(0)</span>'
        elif val > 0:
            color = "var(--danger)" if invert_color else "var(--success)"
            return f'<span style="color:{color}; font-weight:700; font-size:11.5px;">(+{val:,})</span>'
        else:
            color = "var(--success)" if invert_color else "var(--danger)"
            return f'<span style="color:{color}; font-weight:700; font-size:11.5px;">({val:,})</span>'

    note_html = f'<div style="color:var(--primary); font-size:13px; margin-top:8px; font-weight:500;">Note: {note}</div>' if note else ""

    comp_html = f"""
    <div class="before-after-grid">
        <div class="ba-card">
            <div class="ba-metric-name">Observations</div>
            <div class="ba-value-row">
                <span class="ba-val-before">{t_before['rows']:,}</span>
                <span class="ba-arrow">→</span>
                <span class="ba-val-after">{t_after['rows']:,} {fmt_delta(row_delta)}</span>
            </div>
        </div>
        <div class="ba-card">
            <div class="ba-metric-name">Features</div>
            <div class="ba-value-row">
                <span class="ba-val-before">{t_before['cols']:,}</span>
                <span class="ba-arrow">→</span>
                <span class="ba-val-after">{t_after['cols']:,} {fmt_delta(col_delta)}</span>
            </div>
        </div>
        <div class="ba-card">
            <div class="ba-metric-name">Missing Cells</div>
            <div class="ba-value-row">
                <span class="ba-val-before">{t_before['missing_count']:,}</span>
                <span class="ba-arrow">→</span>
                <span class="ba-val-after">{t_after['missing_count']:,} {fmt_delta(miss_delta, invert_color=True)}</span>
            </div>
        </div>
        <div class="ba-card">
            <div class="ba-metric-name">Duplicates</div>
            <div class="ba-value-row">
                <span class="ba-val-before">{t_before['duplicate_count']:,}</span>
                <span class="ba-arrow">→</span>
                <span class="ba-val-after">{t_after['duplicate_count']:,} {fmt_delta(dup_delta, invert_color=True)}</span>
            </div>
        </div>
        <div class="ba-card">
            <div class="ba-metric-name">Memory Footprint</div>
            <div class="ba-value-row">
                <span class="ba-val-before">{t_before['memory_mb']:.2f} MB</span>
                <span class="ba-arrow">→</span>
                <span class="ba-val-after">{t_after['memory_mb']:.2f} MB</span>
            </div>
        </div>
    </div>
    {note_html}
    """
    render_html(comp_html)


def render_step_navigation(prev_page: str = None, next_page: str = None, prev_label: str = "Previous Step", next_label: str = "Next Step"):
    """Render standard Previous and Next action buttons at the bottom of each page."""
    render_html('<div class="step-navigation-bar">')
    col1, col2, col3 = st.columns([1.5, 3, 1.5])

    with col1:
        if prev_page:
            clean_prev = prev_label.replace("←", "").strip()
            if st.button(f"← {clean_prev}", use_container_width=True, key="btn_nav_prev", type="secondary"):
                st.switch_page(prev_page)

    with col3:
        if next_page:
            clean_next = next_label.replace("→", "").strip()
            if st.button(f"{clean_next} →", use_container_width=True, key="btn_nav_next"):
                st.switch_page(next_page)
    render_html('</div>')


def render_sidebar_status():
    """Render unified sidebar branding, dataset telemetry badge, and workflow stepper."""
    with st.sidebar:
        # Branding
        render_html("""
        <div class="sidebar-brand">
            <div class="sidebar-logo-text"><span class="brand-dot"></span>Nexus AI</div>
            <div class="sidebar-tagline">Machine Learning Studio</div>
        </div>
        """)

        # Active Dataset Badge
        df = st.session_state.get("data")
        file_name = st.session_state.get("file_name", "No dataset uploaded")
        if df is not None:
            t = get_dataset_telemetry(df)
            render_html(f"""
            <div class="dataset-badge-card">
                <div class="dataset-badge-title">Active Dataset</div>
                <div class="dataset-badge-name" title="{file_name}">{file_name}</div>
                <div class="dataset-badge-stats">
                    <span>{t['rows']:,} rows</span>
                    <span>{t['cols']} cols</span>
                    <span>{t['missing_pct']:.1f}% null</span>
                </div>
            </div>
            """)
        else:
            render_html("""
            <div class="dataset-badge-card" style="border-color: var(--border-subtle);">
                <div class="dataset-badge-title" style="color:var(--text-muted);">No Active Dataset</div>
                <div class="dataset-badge-name" style="color:var(--text-secondary); font-size:12px;">Awaiting ingestion...</div>
                <div class="dataset-badge-stats"><span>Upload data to begin</span></div>
            </div>
            """)

        # Workflow Progress Indicator
        stages = get_workflow_status()
        done_count = sum(1 for s in stages if s["done"])
        progress_pct = int((done_count / len(stages)) * 100)

        render_html(f"""
        <div class="stepper-header" style="margin-top: 14px;">
            <span>Pipeline Progress</span>
            <span style="color:var(--primary); font-weight:700;">{progress_pct}%</span>
        </div>
        """)
        st.progress(progress_pct / 100.0)

        # Quick Actions
        render_html("<br>")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("Reset", use_container_width=True, help="Reset transformations to original raw dataset", type="secondary"):
                reset_workflow(keep_original_file=True)
                st.rerun()
        with col_r2:
            if st.button("Clear", use_container_width=True, help="Clear all data and start fresh", type="secondary"):
                reset_workflow(keep_original_file=False)
                st.switch_page("pages/02_Upload.py")

        st.caption("Nexus AI v3.0 | Enterprise Architecture")


def dataset_guard() -> bool:
    """
    Guard check for downstream pages. If no dataset is active, renders
    a user-friendly message and jump button.
    Returns True if dataset exists, False if guarded (and stops execution).
    """
    if st.session_state.get("data") is None:
        render_hero(
            badge="Ingestion Required",
            title="Active Dataset Required",
            subtitle="This stage requires an active dataset in memory. Please upload a CSV/Excel file or load a benchmark dataset."
        )
        st.info("No active dataset found in memory.")
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("Jump to Dataset Ingestion", use_container_width=True):
                st.switch_page("pages/02_Upload.py")
        st.stop()
        return False
    return True
