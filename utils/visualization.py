"""
Nexus AI — Enterprise Plotly Visualization Engine
-------------------------------------------------
Publication-quality, interactive 2D & 3D visualization studio and ML diagnostic charts.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Curated vibrant categorical color sequence for enterprise dark charts
PRO_PALETTE = ["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899", "#06B6D4", "#6366F1"]


def _apply_pro_theme(fig: go.Figure, title: str = None) -> go.Figure:
    """Apply uniform enterprise styling to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(family="Inter, -apple-system, sans-serif", size=14, color="#FFFFFF"),
            x=0.01,
            y=0.98,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17, 24, 39, 0.45)",
        font=dict(family="Inter, -apple-system, sans-serif", size=12, color="#94A3B8"),
        margin=dict(l=36, r=30, t=46, b=36),
        legend=dict(
            bgcolor="rgba(17, 24, 39, 0.8)",
            bordercolor="rgba(255, 255, 255, 0.08)",
            borderwidth=1,
            font=dict(size=11, color="#CBD5E1"),
        ),
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#FFFFFF",
            bordercolor="rgba(59, 130, 246, 0.4)",
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.1)",
        zerolinecolor="rgba(148, 163, 184, 0.2)",
        tickfont=dict(size=11, color="#94A3B8"),
        title_font=dict(size=12, color="#CBD5E1"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.1)",
        zerolinecolor="rgba(148, 163, 184, 0.2)",
        tickfont=dict(size=11, color="#94A3B8"),
        title_font=dict(size=12, color="#CBD5E1"),
    )
    return fig


# Backward compatibility alias
_apply_cyber_theme = _apply_pro_theme


# ============================================================
# 1. 2D EXPLORATORY VISUALIZATIONS
# ============================================================

def plot_histogram(df: pd.DataFrame, x_col: str, color_col: str = None, n_bins: int = 30, kde: bool = False) -> go.Figure:
    """Generate distribution histogram with optional color segmentation and marginal box plot."""
    c_val = color_col if color_col and color_col != "None" else None
    marginal = "box" if kde else None

    fig = px.histogram(
        df,
        x=x_col,
        color=c_val,
        nbins=n_bins,
        marginal=marginal,
        color_discrete_sequence=PRO_PALETTE,
        opacity=0.85,
    )
    fig.update_layout(barmode="overlay" if c_val else "relative")
    return _apply_pro_theme(fig, f"Distribution of {x_col}")


def plot_boxplot(df: pd.DataFrame, y_col: str, x_col: str = None, color_col: str = None) -> go.Figure:
    """Generate box plot for outlier detection and quantile distribution."""
    x_val = x_col if x_col and x_col != "None" else None
    c_val = color_col if color_col and color_col != "None" else None

    fig = px.box(
        df,
        x=x_val,
        y=y_col,
        color=c_val,
        points="outliers",
        color_discrete_sequence=PRO_PALETTE,
    )
    return _apply_pro_theme(fig, f"Box Plot: {y_col}" + (f" grouped by {x_val}" if x_val else ""))


def plot_scatter_2d(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None, trendline: bool = False) -> go.Figure:
    """Generate 2D scatter plot with optional regression trendline."""
    c_val = color_col if color_col and color_col != "None" else None
    t_val = "ols" if trendline else None

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=c_val,
        trendline=t_val,
        color_discrete_sequence=PRO_PALETTE,
        opacity=0.8,
    )
    return _apply_pro_theme(fig, f"Scatter: {x_col} vs {y_col}")


def plot_correlation_heatmap(df: pd.DataFrame, columns: list = None) -> go.Figure:
    """Generate Pearson correlation heatmap."""
    cols = columns if columns else df.select_dtypes(include=[np.number]).columns.tolist()
    if len(cols) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient numerical features for correlation matrix", showarrow=False, font=dict(size=14, color="#64748B"))
        return _apply_pro_theme(fig, "Correlation Matrix")

    corr = df[cols].corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(coloraxis_colorbar=dict(title=dict(text="Corr", font=dict(color="#CBD5E1"))))
    return _apply_pro_theme(fig, "Pearson Correlation Matrix")


def plot_missing_heatmap(df: pd.DataFrame) -> go.Figure:
    """Generate missingness binary matrix heatmap."""
    null_matrix = df.isnull().astype(int)
    fig = px.imshow(
        null_matrix,
        color_continuous_scale=[(0, "#111827"), (1, "#EF4444")],
        aspect="auto",
        labels=dict(x="Feature", y="Observation Index", color="Missing"),
    )
    fig.update_layout(coloraxis_showscale=False)
    return _apply_pro_theme(fig, "Missing Values Heatmap (Red = Missing)")


# Alias for backward compatibility
plot_missing_matrix = plot_missing_heatmap


def plot_categorical_distribution(df: pd.DataFrame, col: str, chart_type: str = "bar") -> go.Figure:
    """Plot distribution of categorical feature (Bar or Donut)."""
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "Count"]

    if chart_type == "donut":
        fig = px.pie(
            counts,
            names=col,
            values="Count",
            hole=0.55,
            color_discrete_sequence=PRO_PALETTE,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
    else:
        fig = px.bar(
            counts,
            x=col,
            y="Count",
            text="Count",
            color=col,
            color_discrete_sequence=PRO_PALETTE,
        )
        fig.update_traces(textposition="outside")
    return _apply_pro_theme(fig, f"Distribution of {col}")


# ============================================================
# 2. 3D INTERACTIVE VISUALIZATIONS
# ============================================================

def plot_scatter_3d(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, color_col: str = None, size_col: str = None) -> go.Figure:
    """Generate 3D scatter plot."""
    c_val = color_col if color_col and color_col != "None" else None
    s_val = size_col if size_col and size_col != "None" else None

    fig = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=c_val,
        size=s_val,
        color_discrete_sequence=PRO_PALETTE,
        opacity=0.85,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, b=0, t=36),
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(17,24,39,0.6)", gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(backgroundcolor="rgba(17,24,39,0.6)", gridcolor="rgba(255,255,255,0.08)"),
            zaxis=dict(backgroundcolor="rgba(17,24,39,0.6)", gridcolor="rgba(255,255,255,0.08)"),
        ),
        title=dict(text=f"<b>3D Feature Space: {x_col} × {y_col} × {z_col}</b>", font=dict(family="Inter", size=14, color="#FFFFFF")),
    )
    return fig


def plot_3d_pca_space(pca_df: pd.DataFrame, color_col: str = None, explained_variance: list = None) -> go.Figure:
    """Render 3D PCA projection space."""
    c_val = color_col if color_col and color_col in pca_df.columns else None

    var_str = ""
    if explained_variance is not None and len(explained_variance) >= 3:
        var_str = f" (PC1: {explained_variance[0]*100:.1f}%, PC2: {explained_variance[1]*100:.1f}%, PC3: {explained_variance[2]*100:.1f}%)"

    fig = px.scatter_3d(
        pca_df,
        x="PC_1",
        y="PC_2",
        z="PC_3",
        color=c_val,
        color_discrete_sequence=PRO_PALETTE,
        opacity=0.85,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, b=0, t=36),
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(17,24,39,0.6)", gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(backgroundcolor="rgba(17,24,39,0.6)", gridcolor="rgba(255,255,255,0.08)"),
            zaxis=dict(backgroundcolor="rgba(17,24,39,0.6)", gridcolor="rgba(255,255,255,0.08)"),
        ),
        title=dict(text=f"<b>3D Principal Component Projection{var_str}</b>", font=dict(family="Inter", size=14, color="#FFFFFF")),
    )
    return fig


# ============================================================
# 3. MACHINE LEARNING DIAGNOSTIC VISUALIZATIONS
# ============================================================

def plot_confusion_matrix(cm: np.ndarray, labels: list) -> go.Figure:
    """Render annotated confusion matrix heatmap."""
    labels_str = [str(l) for l in labels]
    fig = px.imshow(
        cm,
        x=labels_str,
        y=labels_str,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        text_auto=True,
    )
    fig.update_layout(
        xaxis_title=dict(text="Predicted Label", font=dict(color="#CBD5E1")),
        yaxis_title=dict(text="Actual True Label", font=dict(color="#CBD5E1")),
    )
    return _apply_pro_theme(fig, "Confusion Matrix Heatmap")


def plot_roc_curve_chart(y_true, y_prob, class_names=None) -> go.Figure:
    """Generate multi-class or binary ROC curve."""
    fig = go.Figure()

    # Binary Classification
    if y_prob.ndim == 1 or y_prob.shape[1] == 2:
        prob_scores = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
        
        # Ensure binary numeric target
        if pd.api.types.is_numeric_dtype(y_true):
            y_bin = y_true
        else:
            classes = np.unique(y_true)
            y_bin = (np.asarray(y_true) == classes[1]).astype(int)

        fpr, tpr, _ = roc_curve(y_bin, prob_scores)
        roc_auc = auc(fpr, tpr)

        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"ROC (AUC = {roc_auc:.3f})",
            line=dict(color="#3B82F6", width=2.5),
        ))
    else:
        # Multi-class One-vs-Rest ROC
        classes = class_names if class_names else [f"Class {i}" for i in range(y_prob.shape[1])]
        y_true_arr = np.asarray(y_true)

        for i, c_name in enumerate(classes):
            if i >= y_prob.shape[1]:
                break
            y_bin = (y_true_arr == i).astype(int) if pd.api.types.is_numeric_dtype(y_true) else (y_true_arr == str(c_name)).astype(int)
            if np.sum(y_bin) == 0:
                continue

            fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            c_color = PRO_PALETTE[i % len(PRO_PALETTE)]

            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=f"{c_name} (AUC = {roc_auc:.3f})",
                line=dict(color=c_color, width=2),
            ))

    # Baseline Chance Line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random Chance",
        line=dict(color="#64748B", width=1.5, dash="dash"),
    ))

    fig.update_layout(
        xaxis_title=dict(text="False Positive Rate", font=dict(color="#CBD5E1")),
        yaxis_title=dict(text="True Positive Rate", font=dict(color="#CBD5E1")),
        xaxis=dict(range=[-0.01, 1.01]),
        yaxis=dict(range=[-0.01, 1.05]),
    )
    return _apply_pro_theme(fig, "Receiver Operating Characteristic (ROC) Curve")


def plot_precision_recall_chart(y_true, y_prob, class_names=None) -> go.Figure:
    """Generate Precision-Recall Curve."""
    fig = go.Figure()

    if y_prob.ndim == 1 or y_prob.shape[1] == 2:
        prob_scores = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
        if pd.api.types.is_numeric_dtype(y_true):
            y_bin = y_true
        else:
            classes = np.unique(y_true)
            y_bin = (np.asarray(y_true) == classes[1]).astype(int)

        precision, recall, _ = precision_recall_curve(y_bin, prob_scores)
        pr_auc = auc(recall, precision)

        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode="lines",
            name=f"PR (AUC = {pr_auc:.3f})",
            line=dict(color="#10B981", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.12)",
        ))
    else:
        classes = class_names if class_names else [f"Class {i}" for i in range(y_prob.shape[1])]
        y_true_arr = np.asarray(y_true)

        for i, c_name in enumerate(classes):
            if i >= y_prob.shape[1]:
                break
            y_bin = (y_true_arr == i).astype(int) if pd.api.types.is_numeric_dtype(y_true) else (y_true_arr == str(c_name)).astype(int)
            if np.sum(y_bin) == 0:
                continue

            precision, recall, _ = precision_recall_curve(y_bin, y_prob[:, i])
            pr_auc = auc(recall, precision)
            c_color = PRO_PALETTE[i % len(PRO_PALETTE)]

            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode="lines",
                name=f"{c_name} (AUC = {pr_auc:.3f})",
                line=dict(color=c_color, width=2),
            ))

    fig.update_layout(
        xaxis_title=dict(text="Recall", font=dict(color="#CBD5E1")),
        yaxis_title=dict(text="Precision", font=dict(color="#CBD5E1")),
        xaxis=dict(range=[-0.01, 1.01]),
        yaxis=dict(range=[-0.01, 1.05]),
    )
    return _apply_pro_theme(fig, "Precision-Recall Curve")


def plot_feature_importance_chart(importance_df: pd.DataFrame) -> go.Figure:
    """Render horizontal ranked feature importance bar chart."""
    df_sorted = importance_df.sort_values("Importance", ascending=True).tail(15)

    fig = px.bar(
        df_sorted,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        coloraxis_showscale=False,
        yaxis_title=dict(text="Feature", font=dict(color="#CBD5E1")),
        xaxis_title=dict(text="Relative Importance", font=dict(color="#CBD5E1")),
    )
    return _apply_pro_theme(fig, "Top Feature Importance & Attribution")


def plot_actual_vs_predicted(y_true, y_pred) -> go.Figure:
    """Generate Regression Actual vs. Predicted scatter plot with perfect diagonal."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode="markers",
        name="Observations",
        marker=dict(color="#38BDF8", size=7, opacity=0.8),
    ))

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))

    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Perfect Identity (y = x)",
        line=dict(color="#F87171", width=1.5, dash="dash"),
    ))

    fig.update_layout(
        xaxis_title=dict(text="Actual Ground Truth", font=dict(color="#CBD5E1")),
        yaxis_title=dict(text="Model Predicted Value", font=dict(color="#CBD5E1")),
    )
    return _apply_pro_theme(fig, "Actual vs. Predicted Values")


def plot_residuals_chart(y_true, y_pred) -> go.Figure:
    """Generate Residuals Error distribution plot for regression."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode="markers",
        name="Residuals",
        marker=dict(color="#F59E0B", size=7, opacity=0.8),
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="#10B981", line_width=1.5)

    fig.update_layout(
        xaxis_title=dict(text="Predicted Value", font=dict(color="#CBD5E1")),
        yaxis_title=dict(text="Residual Error (Actual - Predicted)", font=dict(color="#CBD5E1")),
    )
    return _apply_pro_theme(fig, "Residual Error Distribution")


def plot_prediction_probabilities(classes: list, probabilities: list) -> go.Figure:
    """Render horizontal bar chart of class probabilities."""
    df_p = pd.DataFrame({"Class": [str(c) for c in classes], "Probability": probabilities})
    df_p = df_p.sort_values("Probability", ascending=True)

    fig = px.bar(
        df_p,
        x="Probability",
        y="Class",
        orientation="h",
        text=[f"{p:.1%}" for p in df_p["Probability"]],
        color="Probability",
        color_continuous_scale="Blues",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis=dict(range=[0, 1.15]),
        xaxis_title=dict(text="Confidence Probability", font=dict(color="#CBD5E1")),
        yaxis_title=dict(text="Class", font=dict(color="#CBD5E1")),
    )
    return _apply_pro_theme(fig, "Class Prediction Confidence")
