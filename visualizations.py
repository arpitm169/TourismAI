"""
visualizations.py
=================
All Plotly chart builders for the Tourism Dashboard.

Author : Tourism-AI Team
Version: 1.0.0
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Colour Palette ──────────────────────────────────────────────────────────
CHART_THEME = {
    "bg"         : "#faf8ff",
    "paper_bg"   : "#faf8ff",
    "font_color" : "#374151",
    "grid_color" : "#ede9fe",
    "title_color": "#5b4a8a",
    "title_size" : 15,
    "font_family": "Inter, sans-serif",
}

PASTEL_COLORS = [
    "#a78bfa",  # lavender
    "#86efac",  # mint green
    "#93c5fd",  # sky blue
    "#fca5a5",  # soft coral
    "#fcd34d",  # soft yellow
    "#f9a8d4",  # pink
    "#6ee7b7",  # teal
    "#c4b5fd",  # light purple
]

PASTEL_HEATMAP = [[0, "#faf8ff"], [0.5, "#c4b5fd"], [1, "#5b4a8a"]]
PASTEL_MAP = [[0, "#ede9fe"], [0.5, "#a78bfa"], [1, "#5b4a8a"]]

PALETTE = {
    "primary"   : PASTEL_COLORS[0],
    "secondary" : PASTEL_COLORS[7],
    "accent"    : PASTEL_COLORS[4],
    "success"   : PASTEL_COLORS[1],
    "danger"    : PASTEL_COLORS[3],
    "bg"        : CHART_THEME["bg"],
    "card"      : "#ffffff",
    "text"      : CHART_THEME["font_color"],
}
QUALITATIVE = PASTEL_COLORS


def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        plot_bgcolor   = CHART_THEME["bg"],
        paper_bgcolor  = CHART_THEME["paper_bg"],
        font           = dict(
            color  = CHART_THEME["font_color"],
            family = CHART_THEME["font_family"],
            size   = 12,
        ),
        title          = dict(
            text     = title,
            font     = dict(
                color  = CHART_THEME["title_color"],
                size   = CHART_THEME["title_size"],
                family = CHART_THEME["font_family"],
            ),
            x       = 0.02,
            xanchor = "left",
        ),
        xaxis = dict(
            gridcolor    = CHART_THEME["grid_color"],
            linecolor    = "#e5d8fb",
            tickfont     = dict(color=CHART_THEME["font_color"]),
            title_font   = dict(color=CHART_THEME["font_color"]),
            showgrid     = True,
            zeroline     = False,
        ),
        yaxis = dict(
            gridcolor    = CHART_THEME["grid_color"],
            linecolor    = "#e5d8fb",
            tickfont     = dict(color=CHART_THEME["font_color"]),
            title_font   = dict(color=CHART_THEME["font_color"]),
            showgrid     = True,
            zeroline     = False,
        ),
        legend = dict(
            bgcolor     = "#ffffff",
            bordercolor = "#e5d8fb",
            borderwidth = 1,
            font        = dict(color=CHART_THEME["font_color"]),
        ),
        margin = dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(
        gridcolor=CHART_THEME["grid_color"],
        linecolor="#e5d8fb",
        tickfont=dict(color=CHART_THEME["font_color"]),
        title_font=dict(color=CHART_THEME["font_color"]),
        zeroline=False,
    )
    fig.update_yaxes(
        gridcolor=CHART_THEME["grid_color"],
        linecolor="#e5d8fb",
        tickfont=dict(color=CHART_THEME["font_color"]),
        title_font=dict(color=CHART_THEME["font_color"]),
        zeroline=False,
    )
    fig.update_annotations(
        font=dict(color=CHART_THEME["font_color"], family=CHART_THEME["font_family"])
    )
    fig.update_coloraxes(
        colorbar=dict(
            tickfont=dict(color=CHART_THEME["font_color"]),
            title_font=dict(color=CHART_THEME["font_color"]),
        )
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# EDA Charts
# ═══════════════════════════════════════════════════════════════════════════════

def plot_revenue_distribution(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Revenue Distribution", "Log Revenue"])
    fig.add_trace(go.Histogram(x=df["Revenue"], nbinsx=40,
                               marker_color="#a78bfa", opacity=0.8,
                               name="Revenue"), row=1, col=1)
    fig.add_trace(go.Histogram(x=np.log1p(df["Revenue"]), nbinsx=40,
                               marker_color="#c4b5fd", opacity=0.8,
                               name="Log Revenue"), row=1, col=2)
    fig.update_layout(showlegend=False)
    fig = _apply_theme(fig, title="Revenue Distribution")
    return fig


def plot_top_destinations(df: pd.DataFrame, n: int = 15) -> go.Figure:
    top = df.nlargest(n, "Revenue")[["Location", "Country", "Revenue", "Category"]]
    top["Label"] = top["Location"] + " (" + top["Country"] + ")"
    fig = px.bar(
        top.sort_values("Revenue"),
        x="Revenue", y="Label", orientation="h",
        color="Category", title=f"Top {n} Destinations by Revenue",
        color_discrete_sequence=QUALITATIVE,
    )
    fig.update_traces(texttemplate="$%{x:,.0f}", textposition="outside")
    fig = _apply_theme(fig, title=f"Top {n} Destinations by Revenue")
    return fig


def plot_category_breakdown(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby("Category").agg(
        Avg_Revenue   = ("Revenue",   "mean"),
        Avg_Visitors  = ("Visitors",  "mean"),
        Avg_Rating    = ("Rating",    "mean"),
        Count         = ("Location",  "count"),
    ).reset_index().sort_values("Avg_Revenue", ascending=False)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Avg Revenue by Category",
                                        "Avg Visitors by Category"])
    fig.add_trace(
        go.Bar(x=grp["Category"], y=grp["Avg_Revenue"],
               marker_color="#a78bfa", name="Revenue"),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=grp["Category"], y=grp["Avg_Visitors"],
               marker_color="#86efac", name="Visitors"),
        row=1, col=2
    )
    fig.update_layout(showlegend=False)
    fig = _apply_theme(fig, title="Performance by Category")
    return fig


def plot_country_map(df: pd.DataFrame) -> go.Figure:
    country_agg = df.groupby("Country").agg(
        Total_Revenue  = ("Revenue",  "sum"),
        Total_Visitors = ("Visitors", "sum"),
        Avg_Rating     = ("Rating",   "mean"),
        Count          = ("Location", "count"),
    ).reset_index()

    fig = px.choropleth(
        country_agg,
        locations       = "Country",
        locationmode    = "country names",
        color           = "Total_Revenue",
        hover_data      = ["Total_Visitors", "Avg_Rating", "Count"],
        title           = "Tourism Revenue by Country",
        color_continuous_scale = PASTEL_MAP,
    )
    fig.update_layout(
        geo=dict(bgcolor="#faf8ff", showframe=False,
                 showcoastlines=True, coastlinecolor="#e5d8fb"),
        coloraxis_colorbar=dict(title="Revenue ($)"),
    )
    fig = _apply_theme(fig, title="Tourism Revenue by Country")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    cols = ["Visitors", "Rating", "Revenue", "Revenue_per_Visitor",
            "Popularity_Score", "Accommodation_Available",
            "High_Revenue_Potential"]
    available = [c for c in cols if c in df.columns]
    corr = df[available].corr().round(2)

    fig = go.Figure(go.Heatmap(
        z            = corr.values,
        x            = corr.columns,
        y            = corr.index,
        colorscale   = PASTEL_HEATMAP,
        text         = corr.values,
        texttemplate = "%{text}",
        showscale    = True,
        zmid         = 0,
    ))
    fig = _apply_theme(fig, title="Correlation Matrix")
    return fig


def plot_scatter_revenue_visitors(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df.sample(min(len(df), 500), random_state=42),
        x="Visitors", y="Revenue",
        color="Category", size="Rating",
        hover_data=["Location", "Country", "Rating"],
        title="Revenue vs. Visitor Count",
        color_discrete_sequence=QUALITATIVE,
        opacity=0.75,
        log_x=True, log_y=True,
    )
    fig.update_traces(marker=dict(line=dict(color="#ffffff", width=0.6)))
    fig = _apply_theme(fig, title="Revenue vs. Visitor Count")
    return fig


def plot_rating_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.violin(
        df, y="Rating", x="Category",
        box=True, points="outliers",
        color="Category",
        title="Rating Distribution by Category",
        color_discrete_sequence=QUALITATIVE,
    )
    fig = _apply_theme(fig, title="Rating Distribution by Category")
    return fig


def plot_high_revenue_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["High_Revenue_Potential"].value_counts()
    fig = go.Figure(go.Pie(
        labels    = ["Standard Revenue", "High Revenue Potential"],
        values    = counts.values,
        hole      = 0.55,
        marker    = dict(colors=["#a78bfa", "#86efac", "#93c5fd", "#fca5a5", "#fcd34d"]),
        textfont  = dict(color="#374151"),
        textinfo  = "percent+label",
        hoverinfo = "label+value+percent",
    ))
    fig.update_layout(
        annotations  = [dict(text=f"{counts.get(1, 0)}<br>High-Rev",
                             x=0.5, y=0.5, font_size=14, showarrow=False,
                             font=dict(color="#374151"))],
    )
    fig = _apply_theme(fig, title="High Revenue Potential Distribution")
    return fig


def plot_accommodation_impact(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby("Accommodation_Available").agg(
        Avg_Revenue = ("Revenue",  "mean"),
        Avg_Rating  = ("Rating",   "mean"),
        Count       = ("Location", "count"),
    ).reset_index()
    grp["Label"] = grp["Accommodation_Available"].map({1: "With Accommodation", 0: "Without"})

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Avg Revenue", "Avg Rating"])
    for i, col in enumerate(["Avg_Revenue", "Avg_Rating"], 1):
        fig.add_trace(go.Bar(
            x=grp["Label"], y=grp[col],
            marker_color=["#86efac", "#fca5a5"],
            showlegend=False,
        ), row=1, col=i)
    fig = _apply_theme(fig, title="Impact of Accommodation Availability")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Model Performance Charts
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(conf_matrix: List[List[int]]) -> go.Figure:
    labels = ["Standard", "High Revenue"]
    fig = go.Figure(go.Heatmap(
        z            = conf_matrix,
        x            = [f"Pred: {l}" for l in labels],
        y            = [f"True: {l}" for l in labels],
        colorscale   = PASTEL_HEATMAP,
        text         = conf_matrix,
        texttemplate = "%{text}",
        showscale    = False,
    ))
    fig = _apply_theme(fig, title="Confusion Matrix")
    return fig


def plot_feature_importances(fi: pd.Series, top_n: int = 12) -> go.Figure:
    top = fi.head(top_n).sort_values()
    fig = go.Figure(go.Bar(
        x             = top.values,
        y             = top.index,
        orientation   = "h",
        marker_color  = "#a78bfa",
    ))
    fig = _apply_theme(fig, title=f"Top {top_n} Feature Importances (XGBoost)")
    return fig


def plot_metrics_gauge(metrics: Dict) -> go.Figure:
    """Gauge charts for accuracy, recall, precision, F1."""
    indicators = [
        ("Accuracy",  metrics.get("accuracy",  0)),
        ("Recall",    metrics.get("recall",    0)),
        ("Precision", metrics.get("precision", 0)),
        ("F1 Score",  metrics.get("f1",        0)),
    ]
    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "indicator"}] * 4],
    )
    colors = ["#86efac", "#a78bfa", "#93c5fd", "#fcd34d"]
    for i, (label, value) in enumerate(indicators, 1):
        fig.add_trace(go.Indicator(
            mode  = "gauge+number",
            value = value,
            title = {"text": label, "font": {"size": 13}},
            gauge = {
                "axis"      : {"range": [0, 100], "tickcolor": "#e5d8fb"},
                "bar"       : {"color": colors[i-1]},
                "bgcolor"   : "#ffffff",
                "bordercolor": "#e5d8fb",
                "borderwidth": 1,
                "threshold" : {
                    "line" : {"color": "#fcd34d", "width": 2},
                    "thickness": 0.75,
                    "value": 95,
                },
            },
            number={"suffix": "%"},
        ), row=1, col=i)
    fig = _apply_theme(fig, title="Classification Model Performance")
    return fig


def plot_cv_scores(cv_scores: Dict[str, List[float]]) -> go.Figure:
    """Bar chart of per-fold CV scores with mean ± std reference lines."""
    metrics_to_show = ["accuracy", "recall", "f1"]
    available = [m for m in metrics_to_show if m in cv_scores]

    if not available:
        fig = go.Figure()
        fig.add_annotation(text="No CV scores available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig = _apply_theme(fig, title="Cross-Validation Scores (5-Fold Stratified)")
        return fig

    fig = make_subplots(
        rows=1, cols=len(available),
        subplot_titles=[m.title() for m in available],
    )
    colors = PASTEL_COLORS

    for i, metric in enumerate(available, 1):
        folds = cv_scores[metric]
        mean_val = np.mean(folds)
        std_val = np.std(folds)
        fold_labels = [f"Fold {j+1}" for j in range(len(folds))]

        fig.add_trace(go.Bar(
            x=fold_labels, y=folds,
            marker_color=colors[i - 1],
            name=metric.title(),
            text=[f"{v:.1f}%" for v in folds],
            textposition="outside",
            showlegend=False,
        ), row=1, col=i)

        # Mean line
        fig.add_hline(
            y=mean_val, row=1, col=i,
            line=dict(color="#fcd34d", width=2, dash="dash"),
            annotation_text=f"μ={mean_val:.1f}% ±{std_val:.1f}",
            annotation_position="top right",
            annotation_font=dict(color="#5b4a8a", size=10),
        )

    fig.update_yaxes(range=[80, 102])
    fig = _apply_theme(fig, title="Cross-Validation Scores (5-Fold Stratified)")
    return fig


def plot_model_comparison(comparison: Dict[str, Dict], ensemble_metrics: Dict) -> go.Figure:
    """Grouped bar chart comparing individual models vs ensemble."""
    if not comparison:
        fig = go.Figure()
        fig.add_annotation(text="No comparison data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig = _apply_theme(fig, title="Model Comparison: Individual vs Stacking Ensemble")
        return fig

    # Add ensemble to comparison
    all_models = dict(comparison)
    all_models["Stacking Ensemble"] = {
        "accuracy": ensemble_metrics.get("accuracy", 0),
        "recall": ensemble_metrics.get("recall", 0),
        "precision": ensemble_metrics.get("precision", 0),
        "f1": ensemble_metrics.get("f1", 0),
        "roc_auc": ensemble_metrics.get("roc_auc", 0),
    }

    model_names = list(all_models.keys())
    metrics_list = ["accuracy", "recall", "precision", "f1", "roc_auc"]
    colors = PASTEL_COLORS

    fig = go.Figure()
    for j, metric in enumerate(metrics_list):
        values = [all_models[m].get(metric, 0) for m in model_names]
        fig.add_trace(go.Bar(
            name=metric.replace("_", " ").title(),
            x=model_names, y=values,
            marker_color=colors[j],
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
        ))

    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 105], title="Score (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig = _apply_theme(fig, title="Model Comparison: Individual vs Stacking Ensemble")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Prediction Charts
# ═══════════════════════════════════════════════════════════════════════════════

def plot_prediction_comparison(
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str = "Actual vs Predicted",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual, y=predicted,
        mode="markers",
        marker=dict(color="#a78bfa", opacity=0.6, size=6),
        name="Predictions",
    ))
    mn, mx = float(np.min(actual)), float(np.max(actual))
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines",
        line=dict(color="#86efac", dash="dash", width=2),
        name="Perfect Fit",
    ))
    fig.update_layout(
        xaxis_title  = "Actual",
        yaxis_title  = "Predicted",
    )
    fig = _apply_theme(fig, title=title)
    return fig


def plot_revenue_per_visitor_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = df.groupby(["Category", "Country"])["Revenue_per_Visitor"].mean().unstack()
    fig = go.Figure(go.Heatmap(
        z          = pivot.values,
        x          = pivot.columns,
        y          = pivot.index,
        colorscale = PASTEL_HEATMAP,
        showscale  = True,
    ))
    fig.update_layout(
        xaxis_title = "Country",
        yaxis_title = "Category",
    )
    fig = _apply_theme(fig, title="Revenue per Visitor (Category × Country)")
    return fig
