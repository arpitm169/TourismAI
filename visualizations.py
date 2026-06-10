"""
visualizations.py
=================
All Plotly chart builders for the Tourism Dashboard.


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
    "bg"         : "rgba(0,0,0,0)",
    "paper_bg"   : "rgba(0,0,0,0)",
    "font_color" : "#0f172a",
    "grid_color" : "#e2e8f0",
    "title_color": "#475569",
    "title_size" : 14,
    "font_family": "DM Sans, sans-serif",
}

CHART_COLORS = [
    "#6366f1",  # indigo
    "#10b981",  # emerald
    "#f59e0b",  # amber
    "#ef4444",  # rose
    "#3b82f6",  # blue
    "#8b5cf6",  # violet
    "#06b6d4",  # cyan
    "#f97316",  # orange
]

SEQUENTIAL_TEAL = ["#e0f2fe", "#0891b2", "#164e63"]
SEQUENTIAL_INDIGO = ["#ede9fe", "#6366f1", "#312e81"]
DIVERGING = [[0, "#ef4444"], [0.5, "#f8fafc"], [1, "#6366f1"]]

PASTEL_COLORS = CHART_COLORS
PASTEL_HEATMAP = DIVERGING
PASTEL_MAP = SEQUENTIAL_TEAL

PALETTE = {
    "primary"   : "#6366f1",
    "secondary" : "#8b5cf6",
    "accent"    : "#f59e0b",
    "success"   : "#10b981",
    "danger"    : "#ef4444",
    "bg"        : CHART_THEME["bg"],
    "card"      : "#ffffff",
    "text"      : CHART_THEME["font_color"],
}
QUALITATIVE = CHART_COLORS


def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply the global dashboard theme to a Plotly figure.

    Sets consistent background colours, fonts, grid lines, hover label
    styling, and legend formatting across all charts.

    Args:
        fig: The Plotly figure to style.
        title: Optional chart title string. Displayed top-left in slate grey.

    Returns:
        The same figure with the theme applied in-place.
    """
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
        hoverlabel = dict(
            bgcolor     = "#1e293b",
            bordercolor = "#1e293b",
            font        = dict(color="#f8fafc", family=CHART_THEME["font_family"]),
        ),
        legend = dict(
            bgcolor     = "rgba(255,255,255,0.86)",
            bordercolor = "#e2e8f0",
            borderwidth = 1,
            font        = dict(color=CHART_THEME["font_color"]),
        ),
        margin = dict(l=40, r=20, t=50, b=40),
        transition = dict(duration=300),
    )
    fig.update_xaxes(
        gridcolor=CHART_THEME["grid_color"],
        linecolor="#e2e8f0",
        tickfont=dict(color=CHART_THEME["font_color"]),
        title_font=dict(color=CHART_THEME["font_color"]),
        zeroline=False,
    )
    fig.update_yaxes(
        gridcolor=CHART_THEME["grid_color"],
        linecolor="#e2e8f0",
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
    """Plot side-by-side histograms of raw and log-transformed revenue.

    Displaying both scales makes it easy to spot the long-tail nature of
    tourism revenue while preserving visibility of the central distribution.

    Args:
        df: DataFrame containing a ``Revenue`` column with numeric values.

    Returns:
        A Plotly Figure with two histogram subplots (raw and log1p revenue).
    """
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Revenue Distribution", "Log Revenue"])
    fig.add_trace(go.Histogram(x=df["Revenue"], nbinsx=40,
                               marker_color=CHART_COLORS[0], opacity=0.82,
                               name="Revenue"), row=1, col=1)
    fig.add_trace(go.Histogram(x=np.log1p(df["Revenue"]), nbinsx=40,
                               marker_color=CHART_COLORS[1], opacity=0.82,
                               name="Log Revenue"), row=1, col=2)
    fig.update_layout(showlegend=False)
    fig.update_traces(marker_line_color="#ffffff", marker_line_width=1)
    fig = _apply_theme(fig, title="Revenue Distribution")
    return fig


def plot_top_destinations(df: pd.DataFrame, n: int = 15) -> go.Figure:
    """Plot a horizontal bar chart of the top N destinations by annual revenue.

    Bars are colour-coded by tourism category and annotated with revenue
    values for quick comparison.

    Args:
        df: DataFrame with columns ``Location``, ``Country``, ``Revenue``,
            and ``Category``.
        n: Number of top destinations to display. Defaults to 15.

    Returns:
        A Plotly Figure with a horizontal bar chart sorted ascending by revenue.
    """
    top = df.nlargest(n, "Revenue")[["Location", "Country", "Revenue", "Category"]]
    top["Label"] = top["Location"] + " (" + top["Country"] + ")"
    fig = px.bar(
        top.sort_values("Revenue"),
        x="Revenue", y="Label", orientation="h",
        color="Category", title=f"Top {n} Destinations by Revenue",
        color_discrete_sequence=CHART_COLORS,
    )
    fig.update_traces(
        marker=dict(line=dict(width=0), cornerradius=6),
        texttemplate="$%{x:,.0f}",
        textfont_color="white",
        textposition="inside",
    )
    fig = _apply_theme(fig, title=f"Top {n} Destinations by Revenue")
    return fig


def plot_category_breakdown(df: pd.DataFrame) -> go.Figure:
    """Plot average revenue and average visitors broken down by tourism category.

    Displays two grouped bar charts side-by-side, sorted by descending average
    revenue, to highlight which destination types perform best economically.

    Args:
        df: DataFrame with columns ``Category``, ``Revenue``, ``Visitors``,
            ``Rating``, and ``Location``.

    Returns:
        A Plotly Figure with two bar subplots: avg revenue and avg visitors
        per category.
    """
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
               marker_color=CHART_COLORS[0], marker_cornerradius=4,
               name="Revenue"),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=grp["Category"], y=grp["Avg_Visitors"],
               marker_color=CHART_COLORS[1], marker_cornerradius=4,
               name="Visitors"),
        row=1, col=2
    )
    fig.update_layout(showlegend=False)
    fig = _apply_theme(fig, title="Performance by Category")
    return fig


def plot_country_map(df: pd.DataFrame) -> go.Figure:
    """Render a choropleth world map shaded by total tourism revenue per country.

    Hover tooltips display total visitors, average rating, and destination
    count for each country.

    Args:
        df: DataFrame with columns ``Country``, ``Revenue``, ``Visitors``,
            ``Rating``, and ``Location``.

    Returns:
        A Plotly Figure with a choropleth map using the sequential teal
        colour scale.
    """
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
        geo=dict(bgcolor="rgba(0,0,0,0)", showframe=False,
                 showcoastlines=True, coastlinecolor="#e2e8f0"),
        coloraxis_colorbar=dict(title="Revenue ($)"),
    )
    fig = _apply_theme(fig, title="Tourism Revenue by Country")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Plot a Pearson correlation heatmap for key numeric tourism features.

    Only columns present in the DataFrame are included. The colour scale
    diverges at zero so positive and negative correlations are immediately
    distinguishable.

    Args:
        df: DataFrame containing one or more of: ``Visitors``, ``Rating``,
            ``Revenue``, ``Revenue_per_Visitor``, ``Popularity_Score``,
            ``Accommodation_Available``, ``High_Revenue_Potential``.

    Returns:
        A Plotly Figure with an annotated correlation heatmap.
    """
    cols = ["Visitors", "Rating", "Revenue", "Revenue_per_Visitor",
            "Popularity_Score", "Accommodation_Available",
            "High_Revenue_Potential"]
    available = [c for c in cols if c in df.columns]
    corr = df[available].corr().round(2)

    fig = go.Figure(go.Heatmap(
        z            = corr.values,
        x            = corr.columns,
        y            = corr.index,
        colorscale   = DIVERGING,
        text         = corr.values,
        texttemplate = "%{text}",
        showscale    = True,
        zmid         = 0,
        xgap         = 2,
        ygap         = 2,
    ))
    fig = _apply_theme(fig, title="Correlation Matrix")
    return fig


def plot_scatter_revenue_visitors(df: pd.DataFrame) -> go.Figure:
    """Plot a log-log scatter of revenue vs visitor count coloured by category.

    Samples up to 500 rows for performance. If ``statsmodels`` is installed
    an OLS trendline is overlaid automatically.

    Args:
        df: DataFrame with columns ``Visitors``, ``Revenue``, ``Category``,
            ``Rating``, ``Location``, and ``Country``.

    Returns:
        A Plotly Figure with a scatter plot on log-log axes.
    """
    scatter_kwargs = {}
    try:
        import statsmodels.api  # noqa: F401
        scatter_kwargs["trendline"] = "ols"
    except ImportError:
        pass

    fig = px.scatter(
        df.sample(min(len(df), 500), random_state=42),
        x="Visitors", y="Revenue",
        color="Category", size="Rating",
        hover_data=["Location", "Country", "Rating"],
        title="Revenue vs. Visitor Count",
        color_discrete_sequence=CHART_COLORS,
        opacity=0.75,
        log_x=True, log_y=True,
        **scatter_kwargs,
    )
    fig.update_traces(marker=dict(size=7, opacity=0.75,
                                  line=dict(color="#ffffff", width=0.6)))
    fig = _apply_theme(fig, title="Revenue vs. Visitor Count")
    return fig


def plot_rating_distribution(df: pd.DataFrame) -> go.Figure:
    """Plot violin charts of rating distributions split by tourism category.

    Each violin includes an embedded box plot and individual outlier points,
    giving a compact view of both distribution shape and spread.

    Args:
        df: DataFrame with columns ``Rating`` and ``Category``.

    Returns:
        A Plotly Figure with one violin per category.
    """
    fig = px.violin(
        df, y="Rating", x="Category",
        box=True, points="outliers",
        color="Category",
        title="Rating Distribution by Category",
        color_discrete_sequence=CHART_COLORS,
    )
    fig = _apply_theme(fig, title="Rating Distribution by Category")
    return fig


def plot_high_revenue_pie(df: pd.DataFrame) -> go.Figure:
    """Plot a donut chart showing the share of high-revenue-potential destinations.

    The centre annotation displays the absolute count of high-potential
    destinations for quick reference.

    Args:
        df: DataFrame with a binary ``High_Revenue_Potential`` column
            (1 = high, 0 = standard).

    Returns:
        A Plotly Figure with a donut (hole=0.55) pie chart.
    """
    counts = df["High_Revenue_Potential"].value_counts()
    fig = go.Figure(go.Pie(
        labels    = ["Standard Revenue", "High Revenue Potential"],
        values    = counts.values,
        hole      = 0.55,
        marker    = dict(colors=[CHART_COLORS[0], CHART_COLORS[2]],
                         line=dict(color="#ffffff", width=2)),
        textfont  = dict(color="#374151"),
        textinfo  = "percent+label",
        hoverinfo = "label+value+percent",
        pull      = [0.05, 0],
    ))
    fig.update_layout(
        annotations  = [dict(text=f"{counts.get(1, 0)}<br>High-Rev",
                             x=0.5, y=0.5, font_size=14, showarrow=False,
                             font=dict(color="#374151"))],
    )
    fig = _apply_theme(fig, title="High Revenue Potential Distribution")
    return fig


def plot_accommodation_impact(df: pd.DataFrame) -> go.Figure:
    """Plot the impact of accommodation availability on average revenue and rating.

    Displays two side-by-side bar charts comparing destinations with and
    without on-site accommodation across both metrics.

    Args:
        df: DataFrame with columns ``Accommodation_Available`` (1/0),
            ``Revenue``, ``Rating``, and ``Location``.

    Returns:
        A Plotly Figure with two bar subplots (avg revenue and avg rating).
    """
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
            marker_color=[CHART_COLORS[1], CHART_COLORS[3]],
            marker_cornerradius=4,
            showlegend=False,
        ), row=1, col=i)
    fig = _apply_theme(fig, title="Impact of Accommodation Availability")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Model Performance Charts
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(conf_matrix: List[List[int]]) -> go.Figure:
    """Plot an annotated confusion matrix heatmap for the binary classifier.

    The two classes are labelled "Standard" and "High Revenue". Cell values
    are displayed as integers directly on the heatmap tiles.

    Args:
        conf_matrix: A 2×2 nested list of integers in the format
            ``[[TN, FP], [FN, TP]]`` as returned by
            ``sklearn.metrics.confusion_matrix``.

    Returns:
        A Plotly Figure with a diverging-coloured heatmap and axis labels.
    """
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
    """Plot a horizontal bar chart of the top N XGBoost feature importances.

    Bars are coloured on a sequential indigo scale proportional to importance
    value, making the most influential features immediately visible.

    Args:
        fi: A pandas Series mapping feature names to importance scores,
            sorted descending (as returned by the model training pipeline).
        top_n: Number of top features to display. Defaults to 12.

    Returns:
        A Plotly Figure with a horizontal bar chart of feature importances.
    """
    top = fi.head(top_n).sort_values()
    fig = go.Figure(go.Bar(
        x             = top.values,
        y             = top.index,
        orientation   = "h",
        marker         = dict(color=top.values, colorscale=SEQUENTIAL_INDIGO,
                              cornerradius=4),
    ))
    fig = _apply_theme(fig, title=f"Top {top_n} Feature Importances (XGBoost)")
    return fig


def plot_metrics_gauge(metrics: Dict) -> go.Figure:
    """Render four gauge indicators for classifier accuracy, recall, precision, and F1.

    Each gauge is coloured with threshold bands:
    - 0–70%: light grey (poor)
    - 70–90%: light blue (acceptable)
    - 90–100%: light green (target)

    A dashed threshold line is drawn at 95% to mark the project target.

    Args:
        metrics: Dictionary with keys ``"accuracy"``, ``"recall"``,
            ``"precision"``, and ``"f1"``; values are numeric percentages
            (e.g. ``97.4``).

    Returns:
        A Plotly Figure with four indicator gauges in a single row.
    """
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
    colors = CHART_COLORS[:4]
    for i, (label, value) in enumerate(indicators, 1):
        fig.add_trace(go.Indicator(
            mode  = "gauge+number",
            value = value,
            title = {"text": label, "font": {"size": 13}},
            gauge = {
                "axis"      : {"range": [0, 100], "tickcolor": "#e2e8f0"},
                "bar"       : {"color": colors[i-1]},
                "bgcolor"   : "#ffffff",
                "bordercolor": "#e2e8f0",
                "borderwidth": 1,
                "steps"     : [
                    {"range": [0, 70], "color": "#f1f5f9"},
                    {"range": [70, 90], "color": "#dbeafe"},
                    {"range": [90, 100], "color": "#bbf7d0"},
                ],
                "threshold" : {
                    "line" : {"color": CHART_COLORS[2], "width": 2},
                    "thickness": 0.75,
                    "value": 95,
                },
            },
            number={"suffix": "%"},
        ), row=1, col=i)
    fig = _apply_theme(fig, title="Classification Model Performance")
    return fig


def plot_cv_scores(cv_scores: Dict[str, List[float]]) -> go.Figure:
    """Plot per-fold cross-validation scores with mean ± std reference lines.

    Renders one bar chart subplot per available metric (accuracy, recall, F1).
    A dashed horizontal line annotated with the mean and standard deviation is
    overlaid on each subplot.

    Args:
        cv_scores: Dictionary mapping metric names (``"accuracy"``,
            ``"recall"``, ``"f1"``) to lists of per-fold percentage scores.
            Missing keys are skipped gracefully.

    Returns:
        A Plotly Figure with bar subplots per metric. Returns an annotated
        empty figure if no recognised metrics are present in ``cv_scores``.
    """
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
    colors = CHART_COLORS

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
            line=dict(color=CHART_COLORS[2], width=2, dash="dash"),
            annotation_text=f"μ={mean_val:.1f}% ±{std_val:.1f}",
            annotation_position="top right",
            annotation_font=dict(color=CHART_THEME["title_color"], size=10),
        )

    fig.update_yaxes(range=[80, 102])
    fig = _apply_theme(fig, title="Cross-Validation Scores (5-Fold Stratified)")
    return fig


def plot_model_comparison(comparison: Dict[str, Dict], ensemble_metrics: Dict) -> go.Figure:
    """Plot a grouped bar chart comparing individual base models against the stacking ensemble.

    Metrics shown: accuracy, recall, precision, F1, and ROC-AUC. The ensemble
    is appended as a separate model group so relative uplift is immediately
    visible.

    Args:
        comparison: Dictionary mapping model name strings to metric dicts,
            each containing keys ``"accuracy"``, ``"recall"``, ``"precision"``,
            ``"f1"``, and ``"roc_auc"`` as numeric percentages.
        ensemble_metrics: Metric dict for the stacking ensemble, same
            structure as the per-model dicts in ``comparison``.

    Returns:
        A Plotly Figure with a grouped bar chart. Returns an annotated empty
        figure if ``comparison`` is empty.
    """
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
    colors = CHART_COLORS

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
    """Plot a scatter of actual vs predicted regression values with a perfect-fit line.

    Points are semi-transparent to show density. The dashed diagonal reference
    line (y = x) makes over- and under-prediction easy to spot visually.

    Args:
        actual: 1-D array of ground-truth target values.
        predicted: 1-D array of model-predicted values, same length as
            ``actual``.
        title: Chart title string. Defaults to ``"Actual vs Predicted"``.

    Returns:
        A Plotly Figure with a scatter trace and a perfect-fit reference line.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual, y=predicted,
        mode="markers",
        marker=dict(color=CHART_COLORS[0], opacity=0.6, size=6),
        name="Predictions",
    ))
    mn, mx = float(np.min(actual)), float(np.max(actual))
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines",
        line=dict(color=CHART_COLORS[1], dash="dash", width=2),
        name="Perfect Fit",
    ))
    fig.update_layout(
        xaxis_title  = "Actual",
        yaxis_title  = "Predicted",
    )
    fig = _apply_theme(fig, title=title)
    return fig


def plot_revenue_per_visitor_heatmap(df: pd.DataFrame) -> go.Figure:
    """Plot a heatmap of mean revenue-per-visitor across tourism categories and countries.

    The pivot table groups by ``Category`` (rows) and ``Country`` (columns).
    Cells with no data are shown as blank. The diverging colour scale
    highlights both high- and low-performing combinations.

    Args:
        df: DataFrame with columns ``Category``, ``Country``, and
            ``Revenue_per_Visitor`` (numeric, USD).

    Returns:
        A Plotly Figure with a heatmap where colour encodes average revenue
        per visitor.
    """
    pivot = df.groupby(["Category", "Country"])["Revenue_per_Visitor"].mean().unstack()
    fig = go.Figure(go.Heatmap(
        z          = pivot.values,
        x          = pivot.columns,
        y          = pivot.index,
        colorscale = DIVERGING,
        showscale  = True,
        xgap       = 2,
        ygap       = 2,
    ))
    fig.update_layout(
        xaxis_title = "Country",
        yaxis_title = "Category",
    )
    fig = _apply_theme(fig, title="Revenue per Visitor (Category × Country)")
    return fig
