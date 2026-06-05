"""
app.py
======
Main Streamlit Dashboard — Tourism RAG + Predictive Analytics System
=====================================================================
Author : Tourism-AI Team
Version: 1.0.0

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# ─── Local Modules ───────────────────────────────────────────────────────────
from data_preprocessing import (
    TourismDataPreprocessor,
    generate_synthetic_dataset,
)
from model_training import train_all_models
from rag_pipeline   import TourismRAGPipeline
from agents         import (
    TourismMultiAgentSystem,
    build_df_summary,
    build_metrics_summary,
    AgentRole,
)
from visualizations import (
    plot_revenue_distribution,
    plot_top_destinations,
    plot_category_breakdown,
    plot_country_map,
    plot_correlation_heatmap,
    plot_scatter_revenue_visitors,
    plot_rating_distribution,
    plot_high_revenue_pie,
    plot_accommodation_impact,
    plot_confusion_matrix,
    plot_feature_importances,
    plot_metrics_gauge,
    plot_prediction_comparison,
    plot_revenue_per_visitor_heatmap,
    plot_cv_scores,
    plot_model_comparison,
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "TourismAI — Multi-Agent Analytics",
    page_icon  = "🌍",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
LEGACY_CSS = """
<style>
/* ── Global ──────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.block-container { padding: 1.5rem 2rem; }

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #f5f2ff;
    border-right: 1px solid #e5d8fb;
}
[data-testid="stSidebar"] * { color: #374151 !important; }

/* ── Metric Cards ────────────────────────────────────────────────── */
.metric-card {
    background: #ffffff;
    border: 1px solid #e5d8fb;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    transition: transform .2s, box-shadow .2s;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(167,139,250,.12); }
.metric-value { font-size: 2rem; font-weight: 700; color: #5b4a8a; font-family: 'Space Mono', monospace; }
.metric-label { font-size: 0.78rem; color: #7c6fcd; text-transform: uppercase; letter-spacing: .08em; }
.metric-delta { font-size: 0.75rem; color: #7c6fcd; margin-top: 2px; }

/* ── Section Headers ─────────────────────────────────────────────── */
.section-header {
    font-size: 1.5rem; font-weight: 700;
    color: #5b4a8a; margin-bottom: 1.25rem;
    border-left: 4px solid #c4b5fd;
    padding-left: 0.75rem;
}

/* ── Agent Chat ──────────────────────────────────────────────────── */
.chat-user {
    background: #dbeafe; border-radius: 12px 12px 2px 12px;
    padding: .75rem 1rem; margin: .5rem 0; max-width: 80%; float: right; clear: both;
    color: #374151;
}
.chat-agent {
    background: #ffffff; border: 1px solid #e5d8fb;
    border-radius: 12px 12px 12px 2px;
    padding: .75rem 1rem; margin: .5rem 0; max-width: 90%; float: left; clear: both;
    color: #374151;
}
.agent-tag {
    font-size: .7rem; font-weight: 600; letter-spacing: .1em;
    color: #7c6fcd; text-transform: uppercase; margin-bottom: .3rem;
}
.clearfix { clear: both; }

/* ── Status Badges ───────────────────────────────────────────────── */
.badge-success { background:#ecfdf5; color:#065f46; padding:2px 10px; border-radius:999px; font-size:.75rem; font-weight:600; }
.badge-warn    { background:#fffbeb; color:#92400e; padding:2px 10px; border-radius:999px; font-size:.75rem; font-weight:600; }
.badge-info    { background:#eff6ff; color:#3730a3; padding:2px 10px; border-radius:999px; font-size:.75rem; font-weight:600; }

/* ── Recommendation Cards ────────────────────────────────────────── */
.rec-card {
    background: #ffffff; border: 1px solid #e5d8fb;
    border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: .75rem;
}
.rec-card h4 { color: #5b4a8a; margin: 0 0 .4rem; }
.rec-card p  { color: #7c6fcd; font-size: .85rem; margin: 0; }

/* ── Hide Streamlit branding ─────────────────────────────────────── */
#MainMenu, footer { visibility: hidden; }
header { background: transparent; }
</style>
"""


# ═══════════════════════════════════════════════════════════════════════════════
GLOBAL_CSS = """
<style>
:root {
    --bg-main: #faf8ff;
    --bg-card: #ffffff;
    --bg-sidebar: #f5f2ff;
    --accent-primary: #c4b5fd;
    --accent-secondary: #5b4a8a;
    --accent-light: #ede9fe;
    --border-color: #e5d8fb;
    --text-primary: #374151;
    --text-secondary: #7c6fcd;
    --text-muted: #9ca3af;
    --success: #a7f3d0;
    --warning: #fde68a;
    --error: #fca5a5;
    --user-bubble: #dbeafe;
    --agent-bubble: #ffffff;
}

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"], .stApp {
    color: #374151;
    font-family: 'Inter', sans-serif;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: var(--bg-main) !important;
}
.block-container { padding: 1.5rem 2rem 2.5rem; }
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }
hr { border-color: var(--border-color) !important; opacity: .75; }

h1, h1 span {
    color: #5b4a8a !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}
h2, h2 span {
    color: #5b4a8a !important;
    font-weight: 600 !important;
}
h3, h3 span {
    color: #5b4a8a !important;
    font-weight: 500 !important;
}
p, li, label, [data-testid="stMarkdownContainer"] { color: #374151; }

[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
    background: var(--bg-sidebar) !important;
}
[data-testid="stSidebar"] {
    border-right: 1px solid var(--border-color);
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h3 span {
    color: var(--text-secondary) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: .05em !important;
    text-transform: uppercase !important;
}

.section-header, .chat-page-title {
    border-left: 4px solid var(--accent-primary);
    color: var(--accent-secondary);
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 1.25rem;
    padding-left: .75rem;
}

.metric-card, .rec-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(167,139,250,.08);
    margin-bottom: .75rem;
    padding: 1.25rem 1.5rem;
    transition: transform .2s ease, box-shadow .2s ease;
}
.metric-card:hover, .rec-card:hover {
    box-shadow: 0 6px 18px rgba(167,139,250,.14);
    transform: translateY(-1px);
}
.metric-value {
    color: var(--accent-secondary);
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
}
.metric-label {
    color: var(--text-secondary);
    font-size: .78rem;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
}
.metric-delta {
    color: var(--text-muted);
    font-size: .75rem;
    margin-top: 2px;
}
.rec-card h4 { color: var(--accent-secondary); margin: 0 0 .4rem; }
.rec-card p { color: var(--text-secondary); font-size: .85rem; margin: 0; }

[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(167,139,250,.08);
    padding: 1rem;
}
[data-testid="stMetricLabel"] p { color: var(--text-secondary) !important; }
[data-testid="stMetricValue"] { color: var(--accent-secondary) !important; font-weight: 700 !important; }

.badge-success, .badge-info, .badge-warn {
    border-radius: 999px;
    font-size: .75rem;
    font-weight: 600;
    padding: 2px 10px;
}
.badge-success { background: #ecfdf5; border: 1px solid #a7f3d0; color: #065f46; }
.badge-info { background: #eff6ff; border: 1px solid #bfdbfe; color: #3730a3; }
.badge-warn { background: #fffbeb; border: 1px solid #fde68a; color: #92400e; }

.stButton > button,
[data-testid="stFormSubmitButton"] button,
[data-testid="stBaseButton-primary"],
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #c4b5fd, #a78bfa) !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 8px rgba(124,111,205,.25) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    padding: .5rem 1.5rem !important;
    transition: opacity .2s ease, transform .2s ease, box-shadow .2s ease !important;
}
.stButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover,
[data-testid="stBaseButton-primary"]:hover,
[data-testid="stFileUploaderDropzone"] button:hover {
    opacity: .88 !important;
    transform: translateY(-1px);
}
.stButton > button[kind="secondary"],
[data-testid="stBaseButton-secondary"] {
    background: var(--accent-light) !important;
    border: 1.5px solid var(--border-color) !important;
    box-shadow: none !important;
    color: var(--accent-secondary) !important;
    font-weight: 500 !important;
}

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
    background: #fdfbff !important;
    border: 1.5px solid var(--border-color) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-size: 14px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div:focus-within {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,.15) !important;
}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder {
    color: var(--text-muted) !important;
}

[data-testid="stExpander"] {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    box-shadow: 0 2px 12px rgba(167,139,250,.08);
    overflow: hidden;
}
[data-testid="stExpander"] details { border: none !important; }

.stTabs [data-baseweb="tab-list"] {
    border-bottom: 2px solid var(--border-color);
    gap: .35rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px 8px 0 0;
    color: var(--text-secondary);
}
.stTabs [aria-selected="true"] {
    background: var(--accent-primary) !important;
    color: #ffffff !important;
}
.stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] span {
    color: #ffffff !important;
}

[data-testid="stRadio"] label { color: var(--text-muted) !important; }
[data-testid="stRadio"] input:checked + div,
[data-testid="stRadio"] input:checked + div p,
[data-testid="stRadio"] input:checked + div span {
    color: var(--accent-primary) !important;
    font-weight: 600 !important;
}
[data-testid="stRadio"] [role="radiogroup"] {
    border-bottom: 2px solid var(--border-color);
    padding-bottom: .5rem;
}

[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid var(--border-color);
    border-radius: 8px;
    box-shadow: 0 2px 12px rgba(167,139,250,.08);
    overflow: hidden;
}

[data-testid="stAlert"] {
    border: 0;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(167,139,250,.08);
}
[data-testid="stAlert"][kind="success"] {
    background: #ecfdf5 !important;
    border-left: 4px solid #34d399 !important;
    color: #065f46 !important;
}
[data-testid="stAlert"][kind="warning"] {
    background: #fffbeb !important;
    border-left: 4px solid #f59e0b !important;
    color: #92400e !important;
}
[data-testid="stAlert"][kind="error"] {
    background: #fff1f2 !important;
    border-left: 4px solid #f43f5e !important;
    color: #881337 !important;
}
[data-testid="stAlert"][kind="info"] {
    background: #eff6ff !important;
    border-left: 4px solid #a78bfa !important;
    color: #3730a3 !important;
}
[data-testid="stAlert"] * { color: inherit !important; }

.stSlider [data-baseweb="slider"] > div { color: var(--accent-primary) !important; }
.stSlider [role="slider"] {
    background: var(--accent-primary) !important;
    border-color: var(--accent-primary) !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #fdfbff !important;
    border: 1.5px dashed var(--border-color) !important;
    border-radius: 12px !important;
}
div[data-testid="stHorizontalBlock"] > div { flex: 1 1 0%; }
</style>
"""

# Session State Init
# ═══════════════════════════════════════════════════════════════════════════════

def _init_session() -> None:
    defaults = {
        "df"           : None,
        "processed_df" : None,
        "preprocessor" : None,
        "rag"          : None,
        "mas"          : None,
        "metrics"      : None,
        "clf"          : None,
        "rev_reg"      : None,
        "vis_reg"      : None,
        "chat_history" : [],
        "data_loaded"  : False,
        "models_trained": False,
        "rag_built"    : False,
        "shap_explainer"    : None,
        "shap_values_train" : None,
        "shap_X_train"      : None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## 🌍 TourismAI")
        st.markdown("*Multi-Agent Analytics Platform*")
        st.divider()

        if not st.session_state.get("auto_load_attempted"):
            st.session_state.auto_load_attempted = True
            _AUTO_PATH = "C:/Users/HP-PC/Downloads/tourism dataset/tourism_dataset.csv"
            if not st.session_state.data_loaded:
                try:
                    import pandas as pd
                    raw = pd.read_csv(_AUTO_PATH)
                    _load_and_process(raw)
                    st.sidebar.success("✅ Auto-loaded: tourism_dataset.csv")
                except FileNotFoundError:
                    pass
                except Exception as e:
                    st.sidebar.warning(f"Auto-load failed: {e}")

        # ── Data Upload ──────────────────────────────────────────────────────
        st.markdown("### 📂 Data Source")
        uploaded = st.file_uploader("Upload tourism_dataset.csv", type=["csv"])

        if st.button("🔄 Use Synthetic Dataset", use_container_width=True):
            with st.spinner("Generating synthetic dataset…"):
                raw = generate_synthetic_dataset(n=1200)
                _load_and_process(raw)
            st.success(f"✅ Synthetic dataset ready ({len(st.session_state.df):,} rows)")

        if uploaded:
            if st.session_state.get("last_uploaded_file_id") != uploaded.file_id:
                with st.spinner("Loading dataset…"):
                    import io
                    raw = pd.read_csv(io.BytesIO(uploaded.read()))
                    _load_and_process(raw)
                    st.session_state.last_uploaded_file_id = uploaded.file_id
            st.success(f"✅ Uploaded: {uploaded.name} ({len(st.session_state.df):,} rows)")

        st.divider()

        # ── Training ─────────────────────────────────────────────────────────
        st.markdown("### 🤖 Model Training")
        if st.session_state.data_loaded:
            if st.button("🚀 Train All Models", use_container_width=True, type="primary"):
                _train_models()
        else:
            st.info("Load data first to enable training.")

        st.divider()

        # ── LLM Config ───────────────────────────────────────────────────────
        st.markdown("### 🔑 LLM Configuration")
        api_key = st.text_input(
            "Google Gemini API Key (optional)",
            type     = "password",
            value    = os.getenv("GOOGLE_API_KEY", ""),
            help     = "Leave blank to use intelligent mock responses.",
        )
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        elif "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]

        model = st.selectbox("Model", [
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash-001",
            "gemini-1.5-flash-002",
            "gemini-1.5-flash-8b",
            "gemini-2.0-flash",
        ])
        st.info("After changing the model, click Reinit Agents below to apply.")

        if st.session_state.rag_built and st.button("🔁 Reinit Agents", use_container_width=True):
            _build_agents(api_key, model)
            st.success("Agents re-initialised.")

        # ── LLM Status Indicator ─────────────────────────────────────────
        mas = st.session_state.mas
        if mas is not None:
            if mas.llm is not None:
                st.success("🟢 Gemini active — real LLM responses")
            else:
                st.warning("🟡 Mock mode — no LLM connected")
                if mas.llm_error:
                    with st.expander("⚠️ Why?"):
                        st.error(mas.llm_error)

        st.divider()

        # ── Status ───────────────────────────────────────────────────────────
        st.markdown("### 📊 Pipeline Status")
        _status("Data Loaded",     st.session_state.data_loaded)
        _status("Models Trained",  st.session_state.models_trained)
        _status("RAG Built",       st.session_state.rag_built)

        if st.session_state.models_trained and st.session_state.metrics:
            m = st.session_state.metrics.get("classification", {})
            cv_s = m.get("cv_summary", {})
            cv_acc = cv_s.get('cv_accuracy_mean', m.get('accuracy', '–'))
            cv_std = cv_s.get('cv_accuracy_std', '')
            cv_str = f"{cv_acc}%" + (f" ±{cv_std}" if cv_std else "")
            st.markdown(f"""
            <div style='background:#ffffff;border:1px solid var(--border-color);border-radius:8px;padding:.75rem;margin-top:.5rem'>
            <div style='font-size:.7rem;color:var(--text-secondary);text-transform:uppercase'>Stacking v2</div>
            <div style='color:#065f46;font-weight:700;font-size:1.1rem'>
              {m.get('accuracy','–')}% Acc / {m.get('recall','–')}% Recall
            </div>
            <div style='font-size:.7rem;color:var(--accent-secondary);margin-top:2px'>
              CV Acc: {cv_str}
            </div>
            </div>
            """, unsafe_allow_html=True)

    return model


def _status(label: str, ok: bool) -> None:
    badge = "badge-success" if ok else "badge-warn"
    icon  = "✅" if ok else "⏳"
    st.markdown(
        f"<span class='{badge}'>{icon} {label}</span>",
        unsafe_allow_html=True
    )
    st.write("")


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_and_process(raw: pd.DataFrame) -> None:
    prep = TourismDataPreprocessor()
    df   = prep.preprocess(raw)
    st.session_state.df            = raw
    st.session_state.processed_df  = df
    st.session_state.preprocessor  = prep
    st.session_state.data_loaded   = True
    # Build RAG immediately
    _build_rag(df)


def _build_rag(df: pd.DataFrame) -> None:
    with st.spinner("Building RAG vectorstore…"):
        rag = TourismRAGPipeline()
        n   = rag.ingest_dataframe(df)
        st.session_state.rag       = rag
        st.session_state.rag_built = True
    # Build agents (mock mode until API key set)
    _build_agents()


def _build_agents(api_key: str = "", model: str = "gemini-1.5-flash-latest") -> None:
    rag = st.session_state.rag
    if rag is None:
        return
    mas = TourismMultiAgentSystem(
        rag_pipeline   = rag,
        google_api_key = api_key or os.getenv("GOOGLE_API_KEY"),
        model_name     = model,
    )
    if st.session_state.processed_df is not None and st.session_state.metrics:
        eda = st.session_state.preprocessor.get_eda_summary(st.session_state.processed_df)
        mas.set_context(
            df_summary    = build_df_summary(st.session_state.processed_df, eda),
            model_metrics = build_metrics_summary(st.session_state.metrics),
        )
    st.session_state.mas = mas


def _train_models() -> None:
    df   = st.session_state.processed_df
    prep = st.session_state.preprocessor
    if df is None or prep is None:
        st.error("No data loaded.")
        return
    # Pre-load SHAP artifacts from disk if already saved from a previous run
    if SHAP_AVAILABLE and st.session_state.shap_explainer is None:
        try:
            st.session_state.shap_explainer    = joblib.load("models/shap_explainer.joblib")
            st.session_state.shap_values_train = joblib.load("models/shap_values_train.joblib")
            st.session_state.shap_X_train      = joblib.load("models/shap_X_train.joblib")
        except FileNotFoundError:
            pass
    bar  = st.progress(0, "Training classifer…")
    try:
        metrics, clf, rev_reg, vis_reg = train_all_models(df, prep)
        bar.progress(100, "Training complete!")
        st.session_state.metrics       = metrics
        st.session_state.clf           = clf
        st.session_state.rev_reg       = rev_reg
        st.session_state.vis_reg       = vis_reg
        st.session_state.models_trained = True
        # ── SHAP explainer ───────────────────────────────────────────────────
        if SHAP_AVAILABLE:
            try:
                # Extract the XGBoost base estimator from inside the stacking model.
                # clf.model is CalibratedClassifierCV → .estimator is StackingClassifier
                # → .estimators_ is list of fitted (name, estimator) tuples
                calibrated    = clf.model                         # CalibratedClassifierCV
                xgb_estimator = None

                try:
                    # Path 1: calibrated.estimator is the StackingClassifier directly
                    stacker = calibrated.estimator
                    if hasattr(stacker, "estimators_"):
                        for item in stacker.estimators_:
                            est  = item[1] if isinstance(item, tuple) else item
                            name = item[0] if isinstance(item, tuple) else type(est).__name__
                            if "xgb" in name.lower() or "xgb" in type(est).__name__.lower():
                                xgb_estimator = est
                                break

                    # Path 2: dig through calibrated_classifiers_ (cv=3 creates 3 folds)
                    if xgb_estimator is None and hasattr(calibrated, "calibrated_classifiers_"):
                        for cal_clf in calibrated.calibrated_classifiers_:
                            base = getattr(cal_clf, "estimator", None)
                            if base is None:
                                continue
                            if "xgb" in type(base).__name__.lower():
                                xgb_estimator = base
                                break
                            if hasattr(base, "estimators_"):
                                for item in base.estimators_:
                                    est = item[1] if isinstance(item, tuple) else item
                                    if "xgb" in type(est).__name__.lower():
                                        xgb_estimator = est
                                        break
                            if xgb_estimator is not None:
                                break
                except Exception as dig_err:
                    st.warning(f"Could not extract XGBoost from model structure: {dig_err}")

                if xgb_estimator is not None:
                    explainer   = shap.TreeExplainer(xgb_estimator)
                    clf_feats   = prep.get_classification_features()
                    X_tr_shap, _, _, _ = prep.split(
                        df, features=clf_feats, target="High_Revenue_Potential"
                    )
                    shap_vals   = explainer.shap_values(X_tr_shap)
                    joblib.dump(explainer,  "models/shap_explainer.joblib")
                    joblib.dump(shap_vals,  "models/shap_values_train.joblib")
                    joblib.dump(X_tr_shap,  "models/shap_X_train.joblib")
                    st.session_state.shap_explainer   = explainer
                    st.session_state.shap_values_train = shap_vals
                    st.session_state.shap_X_train     = X_tr_shap
                    st.toast("🔍 SHAP explainer saved!", icon="✅")
            except Exception as e:
                st.warning(f"SHAP explainer could not be built: {e}")

        # Update agent context
        eda = prep.get_eda_summary(df)
        if st.session_state.mas:
            st.session_state.mas.set_context(
                df_summary    = build_df_summary(df, eda),
                model_metrics = build_metrics_summary(metrics),
            )
        st.toast("🎉 All models trained!", icon="🚀")
    except Exception as e:
        bar.empty()
        st.error(f"Training failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Pages
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. Overview ──────────────────────────────────────────────────────────────

def page_overview() -> None:
    st.markdown('<div class="section-header">🌍 System Overview</div>',
                unsafe_allow_html=True)

    st.markdown("""
    Welcome to the **TourismAI Multi-Agent Analytics Platform** — an end-to-end
    system combining machine learning, RAG-powered knowledge retrieval, and
    multi-agent reasoning for intelligent tourism management.
    """)

    cols = st.columns(4)
    cards = [
        ("📊", "EDA",          "Explore distributions, trends, and correlations"),
        ("🤖", "ML Models",    "XGBoost + LightGBM ensemble, ≥95% accuracy"),
        ("💬", "AI Agents",    "4 specialised agents powered by RAG + LLM"),
        ("🎯", "Destinations", "Smart recommendations for development"),
    ]
    for col, (icon, title, desc) in zip(cols, cards):
        col.markdown(f"""
        <div class="metric-card">
          <div style="font-size:2rem">{icon}</div>
          <div class="metric-label">{title}</div>
          <div style="color:var(--text-secondary);font-size:.85rem;margin-top:.3rem">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.data_loaded:
        df  = st.session_state.processed_df
        eda = st.session_state.preprocessor.get_eda_summary(df)
        st.divider()
        st.markdown("### 📈 Dataset at a Glance")
        m1, m2, m3, m4, m5 = st.columns(5)
        _metric_card(m1, "Destinations",  f"{eda['n_rows']:,}",         "total records")
        _metric_card(m2, "Countries",     f"{df['Country'].nunique():,}","unique countries")
        _metric_card(m3, "High Potential",f"{eda['high_rev_pct']}%",    "top-25% revenue")
        _metric_card(m4, "Avg Rating",    f"{eda['avg_rating']}",        "out of 5.0")
        _metric_card(m5, "Avg Rev/Visitor",f"${eda['avg_revenue_per_vis']:.0f}", "per visitor")
    else:
        st.info("👈 Load a dataset from the sidebar to begin.")


def _metric_card(col, label: str, value: str, sublabel: str = "") -> None:
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-delta">{sublabel}</div>
    </div>
    """, unsafe_allow_html=True)


# ── 2. EDA ───────────────────────────────────────────────────────────────────

def page_eda() -> None:
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>',
                unsafe_allow_html=True)
    if not st.session_state.data_loaded:
        st.info("Load a dataset first.")
        return

    df = st.session_state.processed_df

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Distributions", "🗺️ Geography", "🔗 Correlations", "📋 Raw Stats"]
    )

    with tab1:
        c1, c2 = st.columns(2)
        c1.plotly_chart(plot_revenue_distribution(df), use_container_width=True)
        c2.plotly_chart(plot_high_revenue_pie(df),     use_container_width=True)
        st.plotly_chart(plot_top_destinations(df),     use_container_width=True)
        c3, c4 = st.columns(2)
        c3.plotly_chart(plot_category_breakdown(df),   use_container_width=True)
        c4.plotly_chart(plot_rating_distribution(df),  use_container_width=True)
        st.plotly_chart(plot_scatter_revenue_visitors(df), use_container_width=True)

    with tab2:
        st.plotly_chart(plot_country_map(df),           use_container_width=True)
        st.plotly_chart(plot_accommodation_impact(df),  use_container_width=True)
        st.plotly_chart(plot_revenue_per_visitor_heatmap(df), use_container_width=True)

    with tab3:
        st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)
        st.markdown("#### Feature Engineering Results")
        feat_cols = ["Revenue_per_Visitor", "Rating_Revenue_Index",
                     "Popularity_Score", "Visitor_Density_Score"]
        feat_cols = [c for c in feat_cols if c in df.columns]
        st.dataframe(df[feat_cols].describe().round(2), use_container_width=True)

    with tab4:
        st.markdown("#### Dataset Preview")
        display_cols = ["Location", "Country", "Category", "Visitors",
                        "Rating", "Revenue", "Revenue_per_Visitor",
                        "High_Revenue_Potential", "Popularity_Score"]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols].head(100), use_container_width=True, height=400)

        eda = st.session_state.preprocessor.get_eda_summary(df)
        st.markdown("#### Descriptive Statistics")
        st.dataframe(
            pd.DataFrame(eda["numeric_stats"]).T.round(2),
            use_container_width=True,
        )


# ── 3. Models ────────────────────────────────────────────────────────────────

def page_models() -> None:
    st.markdown('<div class="section-header">🤖 Machine Learning Models</div>',
                unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.info("Load a dataset first.")
        return

    if not st.session_state.models_trained:
        st.warning("Models not yet trained. Click **🚀 Train All Models** in the sidebar.")
        return

    metrics = st.session_state.metrics
    clf     = st.session_state.clf

    # ── Performance Gauges ───────────────────────────────────────────────────
    st.plotly_chart(
        plot_metrics_gauge(metrics["classification"]),
        use_container_width=True,
    )

    # ── Detailed Classification Results ─────────────────────────────────────
    st.markdown("### 🎯 Classification: High Revenue Potential")
    c1, c2 = st.columns([1, 1])

    with c1:
        cm = metrics["classification"].get("conf_matrix", [[0, 0], [0, 0]])
        st.plotly_chart(plot_confusion_matrix(cm), use_container_width=True)

    with c2:
        m = metrics["classification"]
        rows = [
            ("Accuracy",  m.get("accuracy",  "–"), "%", "success"),
            ("Recall",    m.get("recall",    "–"), "%", "success"),
            ("Precision", m.get("precision", "–"), "%", "info"),
            ("F1 Score",  m.get("f1",        "–"), "%", "info"),
            ("ROC-AUC",   m.get("roc_auc",   "–"), "%", "info"),
        ]
        for name, val, suf, badge in rows:
            st.markdown(f"""
            <div class="metric-card" style="display:flex;justify-content:space-between;
                                            align-items:center;padding:.75rem 1.25rem">
              <span style="color:var(--text-secondary)">{name}</span>
              <span class="badge-{badge}" style="font-size:1rem;padding:4px 16px">
                {val}{suf}
              </span>
            </div>
            """, unsafe_allow_html=True)

    # ── v2: Cross-Validation Scores ──────────────────────────────────────────
    cv_scores = metrics["classification"].get("cv_scores", {})
    if cv_scores:
        st.markdown("### 📊 Cross-Validation Scores (5-Fold Stratified)")
        cv_s = metrics["classification"].get("cv_summary", {})
        cv_cols = st.columns(5)
        for i, metric_name in enumerate(["accuracy", "recall", "precision", "f1", "roc_auc"]):
            mean_key = f"cv_{metric_name}_mean"
            std_key  = f"cv_{metric_name}_std"
            cv_cols[i].markdown(f"""
            <div class="metric-card" style="text-align:center;padding:.6rem">
              <div class="metric-label">{metric_name.replace('_',' ').title()}</div>
              <div class="metric-value" style="font-size:1.4rem">{cv_s.get(mean_key, '–')}%</div>
              <div class="metric-delta">±{cv_s.get(std_key, '–')}%</div>
            </div>
            """, unsafe_allow_html=True)
        st.plotly_chart(plot_cv_scores(cv_scores), use_container_width=True)

    # ── v2: Model Comparison ─────────────────────────────────────────────────
    comparison = metrics["classification"].get("model_comparison", {})
    if comparison:
        st.markdown("### 🏆 Model Comparison: Individual vs Stacking Ensemble")
        st.plotly_chart(
            plot_model_comparison(comparison, metrics["classification"]),
            use_container_width=True,
        )
        # Show as a table too
        comp_data = dict(comparison)
        comp_data["Stacking Ensemble"] = {
            k: metrics["classification"].get(k, 0)
            for k in ["accuracy", "recall", "precision", "f1", "roc_auc"]
        }
        st.dataframe(
            pd.DataFrame(comp_data).T.round(2).rename(
                columns=lambda c: c.replace('_', ' ').title() + " %"
            ),
            use_container_width=True,
        )

    # ── Feature Importances ──────────────────────────────────────────────────
    if clf and clf.feature_importances_ is not None:
        st.markdown("### 📊 Feature Importances")
        st.plotly_chart(
            plot_feature_importances(clf.feature_importances_),
            use_container_width=True,
        )

    # ── Regression Results ───────────────────────────────────────────────────
    st.markdown("### 📈 Regression Models")
    rc1, rc2 = st.columns(2)

    with rc1:
        rm = metrics["revenue_regression"]
        st.markdown("**Revenue Regressor (Stacking v2)**")
        for k, v in rm.items():
            if k not in ("conf_matrix", "cv_scores"):
                st.metric(k.upper(), f"{v:,.2f}" if isinstance(v, float) else str(v))
        # v2: show CV R²
        rv_cv = rm.get("cv_scores", {})
        if rv_cv:
            st.markdown(f"**CV R²:** {rv_cv.get('r2_mean', '–')}% ± {rv_cv.get('r2_std', '–')}%")

    with rc2:
        vm = metrics["visitors_regression"]
        st.markdown("**Visitors Regressor (Stacking v2)**")
        for k, v in vm.items():
            if k not in ("conf_matrix", "cv_scores"):
                st.metric(k.upper(), f"{v:,.2f}" if isinstance(v, float) else str(v))
        # v2: show CV R²
        vs_cv = vm.get("cv_scores", {})
        if vs_cv:
            st.markdown(f"**CV R²:** {vs_cv.get('r2_mean', '–')}% ± {vs_cv.get('r2_std', '–')}%")


# ── 4. Predict ───────────────────────────────────────────────────────────────

def _prediction_feature_values(
    df: pd.DataFrame,
    visitors: float,
    rating: float,
    revenue: float,
    acc_val: int,
    rev_pv: float,
    log_vis: float,
    log_rev: float,
    pop_score: float,
    cat_val: int,
    cou_val: int,
    rb_enc: int,
    vt_enc: int,
    lf: float,
    category: str,
    country: str,
) -> dict:
    country_rows = df[df["Country"] == country]
    category_rows = df[df["Category"] == category]

    country_rev_mean = country_rows["Revenue"].mean()
    country_vis_mean = country_rows["Visitors"].mean()
    category_rev_density = category_rows["Revenue"].mean()

    return {
        "Visitors": visitors,
        "Rating": rating,
        "Revenue": revenue,
        "Accommodation_Available": acc_val,
        "Revenue_per_Visitor": rev_pv,
        "Visitor_Density_Score": visitors * rating / (revenue + 1e-9),
        "Rating_Revenue_Index": rating * rev_pv,
        "Accommodation_Revenue_Boost": acc_val * rev_pv,
        "Visitors_x_Rating": visitors * rating,
        "Revenue_x_Acc": revenue * acc_val,
        "Rating_Squared": rating ** 2,
        "Log_RevPerVis": np.log1p(rev_pv),
        "Acc_x_Rating": acc_val * rating,
        "Country_Rev_Mean": country_rev_mean if pd.notna(country_rev_mean) else df["Revenue"].mean(),
        "Country_Vis_Mean": country_vis_mean if pd.notna(country_vis_mean) else df["Visitors"].mean(),
        "Category_Rev_Density": (
            category_rev_density if pd.notna(category_rev_density) else df["Revenue"].mean()
        ),
        "Log_Visitors": log_vis,
        "Log_Revenue": log_rev,
        "Popularity_Score": pop_score,
        "Category_Enc": cat_val,
        "Country_Enc": cou_val,
        "Rating_Band_Enc": rb_enc,
        "Visitor_Tier_Enc": vt_enc,
        "Location_Freq": lf,
    }


def page_predict() -> None:
    st.markdown('<div class="section-header">🔮 Interactive Predictions</div>',
                unsafe_allow_html=True)

    if not st.session_state.models_trained:
        st.warning("Train models first.")
        return

    df    = st.session_state.processed_df
    prep  = st.session_state.preprocessor
    clf   = st.session_state.clf
    rev   = st.session_state.rev_reg
    vis   = st.session_state.vis_reg

    st.markdown("Enter destination details to get predictions from all three models.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        visitors   = c1.number_input("Annual Visitors",    1000, 10_000_000, 50_000, 1000)
        rating     = c2.slider("Rating (0–5)",             0.0, 5.0, 3.5, 0.1)
        revenue_in = c3.number_input("Revenue ($)",        1000, 500_000_000, 500_000, 10000)

        c4, c5, c6 = st.columns(3)
        category   = c4.selectbox("Category", sorted(df["Category"].unique()))
        country    = c5.selectbox("Country",  sorted(df["Country"].unique()))
        acc        = c6.radio("Accommodation", ["Yes", "No"])

        submitted = st.form_submit_button("🚀 Predict", use_container_width=True)

    if submitted:
        acc_val  = 1 if acc == "Yes" else 0
        rev_pv   = revenue_in / visitors if visitors > 0 else 0
        log_vis  = np.log1p(visitors)
        log_rev  = np.log1p(revenue_in)
        pop_score = min(100, (visitors / 100_000 * 30 + rating * 15 + rev_pv / 50 * 10))

        cat_enc = prep.label_encoders.get("Category")
        cou_enc = prep.label_encoders.get("Country")
        try:
            cat_val = cat_enc.transform([category])[0] if cat_enc else 0
        except Exception:
            cat_val = 0
        try:
            cou_val = cou_enc.transform([country])[0] if cou_enc else 0
        except Exception:
            cou_val = 0

        # rating_band_enc, visitor_tier_enc, location_freq
        rb_enc = 2  # "High" bucket default
        vt_enc = 2  # "Gold" bucket default
        lf     = 0.01  # default location frequency
        values = _prediction_feature_values(
            df=df,
            visitors=visitors,
            rating=rating,
            revenue=revenue_in,
            acc_val=acc_val,
            rev_pv=rev_pv,
            log_vis=log_vis,
            log_rev=log_rev,
            pop_score=pop_score,
            cat_val=cat_val,
            cou_val=cou_val,
            rb_enc=rb_enc,
            vt_enc=vt_enc,
            lf=lf,
            category=category,
            country=country,
        )

        # ── Classification ───────────────────────────────────────────────────
        clf_feat_names = prep.get_classification_features()
        X_clf = pd.DataFrame([[values.get(f, 0) for f in clf_feat_names]],
                             columns=clf_feat_names)
        hrp_pred  = clf.predict(X_clf)[0]
        hrp_proba = clf.predict_proba(X_clf)[0]

        # ── Revenue Regression ───────────────────────────────────────────────
        rev_feat_names = prep.get_revenue_regression_features()
        X_rev = pd.DataFrame([[values.get(f, 0) for f in rev_feat_names]],
                             columns=rev_feat_names)
        rev_pred = rev.predict(X_rev)[0]
        rev_lo, rev_hi = rev.predict_interval(X_rev, confidence=0.90)

        # ── Visitors Regression ──────────────────────────────────────────────
        vis_feat_names = prep.get_visitors_regression_features()
        X_vis = pd.DataFrame([[values.get(f, 0) for f in vis_feat_names]],
                             columns=vis_feat_names)
        vis_pred = vis.predict(X_vis)[0]
        vis_lo, vis_hi = vis.predict_interval(X_vis, confidence=0.90)

        # ── Display ──────────────────────────────────────────────────────────
        st.divider()
        r1, r2, r3 = st.columns(3)

        badge  = "badge-success" if hrp_pred == 1 else "badge-warn"
        label  = "HIGH REVENUE POTENTIAL ✅" if hrp_pred == 1 else "Standard Revenue ⚠️"
        r1.markdown(f"""
        <div class="metric-card" style="text-align:center">
          <div class="metric-label">Revenue Potential</div>
          <div style="margin:.5rem 0"><span class="{badge}"
          style="font-size:1rem;padding:6px 18px">{label}</span></div>
          <div class="metric-delta">Confidence: {hrp_proba:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        r2.markdown(f"""
        <div class="metric-card" style="text-align:center">
          <div class="metric-label">Predicted Revenue</div>
          <div class="metric-value">${rev_pred:,.0f}</div>
          <div class="metric-delta">90% CI: ${rev_lo[0]:,.0f} - ${rev_hi[0]:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        r3.markdown(f"""
        <div class="metric-card" style="text-align:center">
          <div class="metric-label">Predicted Visitors</div>
          <div class="metric-value">{int(vis_pred):,}</div>
          <div class="metric-delta">90% CI: {int(vis_lo[0]):,} - {int(vis_hi[0]):,}</div>
        </div>
        """, unsafe_allow_html=True)
        # ── SHAP: Why this prediction? ───────────────────────────────────────
        st.divider()

        if not SHAP_AVAILABLE:
            st.info("Install `shap` (`pip install shap`) to enable explainability charts.")
        else:
            explainer   = st.session_state.get("shap_explainer")
            shap_vals   = st.session_state.get("shap_values_train")
            shap_X_tr   = st.session_state.get("shap_X_train")

            if explainer is None:
                # Try loading from disk (handles page refresh without retraining)
                try:
                    explainer = joblib.load("models/shap_explainer.joblib")
                    shap_vals = joblib.load("models/shap_values_train.joblib")
                    shap_X_tr = joblib.load("models/shap_X_train.joblib")
                    st.session_state.shap_explainer    = explainer
                    st.session_state.shap_values_train = shap_vals
                    st.session_state.shap_X_train      = shap_X_tr
                except FileNotFoundError:
                    st.info("SHAP explainer not found. Train models first to generate it.")
                    explainer = None

            if explainer is not None:
                # ── Per-prediction waterfall chart ───────────────────────────
                st.markdown("### 🔍 Why this prediction?")
                st.caption(
                    "The chart below shows which features pushed the model towards "
                    "or away from classifying this destination as High Revenue Potential. "
                    "Red bars increase the prediction; blue bars decrease it."
                )
                try:
                    import matplotlib.pyplot as plt
                    shap_vals_single = explainer.shap_values(X_clf)
                    expected_val     = explainer.expected_value

                    # shap_values may be a list [class0, class1] for binary classifiers
                    if isinstance(shap_vals_single, list):
                        sv = shap_vals_single[1][0]   # class-1 SHAP values, first row
                        ev = expected_val[1] if hasattr(expected_val, '__len__') else expected_val
                    else:
                        sv = shap_vals_single[0]
                        ev = expected_val

                    explanation = shap.Explanation(
                        values        = sv,
                        base_values   = ev,
                        data          = X_clf.values[0],
                        feature_names = list(X_clf.columns),
                    )

                    fig_wf, ax_wf = plt.subplots(figsize=(10, 5))
                    shap.plots.waterfall(explanation, max_display=12, show=False)
                    fig_wf = plt.gcf()
                    fig_wf.patch.set_facecolor("#faf8ff")
                    st.pyplot(fig_wf, use_container_width=True)
                    plt.close(fig_wf)
                except Exception as e:
                    st.warning(f"Waterfall chart could not be rendered: {e}")

                # ── Global feature importance summary ────────────────────────
                with st.expander("📊 Global Feature Importance (SHAP)", expanded=False):
                    st.caption(
                        "Average absolute SHAP value across the training set — "
                        "shows which features matter most to the model overall."
                    )
                    try:
                        import matplotlib.pyplot as plt
                        fig_sum, _ = plt.subplots(figsize=(10, 5))

                        sv_global = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

                        shap.summary_plot(
                            sv_global,
                            shap_X_tr,
                            plot_type  = "bar",
                            max_display = 15,
                            show       = False,
                        )
                        fig_sum = plt.gcf()
                        fig_sum.patch.set_facecolor("#faf8ff")
                        st.pyplot(fig_sum, use_container_width=True)
                        plt.close(fig_sum)
                    except Exception as e:
                        st.warning(f"Summary plot could not be rendered: {e}")


# ── 5. Agent Chat ─────────────────────────────────────────────────────────────

def page_chat() -> None:
    # ── Pastel Chat CSS ───────────────────────────────────────────────────────
    st.markdown("""
    <style>
    /* ── Page Title ──────────────────────────────────────────────────── */
    .chat-page-title {
        color: #5b4a8a;
        font-weight: 700;
        font-size: 1.5rem;
        margin-bottom: 1.25rem;
        border-left: 4px solid #a78bfa;
        padding-left: 0.75rem;
    }

    /* ── Chat Container Card ─────────────────────────────────────────── */
    .chat-card {
        background: #ffffff;
        border: 1px solid var(--border-color);
        border-bottom: 0;
        border-radius: 16px 16px 0 0;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(167,139,250,0.08);
        margin-bottom: 0;
    }
    .chat-page-wrap [data-testid="stForm"],
    [data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid var(--border-color);
        border-radius: 0 0 16px 16px;
        border-top: 0;
        box-shadow: 0 10px 18px rgba(167,139,250,0.08);
        padding: 0 1.5rem 1.5rem;
        margin-bottom: 1.25rem;
    }

    /* ── Chat History Scroll Area ─────────────────────────────────────── */
    .chat-scroll {
        max-height: 500px;
        overflow-y: auto;
        padding-right: 8px;
        scrollbar-width: thin;
        scrollbar-color: var(--border-color) transparent;
    }
    .chat-scroll::-webkit-scrollbar { width: 6px; }
    .chat-scroll::-webkit-scrollbar-track { background: transparent; }
    .chat-scroll::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 3px;
    }

    /* ── User Bubble ─────────────────────────────────────────────────── */
    .user-bubble {
        background: var(--user-bubble);
        color: #1e3a5f;
        border-radius: 18px 18px 4px 18px;
        padding: 10px 16px;
        max-width: 72%;
        margin-left: auto;
        margin-bottom: 12px;
        font-size: 14px;
        box-shadow: 0 2px 8px rgba(100,160,255,0.15);
        word-wrap: break-word;
    }
    .user-bubble-label {
        text-align: right;
        font-size: 10px;
        font-weight: 600;
        color: #60a5fa;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0;
    }

    /* ── Agent Bubble ────────────────────────────────────────────────── */
    .agent-bubble {
        background: var(--agent-bubble);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        border-radius: 18px 18px 18px 4px;
        padding: 12px 16px;
        max-width: 85%;
        margin-right: auto;
        margin-bottom: 12px;
        font-size: 14px;
        box-shadow: 0 2px 8px rgba(180,160,220,0.12);
        word-wrap: break-word;
    }
    .agent-bubble-label {
        font-size: 10px;
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ── Clearfix ────────────────────────────────────────────────────── */
    .chat-clearfix { clear: both; }

    /* ── Agent Dropdown ──────────────────────────────────────────────── */
    .chat-page-wrap [data-testid="stSelectbox"] > div > div {
        background: #fdfbff;
        border: 1.5px solid var(--border-color);
        border-radius: 10px;
        color: var(--text-primary);
    }

    /* ── Chat Input Area (text_area) ─────────────────────────────────── */
    .chat-page-wrap [data-testid="stTextArea"] textarea,
    [data-testid="stForm"] [data-testid="stTextArea"] textarea {
        background: #fdfbff !important;
        border: 1.5px solid var(--border-color) !important;
        border-radius: 12px !important;
        font-size: 14px !important;
        padding: 10px 14px !important;
        color: var(--text-primary) !important;
    }
    .chat-page-wrap [data-testid="stTextArea"] textarea:focus,
    [data-testid="stForm"] [data-testid="stTextArea"] textarea:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px rgba(167,139,250,0.15) !important;
    }

    /* ── Send Button (primary / first form_submit) ───────────────────── */
    .chat-page-wrap [data-testid="stFormSubmitButton"]:first-of-type button {
        background: linear-gradient(135deg, #c4b5fd, #a78bfa) !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.5rem 2rem !important;
        box-shadow: 0 2px 8px rgba(124,111,205,0.25) !important;
        transition: opacity 0.2s ease, transform 0.2s ease;
    }
    .chat-page-wrap [data-testid="stFormSubmitButton"]:first-of-type button:hover {
        opacity: 0.88 !important;
        transform: translateY(-1px);
    }

    /* ── Clear Button (second form_submit) ───────────────────────────── */
    .chat-page-wrap [data-testid="stFormSubmitButton"]:last-of-type button {
        background: var(--accent-light) !important;
        color: var(--accent-secondary) !important;
        border: 1.5px solid var(--border-color) !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: background 0.2s ease;
    }
    .chat-page-wrap [data-testid="stFormSubmitButton"]:last-of-type button:hover {
        background: #f5f2ff !important;
    }

    /* ── Quick Prompt Buttons ────────────────────────────────────────── */
    .chat-page-wrap .quick-prompts-section button {
        background: #ffffff !important;
        border: 1px solid #e5d8fb !important;
        border-radius: 20px !important;
        color: #5b4a8a !important;
        font-size: 13px !important;
        padding: 6px 16px !important;
        transition: background 0.2s ease;
    }
    .chat-page-wrap .quick-prompts-section button:hover {
        background: #f5f2ff !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Wrap entire page for scoped CSS ───────────────────────────────────────
    st.markdown('<div class="chat-page-wrap">', unsafe_allow_html=True)

    st.markdown('<div class="chat-page-title">💬 Multi-Agent Chat</div>',
                unsafe_allow_html=True)

    if not st.session_state.rag_built:
        st.info("Load a dataset first to initialise agents.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    mas = st.session_state.mas

    # ── Agent selector ────────────────────────────────────────────────────────
    agent_options = {
        "🤖 Auto-Route"         : None,
        "📊 Data Analyst"       : "data_analyst",
        "🔮 Prediction Agent"   : "prediction",
        "🎯 Recommendation"     : "recommendation",
        "🌱 Sustainability"     : "sustainability",
    }
    sel_label = st.selectbox("Select Agent", list(agent_options.keys()))
    agent_role = agent_options[sel_label]

    # ── Quick Prompts ─────────────────────────────────────────────────────────
    st.markdown("#### 💡 Quick Prompts")
    st.markdown('<div class="quick-prompts-section">', unsafe_allow_html=True)
    quick = [
        "What are the top 5 highest revenue destinations?",
        "Which category has the best revenue per visitor?",
        "What strategies would you recommend for eco-tourism growth?",
        "How does accommodation availability affect destination revenue?",
        "What sustainability measures reduce over-tourism risk?",
    ]
    cols = st.columns(3)
    for i, q in enumerate(quick):
        if cols[i % 3].button(q, key=f"qp_{i}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": q})
            with st.spinner("Agent thinking…"):
                result = mas.chat(q, agent_role=agent_role)
            st.session_state.chat_history.append({
                "role"   : "assistant",
                "agent"  : result["agent"],
                "content": result["response"],
            })
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Display chat history ──────────────────────────────────────────────────
    chat_html_parts = []
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            chat_html_parts.append(
                '<div class="user-bubble-label">You</div>'
                f'<div class="user-bubble">{msg["content"]}</div>'
                '<div class="chat-clearfix"></div>'
            )
        else:
            agent_name = msg.get("agent", "Agent")
            chat_html_parts.append(
                f'<div class="agent-bubble-label">{agent_name}</div>'
                f'<div class="agent-bubble">{msg["content"]}</div>'
                '<div class="chat-clearfix"></div>'
            )

    chat_body = "\n".join(chat_html_parts) if chat_html_parts else (
        '<div style="text-align:center;color:var(--text-muted);padding:2rem 0;font-size:14px;">'
        'No messages yet — ask something below!</div>'
    )

    st.markdown(
        f'<div class="chat-history" style="max-height:520px;overflow-y:auto;display:flex;flex-direction:column;gap:12px;padding:1rem">{chat_body}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <script>
    const chatContainer = document.querySelector('.chat-history');
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    // fallback: scroll the main content area
    window.scrollTo(0, document.body.scrollHeight);
    </script>
    """, unsafe_allow_html=True)

    # ── Input ─────────────────────────────────────────────────────────────────
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Ask the Tourism AI…",
            placeholder=(
                "e.g. What are the top revenue-driving categories?\n"
                "What should we invest in for beach destinations?\n"
                "How can we make mountain tourism more sustainable?"
            ),
            height=80,
        )
        c1, c2 = st.columns([4, 1])
        submit = c1.form_submit_button("Send 🚀", use_container_width=True)
        clear  = c2.form_submit_button("Clear",   use_container_width=True)

    if clear:
        st.session_state.chat_history = []
        st.rerun()

    if submit and user_input.strip():
        st.session_state.chat_history.append({
            "role": "user", "content": user_input.strip()
        })
        with st.spinner("Agent thinking…"):
            result = mas.chat(user_input.strip(), agent_role=agent_role)

        st.session_state.chat_history.append({
            "role"   : "assistant",
            "agent"  : result["agent"],
            "content": result["response"],
        })
        st.rerun()

    # ── Close page wrapper ────────────────────────────────────────────────────
    st.markdown('</div>', unsafe_allow_html=True)


# ── 6. Recommendations ────────────────────────────────────────────────────────

def page_recommendations() -> None:
    st.markdown('<div class="section-header">🎯 Destination Recommendations</div>',
                unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.info("Load a dataset first.")
        return

    df  = st.session_state.processed_df
    mas = st.session_state.mas

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.expander("🔍 Filter Destinations", expanded=True):
        c1, c2, c3 = st.columns(3)
        cat_filter = c1.multiselect("Category",  df["Category"].unique().tolist(),
                                    default=df["Category"].unique().tolist())
        cou_filter = c2.multiselect("Country",   df["Country"].unique().tolist(),
                                    default=df["Country"].unique().tolist()[:10])
        min_rating = c3.slider("Min Rating",     0.0, 5.0, 3.5, 0.1)
        hrp_only   = st.checkbox("Show High Revenue Potential only", value=False)

    mask = (
        df["Category"].isin(cat_filter)
        & df["Country"].isin(cou_filter)
        & (df["Rating"] >= min_rating)
    )
    if hrp_only:
        mask &= df["High_Revenue_Potential"] == 1
    filtered = df[mask].copy()

    st.markdown(f"**{len(filtered):,} destinations match your filters.**")

    # ── Top Table ─────────────────────────────────────────────────────────────
    display = filtered.nlargest(50, "Popularity_Score")[[
        "Location", "Country", "Category", "Visitors", "Rating",
        "Revenue", "Revenue_per_Visitor", "High_Revenue_Potential", "Popularity_Score"
    ]].copy()
    display["High_Revenue_Potential"] = display["High_Revenue_Potential"].map({1: "✅ Yes", 0: "No"})
    display["Revenue"]               = display["Revenue"].map("${:,.0f}".format)
    display["Revenue_per_Visitor"]   = display["Revenue_per_Visitor"].map("${:.1f}".format)
    display["Popularity_Score"]      = display["Popularity_Score"].map("{:.1f}".format)

    st.dataframe(display.rename(columns={
        "Revenue_per_Visitor": "Rev/Visitor",
        "High_Revenue_Potential": "High Potential",
        "Popularity_Score": "Score",
    }), use_container_width=True, height=400)

    # ── AI Recommendations ────────────────────────────────────────────────────
    st.markdown("### 🤖 AI Development Recommendations")

    if not filtered.empty and mas:
        top3 = filtered.nlargest(3, "Popularity_Score")
        for _, row in top3.iterrows():
            with st.expander(f"🏖️ {row['Location']}, {row['Country']} "
                             f"(Score: {row.get('Popularity_Score', 0):.1f})"):
                query = (
                    f"Give specific development recommendations for {row['Location']} "
                    f"in {row['Country']}, a {row['Category']} destination with "
                    f"{int(row['Visitors']):,} annual visitors, "
                    f"rating {row['Rating']:.1f}/5, "
                    f"revenue ${int(row['Revenue'].replace('$','').replace(',','')) if isinstance(row['Revenue'], str) else int(row['Revenue']):,}, "
                    f"accommodation {'available' if row.get('Accommodation_Available',0)==1 else 'unavailable'}."
                )
                if st.button(f"Generate AI Plan for {row['Location']}", key=f"rec_btn_{row['Location']}"):
                    with st.spinner(f"Generating recommendations for {row['Location']}…"):
                        result = mas.chat(query, agent_role="recommendation")
                    st.markdown(result["response"])


# ═══════════════════════════════════════════════════════════════════════════════
# Main Router
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    model = render_sidebar()

    pages = {
        "🏠 Overview"           : page_overview,
        "📊 EDA"                : page_eda,
        "🤖 Models"             : page_models,
        "🔮 Predict"            : page_predict,
        "💬 Agent Chat"         : page_chat,
        "🎯 Recommendations"    : page_recommendations,
    }

    c1, c2 = st.columns([6, 1])
    with c2:
        if st.button("🔄 Reset UI", help="Refresh page to reset sidebar state"):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown(
        "<small style='color:var(--text-secondary)'>💡 Tip: Click the ☰ menu in top-left to reopen the sidebar, or use the Reset UI button above</small>",
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        list(pages.keys()),
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("---")
    pages[page]()

    # Reinit agents with current model if API key changed
    if st.session_state.rag_built and st.session_state.mas is None:
        _build_agents(model=model)


if __name__ == "__main__":
    main()
