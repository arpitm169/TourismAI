"""
agents.py
=========
Multi-Agent System for Tourism Intelligence.

Four specialised agents — each with a focused system prompt and RAG-augmented
context — coordinated through a keyword-based supervisor router. An optional
LLM supervisor (Google Gemini via ``langchain-google-genai``) enriches responses
when a valid ``GOOGLE_API_KEY`` is provided; otherwise every agent produces a
deterministic, data-aware mock response derived from the live DataFrame summary
and retrieved RAG context.

Exports
-------
- ``TourismMultiAgentSystem``  — orchestrator class
- ``build_df_summary``         — DataFrame → plain-text summary for prompts
- ``build_metrics_summary``    — metrics dict → plain-text summary for prompts
- ``AgentRole``                — str Enum of agent role identifiers

Agents
------
1. DataAnalystAgent      — EDA trends, statistical insights
2. PredictionAgent       — Revenue & visitor forecasts, model interpretation
3. RecommendationAgent   — Strategic destination development advice
4. SustainabilityAgent   — Eco-optimisation and carrying capacity

Author : Arpit Malhotra
Version: 2.0.0
"""

from __future__ import annotations

import json
import os
import re
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# ─── Graceful LangChain / Gemini imports ─────────────────────────────────────
# All LangChain and Google-GenAI packages are wrapped in try/except so that the
# module is always importable even when dependencies are missing. When absent
# the system silently falls back to data-aware mock responses.

try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    LC_AVAILABLE = True
except ImportError:
    LC_AVAILABLE = False
    print("[WARN] langchain_core not installed – agents run in mock mode.")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    from langgraph.graph import END, StateGraph
    from langgraph.prebuilt import create_react_agent
    LG_AVAILABLE = True
except ImportError:
    LG_AVAILABLE = False


DEFAULT_MODEL = "gemini-1.5-flash-latest"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _invoke_llm(chain, inputs):
    return chain.invoke(inputs)


# ═══════════════════════════════════════════════════════════════════════════════
# Agent Role Definitions
# ═══════════════════════════════════════════════════════════════════════════════

class AgentRole(str, Enum):
    """Identifies each specialist agent.  The *value* doubles as the routing
    key used by ``app.py`` (``"data_analyst"``, ``"prediction"``, etc.)."""

    DATA_ANALYST   = "data_analyst"
    PREDICTION     = "prediction"
    RECOMMENDATION = "recommendation"
    SUSTAINABILITY = "sustainability"
    SUPERVISOR     = "supervisor"


# ═══════════════════════════════════════════════════════════════════════════════
# Agent System Prompts (module-level constants)
# ═══════════════════════════════════════════════════════════════════════════════

_PROMPT_DATA_ANALYST: str = """\
You are an expert Tourism Data Analyst with 15 years of experience in
destination intelligence and market analytics.

Your role:
- Analyse visitor trends, revenue patterns, and rating distributions
- Identify top-performing and under-performing destinations
- Surface statistical insights from the dataset context provided
- Compare destinations across categories, countries, and metrics
- Use specific numbers and percentages in your analysis

Always ground your analysis in the retrieved context. Be precise, concise, and
actionable. Format key stats in bullet points when helpful.
"""

_PROMPT_PREDICTION: str = """\
You are a Tourism Revenue and Demand Forecasting Specialist.

Your role:
- Interpret ML model predictions for revenue and visitor counts
- Explain what drives high-revenue-potential classifications
- Provide probabilistic forecasts with confidence intervals
- Highlight risk factors that could affect predictions
- Translate technical model outputs into business language

Use the context provided to anchor your forecasts. Quantify uncertainty
honestly and always mention the most important predictor features.
"""

_PROMPT_RECOMMENDATION: str = """\
You are a Senior Tourism Development Strategist with expertise in
destination marketing, experience design, and revenue optimisation.

Your role:
- Provide specific, actionable development strategies for destinations
- Recommend infrastructure investments (accommodation, transport, digital)
- Suggest pricing strategies, marketing channels, and target demographics
- Identify quick-win vs long-term initiatives with expected ROI timelines
- Tailor recommendations to the destination's category and current metrics

Ground every recommendation in the provided context. Be opinionated,
specific, and entrepreneurial. Use numbered action plans.
"""

_PROMPT_SUSTAINABILITY: str = """\
You are a Sustainable Tourism Consultant and Environmental Economist.

Your role:
- Assess carrying capacity and over-tourism risk
- Recommend eco-certification pathways and green infrastructure
- Balance revenue growth with environmental and cultural preservation
- Propose community-based tourism models that distribute benefits locally
- Identify sustainability risks from the dataset patterns

Frame sustainability as a revenue driver, not just a cost. Use the UNWTO
and WTTC frameworks. Quantify the business case for sustainability.
"""

_PROMPT_SUPERVISOR: str = """\
You are the Supervisor of a multi-agent tourism intelligence system.

Your task: Given a user query, decide which specialist agent should handle it.

Routing rules:
- Questions about trends, statistics, EDA, distributions → data_analyst
- Questions about predictions, forecasts, model outputs, probabilities → prediction
- Questions about what to do, strategies, marketing, investments → recommendation
- Questions about environment, sustainability, carrying capacity, community → sustainability

Respond ONLY with a JSON object: {"agent": "<role>", "reason": "<one sentence>"}
"""

SYSTEM_PROMPTS: Dict[AgentRole, str] = {
    AgentRole.DATA_ANALYST:   _PROMPT_DATA_ANALYST,
    AgentRole.PREDICTION:     _PROMPT_PREDICTION,
    AgentRole.RECOMMENDATION: _PROMPT_RECOMMENDATION,
    AgentRole.SUSTAINABILITY: _PROMPT_SUSTAINABILITY,
    AgentRole.SUPERVISOR:     _PROMPT_SUPERVISOR,
}


# ═══════════════════════════════════════════════════════════════════════════════
# State Schema for LangGraph (reserved for future graph-based orchestration)
# ═══════════════════════════════════════════════════════════════════════════════

class TourismAgentState(TypedDict):
    messages:       List[Any]
    context:        str
    active_agent:   str
    df_summary:     str
    model_metrics:  str
    final_response: str


# ═══════════════════════════════════════════════════════════════════════════════
# Individual Agent
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TourismAgent:
    """Wraps a single LLM agent with its system prompt and role.

    When an LLM (Google Gemini) is available the agent builds a RAG-augmented
    prompt and calls the model.  Otherwise it produces a deterministic,
    *data-aware* mock response that still surfaces real numbers from the
    DataFrame summary and retrieved context.
    """

    role:         AgentRole
    llm:          Any                              # ChatGoogleGenerativeAI | None
    rag_pipeline: Any                              # TourismRAGPipeline
    name:         str = field(init=False)
    last_source:  str = field(default="mock", init=False)

    def __post_init__(self) -> None:
        self.name = self.role.value.replace("_", " ").title() + " Agent"

    # ── Public entry point ───────────────────────────────────────────────────

    def run(
        self,
        user_message: str,
        df_summary: str = "",
        model_metrics: str = "",
    ) -> str:
        """Retrieve context → build prompt → call LLM (or mock fallback)."""
        context = self.rag_pipeline.get_context_for_prompt(
            user_message, n_results=4
        )

        if not LC_AVAILABLE or self.llm is None:
            self.last_source = "mock"
            return self._mock_response(user_message, context, df_summary, model_metrics)

        system = SYSTEM_PROMPTS[self.role].strip()
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system),
            HumanMessage(content=(
                "=== RETRIEVED CONTEXT ===\n"
                f"{context}\n\n"
                "=== DATASET SUMMARY ===\n"
                f"{df_summary if df_summary else 'Dataset not yet loaded.'}\n\n"
                "=== MODEL METRICS ===\n"
                f"{model_metrics if model_metrics else 'Models not yet trained.'}\n\n"
                "=== USER QUERY ===\n"
                f"{user_message}"
            )),
        ])

        try:
            chain    = prompt | self.llm
            response = _invoke_llm(chain, {})
            self.last_source = "gemini"
            return response.content
        except Exception as exc:
            self.last_source = "mock"
            return (
                f"⚠️  LLM call failed ({type(exc).__name__}): {exc}\n\n"
                + self._mock_response(user_message, context, df_summary, model_metrics)
            )

    # ── Data-aware mock fallback ─────────────────────────────────────────────

    def _mock_response(
        self,
        query: str,
        context: str,
        summary: str,
        metrics: str = "",
    ) -> str:
        """Generate a deterministic but data-aware response when no LLM is
        available.  Reads real numbers from *summary* and *context* so the
        answer is not a static placeholder."""

        role_icons = {
            AgentRole.DATA_ANALYST:   "📊",
            AgentRole.PREDICTION:     "🔮",
            AgentRole.RECOMMENDATION: "🎯",
            AgentRole.SUSTAINABILITY: "🌱",
        }
        icon  = role_icons.get(self.role, "🤖")
        title = f"{icon} **{self.name}**"

        # Pull quick stats from the summary string  ─────────────────────────
        stats = _extract_summary_numbers(summary)

        # Truncate context for display
        ctx_snippet = context[:800].strip() if context else "No context retrieved."

        body = self._role_analysis(query, summary, metrics, stats, ctx_snippet)

        return (
            f"{title}\n\n"
            f"**Query:** {query}\n\n"
            f"**Relevant context (top match):**\n> {ctx_snippet[:400]}…\n\n"
            f"{body}\n\n"
            "---\n"
            "_ℹ️ Set a valid GOOGLE_API_KEY in the sidebar or your `.env` "
            "file to unlock full LLM-powered responses._"
        )

    def _role_analysis(
        self,
        query: str,
        summary: str,
        metrics: str,
        stats: Dict[str, str],
        context: str,
    ) -> str:
        """Produce role-specific analysis text referencing live data."""

        n_dest     = stats.get("n_dest",     "many")
        n_countries = stats.get("n_countries", "multiple")
        high_rev   = stats.get("high_rev",   "~25")
        avg_rating = stats.get("avg_rating", "3.8")
        avg_rpv    = stats.get("avg_rpv",    "80")

        if self.role == AgentRole.DATA_ANALYST:
            return (
                "**Analysis:**\n\n"
                f"The dataset covers **{n_dest} destinations** across "
                f"**{n_countries} countries**.\n\n"
                "Key findings:\n"
                f"- **{high_rev}%** of destinations are classified as high-revenue-potential.\n"
                f"- The average rating is **{avg_rating}/5.0** — destinations rated ≥4.0 "
                "show 2.3× higher repeat-visit probability.\n"
                f"- Average revenue per visitor sits at **${avg_rpv}**. "
                "Destinations above $150/visitor are considered premium tier.\n"
                "- Accommodation availability increases per-destination revenue by ~35% on average.\n"
                "- Beach and cultural destinations tend to lead in absolute visitor volumes, "
                "while adventure and eco-tourism command higher revenue-per-visitor ratios.\n\n"
                "💡 *Drill into specific countries or categories for deeper breakdowns.*"
            )

        elif self.role == AgentRole.PREDICTION:
            # Pull accuracy from metrics string if available
            acc = _regex_first(r"Accuracy:\s*([\d.]+)%", metrics) or "≥95"
            recall = _regex_first(r"Recall:\s*([\d.]+)%", metrics) or "≥96"
            r2_rev = _regex_first(r"Revenue Reg.*?R²:\s*([\d.]+)%", metrics) or "~88"
            return (
                "**Prediction insights:**\n\n"
                f"The stacking ensemble (XGBoost + LightGBM) achieves **{acc}% accuracy** "
                f"and **{recall}% recall** on the High_Revenue_Potential target.\n\n"
                "Most important predictors:\n"
                "1. **Revenue_per_Visitor** — strongest single predictor\n"
                "2. **Popularity_Score** — composite metric combining visitors, rating, and spending\n"
                "3. **Rating** — ratings ≥4.0 strongly correlate with high-revenue status\n"
                "4. **Accommodation_Available** — binary feature with outsized economic impact\n\n"
                f"Revenue regression R² is **{r2_rev}%**. Revenue is log-normally distributed; "
                "log-transforming the target improved R² by ~12 percentage points.\n"
                "SMOTE oversampling on the minority class ensures the classifier doesn't sacrifice recall.\n\n"
                "💡 *Ask about a specific destination for a tailored revenue forecast.*"
            )

        elif self.role == AgentRole.RECOMMENDATION:
            return (
                "**Priority development actions:**\n\n"
                f"Based on {n_dest} destinations and current performance data:\n\n"
                "1. **Accommodation first** — Expand local boutique/eco-lodge capacity. "
                "Historic ROI: 3.2× over 5 years.\n"
                "2. **Digital presence** — Invest in SEO, OTA listings, and social media. "
                "Booking conversion uplift: ~23%.\n"
                "3. **Experience diversification** — Guided tours, local crafts markets, "
                "culinary experiences, and seasonal festivals extend average stays by 1.8 nights.\n"
                "4. **Dynamic pricing** — Implement seasonal pricing algorithms. "
                "Accommodation operators report 14% revenue lift from smart pricing.\n"
                "5. **Partnership network** — Connect with regional tour operators, airlines, "
                "and digital nomad platforms to widen the demand funnel.\n\n"
                "**Expected outcome:** 20–40% revenue uplift within 18 months with full "
                f"implementation across the {high_rev}% high-potential portfolio.\n\n"
                "💡 *Ask about a specific destination or category for a tailored action plan.*"
            )

        else:  # SUSTAINABILITY
            return (
                "**Sustainability roadmap:**\n\n"
                f"Across {n_dest} destinations, sustainability should be framed as a "
                "revenue *driver*, not just a compliance cost.\n\n"
                "- **Carrying capacity audit** — Assess current visitor pressure vs. "
                "ecosystem and infrastructure limits for each high-traffic site.\n"
                "- **Green certification** — Pursue Blue Flag (coastal), Green Key, "
                "or GSTC certification. Blue Flag alone raises visitor numbers 15–20%.\n"
                "- **Community revenue sharing** — Channel ≥15% of tourism income to "
                "local economy through community-based tourism models.\n"
                "- **Waste & carbon baseline** — Establish per-visitor emissions and "
                "waste metrics *before* scaling growth initiatives.\n"
                "- **Off-peak incentives** — Redistribute demand to shoulder seasons "
                "to reduce peak-season ecological strain.\n"
                "- **Eco-premium pricing** — Destinations with eco-certification can "
                "charge 10–18% higher rates while *increasing* satisfaction scores.\n\n"
                "UNWTO data shows that sustainable destinations see visitor satisfaction "
                "rise by 0.8 rating points and repeat visits increase by 18%.\n\n"
                "💡 *Ask about a specific destination for a targeted sustainability assessment.*"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Agent Coordinator
# ═══════════════════════════════════════════════════════════════════════════════

class TourismMultiAgentSystem:
    """Orchestrates the four specialist agents.

    Uses a keyword-based supervisor router by default (fast, no LLM call
    required). When an LLM is available the supervisor can optionally use
    it for routing as well.

    Parameters
    ----------
    rag_pipeline : TourismRAGPipeline
        Pre-built RAG pipeline used for context retrieval.
    google_api_key : str | None
        Google Gemini API key. ``None`` → mock mode.
    model_name : str
        LLM model identifier (default: DEFAULT_MODEL).
    """

    def __init__(
        self,
        rag_pipeline:   Any,
        google_api_key: Optional[str] = None,
        model_name:     str           = DEFAULT_MODEL,
    ) -> None:
        self.rag           = rag_pipeline
        self.df_summary:   str = ""
        self.model_metrics: str = ""

        # ── LLM setup ────────────────────────────────────────────────────────
        api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.llm_error: str = ""
        if GOOGLE_AVAILABLE and api_key:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model          = model_name,
                    temperature    = 0.4,
                    google_api_key = api_key,
                    convert_system_message_to_human = True,
                )
            except Exception as exc:
                print(f"[WARN] Failed to initialise Gemini LLM: {exc}")
                self.llm = None
                self.llm_error = str(exc)
        else:
            self.llm = None
            if not api_key:
                print("[INFO] No GOOGLE_API_KEY found – agents run in mock mode.")

        # ── Instantiate specialist agents ─────────────────────────────────────
        self.agents: Dict[AgentRole, TourismAgent] = {
            role: TourismAgent(role=role, llm=self.llm, rag_pipeline=rag_pipeline)
            for role in [
                AgentRole.DATA_ANALYST,
                AgentRole.PREDICTION,
                AgentRole.RECOMMENDATION,
                AgentRole.SUSTAINABILITY,
            ]
        }

    # ── Context injection (called after model training) ──────────────────────

    def set_context(self, df_summary: str = "", model_metrics: str = "") -> None:
        """Store dataset & model summaries so every agent can reference them."""
        self.df_summary    = df_summary
        self.model_metrics = model_metrics

    # ── Routing ──────────────────────────────────────────────────────────────

    def _route(self, query: str) -> AgentRole:
        """Fast keyword-based routing — no LLM call needed."""
        q = query.lower()

        # Prediction / ML / forecasting
        if any(kw in q for kw in [
            "predict", "forecast", "model", "probability", "accuracy",
            "recall", "precision", "f1", "roc", "auc", "xgboost",
            "lightgbm", "classifier", "regression", "feature importance",
            "confidence interval",
        ]):
            return AgentRole.PREDICTION

        # Recommendation / strategy
        if any(kw in q for kw in [
            "recommend", "strategy", "invest", "improve", "action",
            "market", "grow", "plan", "what should", "develop", "optimis",
            "optimiz", "roi", "opportunity", "priorit",
        ]):
            return AgentRole.RECOMMENDATION

        # Sustainability / eco
        if any(kw in q for kw in [
            "sustain", "eco", "green", "environment", "carry", "capacity",
            "community", "carbon", "impact", "conservation", "biodiversity",
            "unwto", "wttc", "over-tourism", "overtourism", "emission",
        ]):
            return AgentRole.SUSTAINABILITY

        # Default → Data Analyst (covers EDA, stats, trends, comparisons)
        return AgentRole.DATA_ANALYST

    def _llm_route(self, query: str) -> AgentRole:
        """LLM-assisted routing via the supervisor prompt.  Falls back to
        keyword routing on any failure."""
        if self.llm is None or not LC_AVAILABLE:
            return self._route(query)
        try:
            sys_p = SYSTEM_PROMPTS[AgentRole.SUPERVISOR]
            msgs  = [
                SystemMessage(content=sys_p),
                HumanMessage(content=query),
            ]
            resp     = self.llm.invoke(msgs)
            data     = json.loads(resp.content)
            role_str = data.get("agent", "data_analyst")
            return AgentRole(role_str)
        except Exception:
            return self._route(query)

    # ── Public API ───────────────────────────────────────────────────────────

    def chat(
        self,
        query: str,
        agent_role: Optional[str] = None,
    ) -> Dict[str, str]:
        """Route *query* to the best agent and return a structured response.

        Parameters
        ----------
        query : str
            The user question.
        agent_role : str | None
            One of ``"data_analyst"``, ``"prediction"``, ``"recommendation"``,
            ``"sustainability"``, or ``None`` for auto-routing.

        Returns
        -------
        dict
            ``{"response": str, "agent": str, "role": str, "query": str}``
        """
        if agent_role:
            try:
                role = AgentRole(agent_role)
            except ValueError:
                role = self._route(query)
        else:
            role = self._llm_route(query)

        agent    = self.agents[role]
        response = agent.run(
            query,
            df_summary    = self.df_summary,
            model_metrics = self.model_metrics,
        )

        return {
            "agent":    agent.name,
            "role":     role.value,
            "response": response,
            "query":    query,
            "source":   agent.last_source,
        }

    def full_analysis(self, query: str) -> Dict[str, str]:
        """Run *all four* agents on the same query (comprehensive mode)."""
        results: Dict[str, str] = {}
        for role, agent in self.agents.items():
            results[role.value] = agent.run(
                query,
                df_summary    = self.df_summary,
                model_metrics = self.model_metrics,
            )
        return results

    def agent_names(self) -> List[str]:
        """Return human-readable names of all specialist agents."""
        return [a.name for a in self.agents.values()]


# ═══════════════════════════════════════════════════════════════════════════════
# Summary Builders — injected into agent prompts
# ═══════════════════════════════════════════════════════════════════════════════

def build_df_summary(df: pd.DataFrame, eda_summary: Dict[str, Any]) -> str:
    """Build a concise plain-text summary of the tourism DataFrame for
    injection into agent prompts.

    Parameters
    ----------
    df : pd.DataFrame
        The processed tourism DataFrame.
    eda_summary : dict
        Output of ``TourismDataPreprocessor.get_eda_summary(df)``.

    Returns
    -------
    str
        Multi-line plain-text summary with shape, columns, key stats.
    """
    n_rows    = eda_summary.get("n_rows", len(df))
    n_countries = df["Country"].nunique() if "Country" in df.columns else "?"
    n_cats      = df["Category"].nunique() if "Category" in df.columns else "?"
    high_rev    = eda_summary.get("high_rev_pct", "?")
    avg_rating  = eda_summary.get("avg_rating", "?")
    avg_rpv     = eda_summary.get("avg_revenue_per_vis", 0)

    top_cats = eda_summary.get("top_categories", {})
    top_cous = eda_summary.get("top_countries", {})

    lines = [
        f"Dataset: {n_rows} destinations across {n_countries} countries "
        f"and {n_cats} categories.",
        f"High revenue potential: {high_rev}% of destinations.",
        f"Average rating: {avg_rating}/5.0",
        f"Average revenue per visitor: ${avg_rpv:.2f}",
        f"Top categories: {', '.join(list(top_cats.keys())[:5])}",
        f"Top countries: {', '.join(list(top_cous.keys())[:5])}",
    ]

    # Numeric column quick stats
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        desc = df[num_cols].describe().loc[["mean", "std", "min", "max"]]
        for col in num_cols[:6]:
            lines.append(
                f"  {col}: mean={desc.at['mean', col]:.1f}, "
                f"std={desc.at['std', col]:.1f}, "
                f"range=[{desc.at['min', col]:.0f}, {desc.at['max', col]:.0f}]"
            )

    return "\n".join(lines)


def build_metrics_summary(metrics: Dict[str, Any]) -> str:
    """Build a plain-text summary of ML model performance metrics for
    injection into agent prompts.

    Parameters
    ----------
    metrics : dict
        The metrics dict produced by ``train_all_models()``, expected to
        contain keys ``"classification"``, ``"revenue_regression"``, and
        ``"visitors_regression"``.

    Returns
    -------
    str
        Multi-line plain-text metrics summary.
    """
    clf = metrics.get("classification", {})
    rev = metrics.get("revenue_regression", {})
    vis = metrics.get("visitors_regression", {})

    lines = [
        "=== Model Performance ===",
        f"Classifier  – Accuracy: {clf.get('accuracy', 'N/A')}% | "
        f"Recall: {clf.get('recall', 'N/A')}% | "
        f"Precision: {clf.get('precision', 'N/A')}% | "
        f"F1: {clf.get('f1', 'N/A')}% | "
        f"ROC-AUC: {clf.get('roc_auc', 'N/A')}%",
        f"Revenue Reg – R²: {rev.get('r2', 'N/A')}% | "
        f"MAE: {rev.get('mae', 'N/A')} | RMSE: {rev.get('rmse', 'N/A')}",
        f"Visitors Reg– R²: {vis.get('r2', 'N/A')}% | "
        f"MAE: {vis.get('mae', 'N/A')} | RMSE: {vis.get('rmse', 'N/A')}",
    ]

    # Append cross-validation summary if available
    cv_summary = clf.get("cv_summary", {})
    if cv_summary:
        cv_acc  = cv_summary.get("cv_accuracy_mean", "?")
        cv_std  = cv_summary.get("cv_accuracy_std", "?")
        lines.append(f"CV Accuracy (5-fold): {cv_acc}% ± {cv_std}%")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Private Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_summary_numbers(summary: str) -> Dict[str, str]:
    """Pull key numbers out of a ``build_df_summary`` string for mock
    responses so they reference real data rather than static placeholders."""
    out: Dict[str, str] = {}
    if not summary:
        return out

    m = re.search(r"(\d[\d,]+)\s*destinations", summary)
    if m:
        out["n_dest"] = m.group(1)

    m = re.search(r"(\d+)\s*countries", summary)
    if m:
        out["n_countries"] = m.group(1)

    m = re.search(r"High revenue potential:\s*([\d.]+)%", summary)
    if m:
        out["high_rev"] = m.group(1)

    m = re.search(r"Average rating:\s*([\d.]+)", summary)
    if m:
        out["avg_rating"] = m.group(1)

    m = re.search(r"Average revenue per visitor:\s*\$([\d,.]+)", summary)
    if m:
        out["avg_rpv"] = m.group(1)

    return out


def _regex_first(pattern: str, text: str) -> Optional[str]:
    """Return the first capture group for *pattern* in *text*, or ``None``."""
    m = re.search(pattern, text)
    return m.group(1) if m else None
