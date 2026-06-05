"""
data_preprocessing.py
=====================
Handles all data loading, cleaning, EDA, and feature engineering
for the Tourism RAG + Predictive Analytics System.

Author : Tourism-AI Team
Version: 1.0.0
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

# ─── Constants ───────────────────────────────────────────────────────────────

REQUIRED_COLUMNS: List[str] = [
    "Location", "Country", "Category",
    "Visitors", "Rating", "Revenue", "Accommodation_Available",
]

HIGH_REVENUE_PERCENTILE = 0.75          # top-25 % flagged as high-potential
RANDOM_STATE            = 42


# ─── Main Preprocessing Class ────────────────────────────────────────────────

class TourismDataPreprocessor:
    """
    End-to-end preprocessing pipeline.

    Usage
    -----
    >>> prep = TourismDataPreprocessor()
    >>> df   = prep.load_data("data/tourism_dataset.csv")
    >>> df   = prep.preprocess(df)
    >>> X_train, X_test, y_train, y_test = prep.split(df)
    """

    def __init__(self) -> None:
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler   = StandardScaler()
        self.minmax   = MinMaxScaler()
        self.stats_   : Optional[pd.DataFrame] = None
        self._revenue_threshold: Optional[float] = None
        self._potential_threshold: Optional[float] = None

    # ─── Loading ─────────────────────────────────────────────────────────────

    def load_data(self, path: str | Path) -> pd.DataFrame:
        """Load CSV; validate required columns; return raw DataFrame."""
        df = pd.read_csv(path)
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing columns: {missing}")
        df.columns = df.columns.str.strip()
        return df

    # ─── Cleaning ────────────────────────────────────────────────────────────

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicates, impute missing values, fix dtypes."""
        df = df.copy()
        df.drop_duplicates(inplace=True)

        # --- numeric coercion ------------------------------------------------
        for col in ["Visitors", "Rating", "Revenue"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # --- impute numeric with median (robust to outliers) -----------------
        for col in ["Visitors", "Rating", "Revenue"]:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

        # --- impute categorical with mode ------------------------------------
        for col in ["Location", "Country", "Category", "Accommodation_Available"]:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)

        # --- clamp Rating to [0, 5] ------------------------------------------
        df["Rating"] = df["Rating"].clip(0, 5)

        # --- standardise boolean / yes-no column -----------------------------
        df["Accommodation_Available"] = (
            df["Accommodation_Available"]
            .astype(str).str.strip().str.lower()
            .map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
            .fillna(0).astype(int)
        )

        return df

    # ─── Feature Engineering ─────────────────────────────────────────────────

    def compute_popularity_score(self, df: pd.DataFrame, fit: bool = True) -> pd.Series:
        """Compute Popularity_Score with scaler fit controlled by caller."""
        cols = ["Visitors", "Rating"]
        scaled = (
            self.minmax.fit_transform(df[cols])
            if fit
            else self.minmax.transform(df[cols])
        )
        score = (0.6 * scaled[:, 0] + 0.4 * scaled[:, 1]) * 100
        return pd.Series(score, index=df.index, name="Popularity_Score")

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all derived features used by ML models."""
        df = df.copy()

        # ── Ratios & composites ──────────────────────────────────────────────
        df["Revenue_per_Visitor"] = np.where(
            df["Visitors"] > 0, df["Revenue"] / df["Visitors"], 0
        )
        df["Visitor_Density_Score"] = (
            df["Visitors"] * df["Rating"] / (df["Revenue"] + 1e-9)
        )
        df["Rating_Revenue_Index"] = df["Rating"] * df["Revenue_per_Visitor"]
        df["Accommodation_Revenue_Boost"] = (
            df["Accommodation_Available"] * df["Revenue_per_Visitor"]
        )

        # ── NEW: Interaction & polynomial features ───────────────────────────
        df["Visitors_x_Rating"] = df["Visitors"] * df["Rating"]
        df["Revenue_x_Acc"] = df["Revenue"] * df["Accommodation_Available"]
        df["Rating_Squared"] = df["Rating"] ** 2
        df["Log_RevPerVis"] = np.log1p(df["Revenue_per_Visitor"])
        df["Acc_x_Rating"] = df["Accommodation_Available"] * df["Rating"]

        # ── Log-transform skewed columns ─────────────────────────────────────
        df["Log_Visitors"] = np.log1p(df["Visitors"])
        df["Log_Revenue"]  = np.log1p(df["Revenue"])

        # ── Binned features ──────────────────────────────────────────────────
        df["Rating_Band"] = pd.cut(
            df["Rating"],
            bins=[0, 2, 3, 4, 5],
            labels=["Low", "Medium", "High", "Premium"],
        )
        df["Visitor_Tier"] = pd.qcut(
            df["Visitors"].rank(method="first"),
            q=4,
            labels=["Bronze", "Silver", "Gold", "Platinum"],
        )

        # ── Target: High_Revenue_Potential (binary classification) ──────────
        # A destination's potential is a business score, not just a direct copy
        # of Revenue. This keeps the classifier useful while still grounding the
        # label in measurable tourism signals.
        def _safe_minmax(col: str) -> pd.Series:
            values = df[col].astype(float)
            spread = values.max() - values.min()
            if spread == 0:
                return pd.Series(0.5, index=df.index)
            return (values - values.min()) / spread

        df["Revenue_Potential_Score"] = (
            0.45 * _safe_minmax("Revenue")
            + 0.30 * _safe_minmax("Visitors")
            + 0.20 * _safe_minmax("Rating")
            + 0.05 * _safe_minmax("Accommodation_Available")
        ) * 100

        threshold = df["Revenue_Potential_Score"].quantile(HIGH_REVENUE_PERCENTILE)
        self._potential_threshold = threshold
        self._revenue_threshold = df["Revenue"].quantile(HIGH_REVENUE_PERCENTILE)
        df["High_Revenue_Potential"] = (
            df["Revenue_Potential_Score"] >= threshold
        ).astype(int)

        print(f"[TARGET] High_Revenue_Potential distribution:")
        print(df["High_Revenue_Potential"].value_counts(normalize=True).round(3))

        df["Popularity_Score"] = self.compute_popularity_score(df, fit=True)

        return df

    # ─── Encoding ────────────────────────────────────────────────────────────

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label-encode categorical features for ML."""
        df = df.copy()
        cat_cols = ["Category", "Rating_Band", "Visitor_Tier", "Country"]

        # Location has high cardinality → frequency encode
        loc_freq = df["Location"].value_counts(normalize=True)
        df["Location_Freq"] = df["Location"].map(loc_freq)

        for col in cat_cols:
            le = LabelEncoder()
            df[f"{col}_Enc"] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        return df

    # ─── Full Pipeline ────────────────────────────────────────────────────────

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full preprocessing pipeline."""
        df = self.clean(df)
        df = self.engineer_features(df)
        df = self.encode_categoricals(df)
        self.stats_ = self._compute_stats(df)
        return df

    # ─── Feature Sets ─────────────────────────────────────────────────────────

    @staticmethod
    def get_classification_features() -> List[str]:
        """Feature list for High_Revenue_Potential classifier."""
        return [
            "Revenue", "Log_Revenue",
            "Visitors", "Rating", "Accommodation_Available",
            "Log_Visitors", "Popularity_Score",
            "Category_Enc", "Country_Enc", "Rating_Band_Enc",
            "Visitor_Tier_Enc", "Location_Freq",
            "Visitors_x_Rating", "Rating_Squared", "Acc_x_Rating",
            "Country_Vis_Mean",
        ]

    @staticmethod
    def get_revenue_regression_features() -> List[str]:
        """Feature list for Revenue regressor."""
        return [
            "Visitors", "Rating", "Accommodation_Available",
            "Log_Visitors", "Category_Enc", "Country_Enc",
            "Rating_Band_Enc", "Visitor_Tier_Enc", "Location_Freq",
            "Popularity_Score",
            # v2 features
            "Visitors_x_Rating", "Rating_Squared", "Revenue_x_Acc",
            "Acc_x_Rating", "Country_Vis_Mean", "Category_Rev_Density",
        ]

    @staticmethod
    def get_visitors_regression_features() -> List[str]:
        """Feature list for Visitors regressor."""
        return [
            "Revenue", "Rating", "Accommodation_Available",
            "Log_Revenue", "Category_Enc", "Country_Enc",
            "Rating_Band_Enc", "Location_Freq", "Revenue_per_Visitor",
            # v2 features
            "Log_RevPerVis", "Revenue_x_Acc", "Acc_x_Rating",
            "Country_Rev_Mean", "Category_Rev_Density",
        ]

    # ─── Train / Test Split ──────────────────────────────────────────────────

    def split(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        target: str = "High_Revenue_Potential",
        test_size: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Stratified split (stratify on target for classification)."""
        from sklearn.model_selection import train_test_split

        if features is None:
            features = self.get_classification_features()

        y = df[target]
        stratify = y if y.nunique() <= 10 else None

        train_idx, test_idx = train_test_split(
            df.index,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )

        dynamic_features = {
            "Popularity_Score", "Country_Rev_Mean",
            "Country_Vis_Mean", "Category_Rev_Density",
        }
        base_features = [f for f in features if f not in dynamic_features]

        X_train = df.loc[train_idx, base_features].copy()
        X_test = df.loc[test_idx, base_features].copy()
        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]

        if "Popularity_Score" in features:
            X_train["Popularity_Score"] = self.compute_popularity_score(train_df, fit=True)
            X_test["Popularity_Score"] = self.compute_popularity_score(test_df, fit=False)

        if "Country_Rev_Mean" in features:
            country_rev_mean = train_df.groupby("Country")["Revenue"].mean()
            fallback = country_rev_mean.mean()
            X_train["Country_Rev_Mean"] = train_df["Country"].map(country_rev_mean).fillna(fallback)
            X_test["Country_Rev_Mean"] = test_df["Country"].map(country_rev_mean).fillna(fallback)

        if "Country_Vis_Mean" in features:
            country_vis_mean = train_df.groupby("Country")["Visitors"].mean()
            fallback = country_vis_mean.mean()
            X_train["Country_Vis_Mean"] = train_df["Country"].map(country_vis_mean).fillna(fallback)
            X_test["Country_Vis_Mean"] = test_df["Country"].map(country_vis_mean).fillna(fallback)

        if "Category_Rev_Density" in features:
            category_rev_density = train_df.groupby("Category")["Revenue"].mean()
            fallback = category_rev_density.mean()
            X_train["Category_Rev_Density"] = train_df["Category"].map(category_rev_density).fillna(fallback)
            X_test["Category_Rev_Density"] = test_df["Category"].map(category_rev_density).fillna(fallback)

        X_train = X_train[features]
        X_test = X_test[features]
        y_train = y.loc[train_idx]
        y_test = y.loc[test_idx]

        return X_train, X_test, y_train, y_test

    # ─── Statistics / EDA helpers ────────────────────────────────────────────

    def _compute_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        return df[num_cols].describe().T

    def get_eda_summary(self, df: pd.DataFrame) -> Dict:
        """Return a dict of EDA metrics for display in the dashboard."""
        return {
            "n_rows"              : len(df),
            "n_cols"              : df.shape[1],
            "missing_pct"         : df.isnull().mean().mul(100).round(2).to_dict(),
            "dtypes"              : df.dtypes.astype(str).to_dict(),
            "numeric_stats"       : df.describe().round(2).to_dict(),
            "high_rev_pct"        : round(df["High_Revenue_Potential"].mean() * 100, 1),
            "top_categories"      : df["Category"].value_counts().head(10).to_dict(),
            "top_countries"       : df["Country"].value_counts().head(10).to_dict(),
            "revenue_threshold"   : self._revenue_threshold,
            "potential_threshold" : self._potential_threshold,
            "avg_rating"          : round(df["Rating"].mean(), 2),
            "avg_revenue_per_vis" : round(df["Revenue_per_Visitor"].mean(), 2),
        }


# ─── Synthetic Dataset Generator (fallback when no CSV is uploaded) ──────────

def generate_synthetic_dataset(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic tourism dataset so the app can be
    demonstrated without a real upload.
    """
    rng = np.random.default_rng(seed)

    countries     = ["France", "Italy", "Spain", "Japan", "USA",
                     "Thailand", "India", "Germany", "UK", "Australia",
                     "Brazil", "Mexico", "Canada", "UAE", "Singapore"]
    categories    = ["Beach", "Mountain", "Cultural", "Adventure",
                     "Wildlife", "City", "Heritage", "Eco-Tourism",
                     "Religious", "Sports"]

    country_arr   = rng.choice(countries, n)
    category_arr  = rng.choice(categories, n)
    acc_arr       = rng.choice([0, 1], n, p=[0.25, 0.75])

    # Visitors: log-normal with category modifier
    cat_multiplier = {c: 1 + i * 0.15 for i, c in enumerate(categories)}
    base_visitors  = rng.lognormal(mean=9, sigma=1.2, size=n).astype(int)
    visitors       = np.clip(
        (base_visitors * np.array([cat_multiplier[c] for c in category_arr])).astype(int),
        500, 5_000_000
    )

    # Rating: beta-distributed, skewed toward 3–5
    rating = np.round(rng.beta(5, 2, n) * 5, 1).clip(0, 5)

    # Revenue: correlated with visitors + rating + accommodation
    revenue = (
        visitors * rng.uniform(10, 150, n)
        * (rating / 3)
        * (1 + 0.3 * acc_arr)
        + rng.normal(0, 50_000, n)
    ).clip(1_000).astype(int)

    location_list = [
        f"{cat} Site {i+1}" for i, cat in
        enumerate(category_arr)
    ]

    df = pd.DataFrame({
        "Location"               : location_list,
        "Country"                : country_arr,
        "Category"               : category_arr,
        "Visitors"               : visitors,
        "Rating"                 : rating,
        "Revenue"                : revenue,
        "Accommodation_Available": np.where(acc_arr == 1, "Yes", "No"),
    })

    return df


# ─── CLI helper ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    prep = TourismDataPreprocessor()
    if path:
        raw = prep.load_data(path)
    else:
        print("No path provided – using synthetic dataset.")
        raw = generate_synthetic_dataset()

    processed = prep.preprocess(raw)
    summary   = prep.get_eda_summary(processed)
    print("\n=== EDA Summary ===")
    for k, v in summary.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")
    print(f"\nProcessed shape: {processed.shape}")
    print(f"Features ready. High Revenue destinations: {summary['high_rev_pct']}%")
