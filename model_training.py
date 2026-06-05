"""
model_training.py
=================
Trains high-accuracy classification and regression models for the
Tourism Predictive Analytics System.

Targets:
  Classification : High_Revenue_Potential  (≥95 % acc, ≥96 % recall)
  Regression     : Revenue, Visitors

v2 improvements:
  - StackingClassifier / StackingRegressor (replaces simple Voting)
  - Optuna hyperparameter tuning (20 trials, Bayesian search)
  - Stratified 5-fold cross-validation with per-fold metrics
  - CalibratedClassifierCV for reliable probability estimates
  - Early stopping for XGBoost / LightGBM
  - ElasticNet diversity in regression stacking
  - Model comparison metrics (individual vs ensemble)

Author : Tourism-AI Team
Version: 2.0.0
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─── Optional heavy deps (graceful import) ───────────────────────────────────
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[WARN] xgboost not found - falling back to LightGBM / RF.")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("[WARN] lightgbm not found - falling back to RF.")

try:
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("[WARN] imbalanced-learn not found - class_weight='balanced' will be used.")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[INFO] optuna not found - using static hyperparameters.")

RANDOM_STATE = 42
MODEL_DIR    = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

N_OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", "20"))  # Bayesian HP search budget
N_CV_FOLDS      = 3           # Stratified K-Fold count
N_JOBS          = int(os.getenv("MODEL_N_JOBS", "-1"))


# ═══════════════════════════════════════════════════════════════════════════════
# Optuna Hyperparameter Tuning
# ═══════════════════════════════════════════════════════════════════════════════

def _optuna_xgb_params(trial) -> Dict:
    """Suggest XGBoost hyperparameters for an Optuna trial."""
    return {
        "n_estimators"     : trial.suggest_int("xgb_n_estimators", 150, 400, step=50),
        "max_depth"        : trial.suggest_int("xgb_max_depth", 4, 10),
        "learning_rate"    : trial.suggest_float("xgb_lr", 0.01, 0.15, log=True),
        "subsample"        : trial.suggest_float("xgb_subsample", 0.7, 0.95),
        "colsample_bytree" : trial.suggest_float("xgb_colsample", 0.6, 0.95),
        "min_child_weight" : trial.suggest_int("xgb_mcw", 1, 10),
        "gamma"            : trial.suggest_float("xgb_gamma", 0.0, 0.5),
        "reg_alpha"        : trial.suggest_float("xgb_alpha", 0.0, 1.0),
        "reg_lambda"       : trial.suggest_float("xgb_lambda", 0.5, 3.0),
    }


def _optuna_lgb_params(trial) -> Dict:
    """Suggest LightGBM hyperparameters for an Optuna trial."""
    return {
        "n_estimators"     : trial.suggest_int("lgb_n_estimators", 150, 400, step=50),
        "max_depth"        : trial.suggest_int("lgb_max_depth", 4, 10),
        "learning_rate"    : trial.suggest_float("lgb_lr", 0.01, 0.15, log=True),
        "num_leaves"       : trial.suggest_int("lgb_num_leaves", 31, 127),
        "subsample"        : trial.suggest_float("lgb_subsample", 0.7, 0.95),
        "colsample_bytree" : trial.suggest_float("lgb_colsample", 0.6, 0.95),
        "min_child_samples": trial.suggest_int("lgb_mcs", 5, 50),
        "reg_alpha"        : trial.suggest_float("lgb_alpha", 0.0, 1.0),
        "reg_lambda"       : trial.suggest_float("lgb_lambda", 0.5, 3.0),
    }


def _tune_classifier_hp(X_train, y_train) -> Dict:
    """
    Run Optuna Bayesian optimisation to find best XGB + LGB hyperparameters
    for the classification task.  Returns dict with 'xgb' and 'lgb' keys.
    """
    if not OPTUNA_AVAILABLE:
        return {}

    best_params = {}

    def objective(trial):
        params = {}
        if XGB_AVAILABLE:
            xgb_p = _optuna_xgb_params(trial)
            params["xgb"] = xgb_p
        if LGB_AVAILABLE:
            lgb_p = _optuna_lgb_params(trial)
            params["lgb"] = lgb_p

        # Build a quick stacking ensemble and score with 2-fold CV
        estimators = []
        if XGB_AVAILABLE:
            estimators.append(("xgb", xgb.XGBClassifier(
                **xgb_p, eval_metric="logloss", use_label_encoder=False,
                class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=N_JOBS,
            )))
        if LGB_AVAILABLE:
            estimators.append(("lgb", lgb.LGBMClassifier(
                **lgb_p, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=-1,
            )))
        estimators.append(("rf", RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=N_JOBS,
        )))

        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            cv=3, n_jobs=N_JOBS,
        )
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(stack, X_train, y_train, cv=skf, scoring="f1", n_jobs=N_JOBS)
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    print(f"[Optuna] Best f1: {study.best_value:.4f} after {len(study.trials)} trials")

    best_trial = study.best_trial
    # Extract per-model params from best trial
    if XGB_AVAILABLE:
        best_params["xgb"] = {
            "n_estimators"     : best_trial.params.get("xgb_n_estimators", 300),
            "max_depth"        : best_trial.params.get("xgb_max_depth", 7),
            "learning_rate"    : best_trial.params.get("xgb_lr", 0.05),
            "subsample"        : best_trial.params.get("xgb_subsample", 0.85),
            "colsample_bytree" : best_trial.params.get("xgb_colsample", 0.80),
            "min_child_weight" : best_trial.params.get("xgb_mcw", 3),
            "gamma"            : best_trial.params.get("xgb_gamma", 0.1),
            "reg_alpha"        : best_trial.params.get("xgb_alpha", 0.1),
            "reg_lambda"       : best_trial.params.get("xgb_lambda", 1.5),
        }
    if LGB_AVAILABLE:
        best_params["lgb"] = {
            "n_estimators"     : best_trial.params.get("lgb_n_estimators", 300),
            "max_depth"        : best_trial.params.get("lgb_max_depth", 7),
            "learning_rate"    : best_trial.params.get("lgb_lr", 0.05),
            "num_leaves"       : best_trial.params.get("lgb_num_leaves", 63),
            "subsample"        : best_trial.params.get("lgb_subsample", 0.85),
            "colsample_bytree" : best_trial.params.get("lgb_colsample", 0.80),
            "min_child_samples": best_trial.params.get("lgb_mcs", 20),
            "reg_alpha"        : best_trial.params.get("lgb_alpha", 0.1),
            "reg_lambda"       : best_trial.params.get("lgb_lambda", 1.5),
        }

    print(f"[Optuna] Best f1 (3-fold CV): {study.best_value:.4f} "
          f"after {len(study.trials)} trials")
    return best_params


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Classification
# ═══════════════════════════════════════════════════════════════════════════════

class HighRevenuePotentialClassifier:
    """
    Stacking ensemble (XGBoost + LightGBM + RF → LogisticRegression meta)
    classifier for predicting High_Revenue_Potential.

    v2 improvements over v1 VotingClassifier:
    - Stacking learns optimal model weights from data (vs equal voting)
    - Optuna tunes XGB/LGB hyperparameters via Bayesian search
    - CalibratedClassifierCV for reliable probability estimates
    - 5-fold stratified CV metrics reported alongside held-out test metrics
    - SMOTE handles class imbalance
    """

    def __init__(self) -> None:
        self.model     : Optional[Any] = None
        self.metrics_  : Dict          = {}
        self.feature_importances_: Optional[pd.Series] = None
        self._cv_scores: Dict[str, List[float]] = {}
        self._model_comparison: Dict[str, Dict] = {}
        self.decision_threshold_: float = 0.5

    # ── Build base estimators ────────────────────────────────────────────────

    def _build_xgb(self, hp: Optional[Dict] = None, early_stopping: bool = False) -> Any:
        params = hp or {
            "n_estimators": 200, "max_depth": 7, "learning_rate": 0.05,
            "subsample": 0.85, "colsample_bytree": 0.80,
            "min_child_weight": 3, "gamma": 0.1,
            "reg_alpha": 0.1, "reg_lambda": 1.5,
        }
        if early_stopping:
            params = {**params, "early_stopping_rounds": 30}
        return xgb.XGBClassifier(
            **params,
            class_weight      = "balanced",
            scale_pos_weight  = 1,
            eval_metric       = "logloss",
            use_label_encoder = False,
            random_state      = RANDOM_STATE,
            n_jobs            = N_JOBS,
        )

    def _build_lgb(self, hp: Optional[Dict] = None, early_stopping: bool = False) -> Any:
        params = hp or {
            "n_estimators": 200, "max_depth": 7, "learning_rate": 0.05,
            "num_leaves": 63, "subsample": 0.85, "colsample_bytree": 0.80,
            "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 1.5,
        }
        if early_stopping:
            params = {**params, "early_stopping_rounds": 30}
        return lgb.LGBMClassifier(
            **params,
            class_weight       = "balanced",
            random_state       = RANDOM_STATE,
            n_jobs             = N_JOBS,
            verbose            = -1,
        )

    def _build_rf(self) -> Any:
        return RandomForestClassifier(
            n_estimators  = 150,
            max_depth     = 12,
            min_samples_split = 5,
            class_weight  = "balanced",
            random_state  = RANDOM_STATE,
            n_jobs        = N_JOBS,
        )

    def _build_stacking(self, xgb_hp=None, lgb_hp=None) -> Any:
        """Build a StackingClassifier with LogisticRegression meta-learner."""
        estimators = []
        if XGB_AVAILABLE:
            estimators.append(("xgb", self._build_xgb(xgb_hp)))
        if LGB_AVAILABLE:
            estimators.append(("lgb", self._build_lgb(lgb_hp)))
        estimators.append(("rf", self._build_rf()))

        return StackingClassifier(
            estimators      = estimators,
            final_estimator = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            cv              = 5,
            stack_method    = "predict_proba",
            n_jobs          = N_JOBS,
            passthrough     = False,
        )

    # ── Cross-Validation ─────────────────────────────────────────────────────

    def _run_cv(self, X, y) -> Dict[str, List[float]]:
        """Run stratified K-fold CV and return per-fold scores."""
        skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scoring = ["accuracy", "recall", "precision", "f1", "roc_auc"]
        cv_results = cross_validate(
            self._build_stacking(),
            X, y, cv=skf, scoring=scoring,
            n_jobs=N_JOBS, return_train_score=False,
        )
        fold_scores = {}
        for metric in scoring:
            fold_scores[metric] = (cv_results[f"test_{metric}"] * 100).tolist()
        return fold_scores

    # ── Individual model comparison ──────────────────────────────────────────

    def _compare_models(self, X_test, y_test, X_train, y_train) -> Dict[str, Dict]:
        """Train and evaluate each base estimator individually for comparison."""
        comparison = {}
        models = []
        if XGB_AVAILABLE:
            models.append(("XGBoost", self._build_xgb(early_stopping=True)))
        if LGB_AVAILABLE:
            models.append(("LightGBM", self._build_lgb(early_stopping=True)))
        models.append(("RandomForest", self._build_rf()))

        for name, model in models:
            try:
                if name in ("XGBoost", "LightGBM"):
                    # Early stopping needs a validation set, so use it for the
                    # standalone model comparison fits. The stack/CV builders
                    # intentionally omit it because sklearn calls fit without
                    # estimator-specific eval_set arguments.
                    from sklearn.model_selection import train_test_split
                    X_tr, X_val, y_tr, y_val = train_test_split(
                        X_train, y_train, test_size=0.15, random_state=RANDOM_STATE,
                        stratify=y_train,
                    )
                    if name == "XGBoost":
                        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                    else:
                        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
                else:
                    model.fit(X_train, y_train)

                y_proba = model.predict_proba(X_test)[:, 1]
                y_pred = (y_proba >= 0.4).astype(int)
                comparison[name] = {
                    "accuracy"  : round(accuracy_score(y_test, y_pred)  * 100, 2),
                    "recall"    : round(recall_score(y_test, y_pred)     * 100, 2),
                    "precision" : round(precision_score(y_test, y_pred)  * 100, 2),
                    "f1"        : round(f1_score(y_test, y_pred)         * 100, 2),
                    "roc_auc"   : round(roc_auc_score(y_test, y_proba)   * 100, 2),
                }
            except Exception as e:
                print(f"[WARN] {name} comparison failed: {e}")
                comparison[name] = {"accuracy": 0, "recall": 0, "precision": 0, "f1": 0, "roc_auc": 0}

        return comparison

    def _best_probability_threshold(
        self,
        y_true: pd.Series,
        y_proba: np.ndarray,
    ) -> float:
        """Pick the held-out probability threshold with the strongest F1 score."""
        best_threshold = 0.5
        best_score = -1.0
        for threshold in np.linspace(0.05, 0.95, 181):
            y_pred = (y_proba >= threshold).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        return best_threshold

    def _predict_with_threshold(
        self,
        X: pd.DataFrame,
        y_proba: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply the fitted probability cutoff."""
        if y_proba is None:
            y_proba = self.model.predict_proba(X)[:, 1]
        return (y_proba >= self.decision_threshold_).astype(int)

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test : pd.DataFrame,
        y_test : pd.Series,
    ) -> Dict:
        """
        Train with SMOTE → Optuna tuning → Stacking ensemble → Calibration.
        Returns evaluation metrics dict.
        """
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )

        # --- handle class imbalance ---
        from collections import Counter
        if IMBLEARN_AVAILABLE:
            raw_train_size = len(y_fit)
            print(f"[SMOTE] Class distribution before: {Counter(y_fit)}")
            smt = SMOTETomek(random_state=RANDOM_STATE)
            X_fit, y_fit = smt.fit_resample(X_fit, y_fit)
            print(f"[SMOTE] Class distribution after:  {Counter(y_fit)}")
            print(f"[SMOTE] {raw_train_size} -> {len(y_fit)} samples "
                  f"(class balance: {np.bincount(y_fit)})")
        else:
            print("[INFO] SMOTE unavailable; using raw data with balanced class_weight.")

        # --- Optuna hyperparameter tuning ---
        tuned_hp = _tune_classifier_hp(X_fit, y_fit)
        xgb_hp = tuned_hp.get("xgb")
        lgb_hp = tuned_hp.get("lgb")

        # --- build stacking ensemble ---
        stacker = self._build_stacking(xgb_hp=xgb_hp, lgb_hp=lgb_hp)
        stacker.fit(X_fit, y_fit)

        # --- calibrate probabilities ---
        try:
            self.model = CalibratedClassifierCV(
                stacker, cv=3, method="isotonic",
            )
            self.model.fit(X_fit, y_fit)
        except Exception:
            # Fallback to uncalibrated if calibration fails
            self.model = stacker
            print("[INFO] Calibration failed; using uncalibrated stacker.")

        # --- tune threshold on validation data only ---
        val_proba = self.model.predict_proba(X_val)[:, 1]
        self.decision_threshold_ = self._best_probability_threshold(y_val, val_proba)

        # --- predict on held-out test set ---
        y_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self._predict_with_threshold(X_test, y_proba)

        # --- cross-validation scores ---
        print(f"[CV] Running {N_CV_FOLDS}-fold stratified cross-validation...")
        self._cv_scores = self._run_cv(X_fit, y_fit)
        cv_summary = {
            f"cv_{k}_mean": round(np.mean(v), 2)
            for k, v in self._cv_scores.items()
        }
        cv_summary.update({
            f"cv_{k}_std": round(np.std(v), 2)
            for k, v in self._cv_scores.items()
        })

        # --- individual model comparison (skipped by default for speed) ---
        run_comparison = os.getenv("RUN_MODEL_COMPARISON", "0") == "1"
        if run_comparison:
            print("[Compare] Evaluating individual base models...")
            self._model_comparison = self._compare_models(X_test, y_test, X_fit, y_fit)
        else:
            print("[Compare] Skipping individual model comparison (set RUN_MODEL_COMPARISON=1 to enable)")
            self._model_comparison = {}

        # --- metrics ---
        self.metrics_ = {
            "accuracy"  : round(accuracy_score(y_test, y_pred)  * 100, 2),
            "recall"    : round(recall_score(y_test, y_pred, zero_division=0)     * 100, 2),
            "precision" : round(precision_score(y_test, y_pred, zero_division=0)  * 100, 2),
            "f1"        : round(f1_score(y_test, y_pred, zero_division=0)         * 100, 2),
            "roc_auc"   : round(roc_auc_score(y_test, y_proba)   * 100, 2),
            "conf_matrix"        : confusion_matrix(y_test, y_pred).tolist(),
            "classification_rep" : classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
            # v2 additions
            "cv_scores"          : self._cv_scores,
            "cv_summary"         : cv_summary,
            "model_comparison"   : self._model_comparison,
            "optuna_tuned"       : bool(tuned_hp),
            "decision_threshold" : round(self.decision_threshold_, 4),
        }

        # --- feature importances (from stacking base estimators) ---
        try:
            base_model = stacker  # use the unwrapped stacker
            if hasattr(base_model, "estimators_"):
                for est in base_model.estimators_:
                    if hasattr(est, "feature_importances_"):
                        fi = pd.Series(
                            est.feature_importances_,
                            index=X_fit.columns,
                        ).sort_values(ascending=False)
                        self.feature_importances_ = fi
                        break
        except Exception:
            pass

        print(f"\n[Classification Results - Stacking v2]")
        print(f"  Accuracy  : {self.metrics_['accuracy']}%")
        print(f"  Recall    : {self.metrics_['recall']}%")
        print(f"  Precision : {self.metrics_['precision']}%")
        print(f"  F1 Score  : {self.metrics_['f1']}%")
        print(f"  ROC-AUC   : {self.metrics_['roc_auc']}%")
        print(f"  Decision threshold: {self.decision_threshold_:.3f}")
        print(f"  CV Accuracy: {cv_summary.get('cv_accuracy_mean', '?')}% "
              f"+/- {cv_summary.get('cv_accuracy_std', '?')}%")
        print(f"  Optuna tuned: {bool(tuned_hp)}")

        return self.metrics_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict_with_threshold(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str = "models/classifier.joblib") -> None:
        state = {
            "model": self.model,
            "metrics_": self.metrics_,
            "feature_importances_": self.feature_importances_,
            "_cv_scores": self._cv_scores,
            "_model_comparison": self._model_comparison,
            "decision_threshold_": self.decision_threshold_,
        }
        joblib.dump(state, path)
        print(f"[Saved] Classifier -> {path}")

    @classmethod
    def load(cls, path: str = "models/classifier.joblib") -> "HighRevenuePotentialClassifier":
        state = joblib.load(path)
        if isinstance(state, dict):
            obj = cls()
            obj.model = state.get("model")
            obj.metrics_ = state.get("metrics_", {})
            obj.feature_importances_ = state.get("feature_importances_")
            obj._cv_scores = state.get("_cv_scores", {})
            obj._model_comparison = state.get("_model_comparison", {})
            obj.decision_threshold_ = state.get("decision_threshold_", 0.5)
            return obj
        return state


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Revenue Regression
# ═══════════════════════════════════════════════════════════════════════════════

class RevenueRegressor:
    """
    Stacking ensemble for predicting tourism Revenue (USD).
    Uses log-target transform to handle right-skewed distribution.

    v2: StackingRegressor with Ridge meta-learner, ElasticNet diversity,
    early stopping, and cross-validation metrics.
    """

    def __init__(self) -> None:
        self.model   : Optional[Any] = None
        self.metrics_: Dict          = {}
        self._cv_scores: Dict        = {}

    def _build_xgb(self, early_stopping: bool = False):
        extra = {"early_stopping_rounds": 30} if early_stopping else {}
        return xgb.XGBRegressor(
            n_estimators   = 250,
            max_depth      = 6,
            learning_rate  = 0.05,
            subsample      = 0.85,
            colsample_bytree = 0.80,
            reg_alpha      = 0.1,
            reg_lambda     = 1.5,
            **extra,
            random_state   = RANDOM_STATE,
            n_jobs         = N_JOBS,
        )

    def _build_lgb(self, early_stopping: bool = False):
        extra = {"early_stopping_rounds": 30} if early_stopping else {}
        return lgb.LGBMRegressor(
            n_estimators   = 250,
            max_depth      = 6,
            learning_rate  = 0.05,
            num_leaves     = 63,
            subsample      = 0.85,
            colsample_bytree = 0.80,
            **extra,
            random_state   = RANDOM_STATE,
            n_jobs         = N_JOBS,
            verbose        = -1,
        )

    def train(self, X_train, y_train, X_test, y_test) -> Dict:
        # log1p target transform for stable training
        y_tr_log = np.log1p(y_train)
        y_te_log = np.log1p(y_test)

        estimators = []
        if XGB_AVAILABLE:
            estimators.append(("xgb", self._build_xgb()))
        if LGB_AVAILABLE:
            estimators.append(("lgb", self._build_lgb()))
        # v2 fix: use RandomForestRegressor (was RandomForestClassifier — BUG)
        estimators.append(("rf", RandomForestRegressor(
            n_estimators=200, max_depth=10,
            random_state=RANDOM_STATE, n_jobs=N_JOBS,
        )))
        # v2: add ElasticNet for linear diversity in the stack
        estimators.append(("enet", ElasticNet(
            alpha=0.1, l1_ratio=0.5, max_iter=2000,
            random_state=RANDOM_STATE,
        )))

        # v2: cross-validation on log-target
        from sklearn.model_selection import cross_val_score
        cv_r2 = cross_val_score(
            StackingRegressor(
                estimators=[(n, clone(e)) for n, e in estimators],
                final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=N_JOBS,
            ),
            X_train, y_tr_log, cv=5, scoring="r2", n_jobs=N_JOBS,
        )
        self._cv_scores = {
            "r2_folds": (cv_r2 * 100).tolist(),
            "r2_mean": round(cv_r2.mean() * 100, 2),
            "r2_std": round(cv_r2.std() * 100, 2),
        }

        # v2: StackingRegressor with Ridge meta-learner
        self.model = StackingRegressor(
            estimators      = estimators,
            final_estimator = Ridge(alpha=1.0),
            cv              = 5,
            n_jobs          = N_JOBS,
            passthrough     = False,
        )
        self.model.fit(X_train, y_tr_log)

        y_pred_log = self.model.predict(X_test)
        y_pred     = np.expm1(y_pred_log)

        self.metrics_ = {
            "r2"  : round(r2_score(y_test, y_pred)                * 100, 2),
            "mae" : round(mean_absolute_error(y_test, y_pred),       2),
            "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            "cv_scores": self._cv_scores,
        }

        print(f"\n[Revenue Regression - Stacking v2]  R2={self.metrics_['r2']}%  "
              f"MAE={self.metrics_['mae']:,.0f}  RMSE={self.metrics_['rmse']:,.0f}  "
              f"CV R2={self._cv_scores['r2_mean']}% +/- {self._cv_scores['r2_std']}%")
        return self.metrics_

    def predict(self, X) -> np.ndarray:
        return np.expm1(self.model.predict(X))

    def predict_interval(self, X, confidence: float = 0.90) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate prediction intervals from fitted stack base-estimator spread."""
        if not hasattr(self.model, "estimators_"):
            pred = self.predict(X)
            return pred, pred

        alpha = (1 - confidence) / 2
        base_preds = np.vstack([est.predict(X) for est in self.model.estimators_])
        lower = np.expm1(np.quantile(base_preds, alpha, axis=0))
        upper = np.expm1(np.quantile(base_preds, 1 - alpha, axis=0))
        return np.maximum(lower, 0), np.maximum(upper, 0)

    def save(self, path="models/revenue_regressor.joblib"):
        state = {
            "model": self.model,
            "metrics_": self.metrics_,
            "_cv_scores": self._cv_scores,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path="models/revenue_regressor.joblib"):
        state = joblib.load(path)
        if isinstance(state, dict):
            obj = cls()
            obj.model = state.get("model")
            obj.metrics_ = state.get("metrics_", {})
            obj._cv_scores = state.get("_cv_scores", {})
            return obj
        return state


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Visitors Regression
# ═══════════════════════════════════════════════════════════════════════════════

class VisitorsRegressor:
    """
    Predicts annual visitor count for a destination.
    v2: StackingRegressor with Ridge meta-learner + ElasticNet diversity.
    """

    def __init__(self) -> None:
        self.model   : Optional[Any] = None
        self.metrics_: Dict          = {}
        self._cv_scores: Dict        = {}

    def train(self, X_train, y_train, X_test, y_test) -> Dict:
        y_tr_log = np.log1p(y_train)

        estimators = []
        if XGB_AVAILABLE:
            estimators.append(("xgb", xgb.XGBRegressor(
                n_estimators=250, max_depth=6, learning_rate=0.06,
                subsample=0.85, colsample_bytree=0.80,
                random_state=RANDOM_STATE, n_jobs=N_JOBS,
            )))
        if LGB_AVAILABLE:
            estimators.append(("lgb", lgb.LGBMRegressor(
                n_estimators=250, max_depth=6, learning_rate=0.06,
                num_leaves=63, subsample=0.85, colsample_bytree=0.80,
                random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=-1,
            )))
        # v2: always include RF and ElasticNet for robustness
        estimators.append(("rf", RandomForestRegressor(
            n_estimators=200, max_depth=10,
            random_state=RANDOM_STATE, n_jobs=N_JOBS,
        )))
        estimators.append(("enet", ElasticNet(
            alpha=0.1, l1_ratio=0.5, max_iter=2000,
            random_state=RANDOM_STATE,
        )))

        # v2: cross-validation
        from sklearn.model_selection import cross_val_score
        cv_r2 = cross_val_score(
            StackingRegressor(
                estimators=[(n, clone(e)) for n, e in estimators],
                final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=N_JOBS,
            ),
            X_train, y_tr_log, cv=5, scoring="r2", n_jobs=N_JOBS,
        )
        self._cv_scores = {
            "r2_folds": (cv_r2 * 100).tolist(),
            "r2_mean": round(cv_r2.mean() * 100, 2),
            "r2_std": round(cv_r2.std() * 100, 2),
        }

        # v2: StackingRegressor
        self.model = StackingRegressor(
            estimators      = estimators,
            final_estimator = Ridge(alpha=1.0),
            cv              = 5,
            n_jobs          = N_JOBS,
            passthrough     = False,
        )
        self.model.fit(X_train, y_tr_log)
        y_pred = np.expm1(self.model.predict(X_test))

        self.metrics_ = {
            "r2"  : round(r2_score(y_test, y_pred)                  * 100, 2),
            "mae" : round(mean_absolute_error(y_test, y_pred),         2),
            "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            "cv_scores": self._cv_scores,
        }

        print(f"[Visitors Regression - Stacking v2] R2={self.metrics_['r2']}%  "
              f"MAE={self.metrics_['mae']:,.0f}  RMSE={self.metrics_['rmse']:,.0f}  "
              f"CV R2={self._cv_scores['r2_mean']}% +/- {self._cv_scores['r2_std']}%")
        return self.metrics_

    def predict(self, X) -> np.ndarray:
        return np.expm1(self.model.predict(X))

    def predict_interval(self, X, confidence: float = 0.90) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate prediction intervals from fitted stack base-estimator spread."""
        if not hasattr(self.model, "estimators_"):
            pred = self.predict(X)
            return pred, pred

        alpha = (1 - confidence) / 2
        base_preds = np.vstack([est.predict(X) for est in self.model.estimators_])
        lower = np.expm1(np.quantile(base_preds, alpha, axis=0))
        upper = np.expm1(np.quantile(base_preds, 1 - alpha, axis=0))
        return np.maximum(lower, 0), np.maximum(upper, 0)

    def save(self, path="models/visitors_regressor.joblib"):
        state = {
            "model": self.model,
            "metrics_": self.metrics_,
            "_cv_scores": self._cv_scores,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path="models/visitors_regressor.joblib"):
        state = joblib.load(path)
        if isinstance(state, dict):
            obj = cls()
            obj.model = state.get("model")
            obj.metrics_ = state.get("metrics_", {})
            obj._cv_scores = state.get("_cv_scores", {})
            return obj
        return state


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Convenience: train_all_models
# ═══════════════════════════════════════════════════════════════════════════════

def train_all_models(
    df: pd.DataFrame,
    preprocessor,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[Dict, HighRevenuePotentialClassifier, RevenueRegressor, VisitorsRegressor]:
    """
    Train all three models and return (metrics_dict, clf, rev_reg, vis_reg).
    This is the one-call entry point used by the Streamlit app.
    """
    results: Dict = {}

    # ── Classification ───────────────────────────────────────────────────────
    if progress_callback:
        progress_callback(0.15, "Step 1/3: Tuning classifier hyperparameters (this takes ~40s)…")
    print("\n" + "="*60)
    print("  PHASE 1/3: Classification (Stacking + Optuna)")
    print("="*60)
    clf_features = preprocessor.get_classification_features()
    X_tr, X_te, y_tr, y_te = preprocessor.split(
        df, features=clf_features, target="High_Revenue_Potential"
    )
    clf = HighRevenuePotentialClassifier()
    results["classification"] = clf.train(X_tr, y_tr, X_te, y_te)
    clf.save()

    # ── Revenue Regression ───────────────────────────────────────────────────
    if progress_callback:
        progress_callback(0.50, "Step 2/3: Training revenue regression model…")
    print("\n" + "="*60)
    print("  PHASE 2/3: Revenue Regression (Stacking)")
    print("="*60)
    rev_feats = preprocessor.get_revenue_regression_features()
    # We need Log_Revenue present in the visitors features, so add it
    if "Log_Revenue" not in df.columns:
        df["Log_Revenue"] = np.log1p(df["Revenue"])
    X_tr_r, X_te_r, y_tr_r, y_te_r = preprocessor.split(
        df, features=rev_feats, target="Revenue"
    )
    rev_reg = RevenueRegressor()
    results["revenue_regression"] = rev_reg.train(X_tr_r, y_tr_r, X_te_r, y_te_r)
    rev_reg.save()

    # ── Visitors Regression ──────────────────────────────────────────────────
    if progress_callback:
        progress_callback(0.75, "Step 3/3: Training visitor regression model…")
    print("\n" + "="*60)
    print("  PHASE 3/3: Visitors Regression (Stacking)")
    print("="*60)
    vis_feats = preprocessor.get_visitors_regression_features()
    X_tr_v, X_te_v, y_tr_v, y_te_v = preprocessor.split(
        df, features=vis_feats, target="Visitors"
    )
    vis_reg = VisitorsRegressor()
    results["visitors_regression"] = vis_reg.train(X_tr_v, y_tr_v, X_te_v, y_te_v)
    vis_reg.save()

    if progress_callback:
        progress_callback(1.0, "All models trained successfully!")

    return results, clf, rev_reg, vis_reg


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_preprocessing import TourismDataPreprocessor, generate_synthetic_dataset

    raw  = generate_synthetic_dataset(n=600)
    prep = TourismDataPreprocessor()
    df   = prep.preprocess(raw)

    metrics, clf, rev_reg, vis_reg = train_all_models(df, prep)
    print("\n=== Training Complete (v2 - Stacking + Optuna) ===")
    print(f"Classification Accuracy : {metrics['classification']['accuracy']}%")
    print(f"Classification Recall   : {metrics['classification']['recall']}%")
    cv_s = metrics['classification'].get('cv_summary', {})
    print(f"Classification CV Acc   : {cv_s.get('cv_accuracy_mean', '?')}% "
          f"+/- {cv_s.get('cv_accuracy_std', '?')}%")
    print(f"Revenue Regression R2   : {metrics['revenue_regression']['r2']}%")
    print(f"Visitors Regression R2  : {metrics['visitors_regression']['r2']}%")
    print(f"\nModel comparison: {list(metrics['classification'].get('model_comparison', {}).keys())}")

    print("\n=== Verifying Model Loading ===")
    try:
        clf_loaded = HighRevenuePotentialClassifier.load()
        print("[OK] Classifier loaded successfully. Internal model type:", type(clf_loaded.model))
        
        rev_loaded = RevenueRegressor.load()
        print("[OK] Revenue Regressor loaded successfully. Internal model type:", type(rev_loaded.model))
        
        vis_loaded = VisitorsRegressor.load()
        print("[OK] Visitors Regressor loaded successfully. Internal model type:", type(vis_loaded.model))
        
        # Test predictions with loaded models
        clf_feats = prep.get_classification_features()
        X_tr, _, _, _ = prep.split(df, features=clf_feats, target="High_Revenue_Potential")
        pred_clf = clf_loaded.predict(X_tr.head(5))
        print("[OK] Loaded Classifier sample prediction:", pred_clf)
        
        rev_feats = prep.get_revenue_regression_features()
        X_tr_r, _, _, _ = prep.split(df, features=rev_feats, target="Revenue")
        pred_rev = rev_loaded.predict(X_tr_r.head(5))
        print("[OK] Loaded Revenue Regressor sample prediction:", pred_rev)
        
        vis_feats = prep.get_visitors_regression_features()
        X_tr_v, _, _, _ = prep.split(df, features=vis_feats, target="Visitors")
        pred_vis = vis_loaded.predict(X_tr_v.head(5))
        print("[OK] Loaded Visitors Regressor sample prediction:", pred_vis)
        
        print("\nAll model load/save verifications passed!")
    except Exception as ex:
        print("[ERROR] Model loading or prediction failed:", ex)
        raise ex
