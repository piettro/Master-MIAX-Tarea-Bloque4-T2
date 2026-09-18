"""Central configuration of the project.

Every path, constant, cost, feature list and hyper-parameter used by the
pipeline lives here so that no magic number is hard-coded elsewhere.
Runtime options that the command line can change are grouped in
:class:`RunSettings`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CONSTRUCTION_FILENAME = "cs_construccion.csv"
PRODUCTION_FILENAME = "cs_produccion.csv"
DICTIONARY_FILENAME = "DataDictionary.csv"
DICTIONARY_SEPARATOR = ";"

# Delivery files and column name mandated by the original submission.
SCENARIO_1_OUTPUT_FILENAME = "cs_produccion1.csv"
SCENARIO_2_OUTPUT_FILENAME = "cs_produccion2.csv"
PREDICTION_COLUMN = "prediccion"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Data schema
# --------------------------------------------------------------------------
TARGET = "SeriousDlqin2yrs"

REVOLVING_UTILIZATION = "RevolvingUtilizationOfUnsecuredLines"
AGE = "age"
PAST_DUE_30_59 = "NumberOfTime30-59DaysPastDueNotWorse"
DEBT_RATIO = "DebtRatio"
MONTHLY_INCOME = "MonthlyIncome"
OPEN_CREDIT_LINES = "NumberOfOpenCreditLinesAndLoans"
PAST_DUE_90 = "NumberOfTimes90DaysLate"
REAL_ESTATE_LINES = "NumberRealEstateLoansOrLines"
PAST_DUE_60_89 = "NumberOfTime60-89DaysPastDueNotWorse"
DEPENDENTS = "NumberOfDependents"

BASE_FEATURES: tuple[str, ...] = (
    REVOLVING_UTILIZATION,
    AGE,
    PAST_DUE_30_59,
    DEBT_RATIO,
    MONTHLY_INCOME,
    OPEN_CREDIT_LINES,
    PAST_DUE_90,
    REAL_ESTATE_LINES,
    PAST_DUE_60_89,
    DEPENDENTS,
)
DELINQUENCY_COLUMNS: tuple[str, ...] = (
    PAST_DUE_30_59,
    PAST_DUE_60_89,
    PAST_DUE_90,
)
HEAVY_TAIL_COLUMNS: tuple[str, ...] = (
    REVOLVING_UTILIZATION,
    DEBT_RATIO,
    MONTHLY_INCOME,
)

# Flags created by the cleaner (informative missingness / special codes).
MISSING_INCOME_FLAG = "MissingIncome"
SPECIAL_CODE_FLAG = "SpecialDelinquencyCode"
FLAG_FEATURES: tuple[str, ...] = (MISSING_INCOME_FLAG, SPECIAL_CODE_FLAG)

# Derived features (feature engineering).
TOTAL_PAST_DUE = "TotalPastDue"
INCOME_PER_DEPENDENT = "IncomePerDependent"
MONTHLY_DEBT_AMOUNT = "MonthlyDebtAmount"
REAL_ESTATE_SHARE = "RealEstateLineShare"
DERIVED_FEATURES: tuple[str, ...] = (
    TOTAL_PAST_DUE,
    INCOME_PER_DEPENDENT,
    MONTHLY_DEBT_AMOUNT,
    REAL_ESTATE_SHARE,
)

CLEAN_FEATURES: tuple[str, ...] = BASE_FEATURES + FLAG_FEATURES
EXTENDED_FEATURES: tuple[str, ...] = CLEAN_FEATURES + DERIVED_FEATURES

# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
MIN_VALID_AGE = 1                 # age == 0 is a recording error
SPECIAL_CODE_MIN_VALUE = 96       # 96 / 98 are administrative codes
SPECIAL_CODE_REPLACEMENT = 20     # above any genuine count, keeps order
CAP_QUANTILE = 0.99               # winsorisation of heavy tails

# --------------------------------------------------------------------------
# Cost scenarios (assignment brief, section 1)
# --------------------------------------------------------------------------
SCENARIO_1_KEY = "scenario_1"
SCENARIO_2_KEY = "scenario_2"
SCENARIO_COSTS: dict[str, tuple[float, float]] = {
    # key: (cost of a false positive, cost of a false negative)
    SCENARIO_1_KEY: (1.0, 1.0),
    SCENARIO_2_KEY: (1.0, 10.0),
}
SCENARIO_LABELS: dict[str, str] = {
    SCENARIO_1_KEY: "Scenario 1 (FP=1, FN=1)",
    SCENARIO_2_KEY: "Scenario 2 (FP=1, FN=10)",
}
SCENARIO_OUTPUT_FILENAMES: dict[str, str] = {
    SCENARIO_1_KEY: SCENARIO_1_OUTPUT_FILENAME,
    SCENARIO_2_KEY: SCENARIO_2_OUTPUT_FILENAME,
}
# Scenario whose policy is audited in depth (most restrictive one).
AUDIT_SCENARIO_KEY = SCENARIO_2_KEY

# Decision convention: 1 = predicted default -> deny, 0 = approve.
ACTION_APPROVE = 0
ACTION_DENY = 1
ACTION_NAMES: dict[int, str] = {ACTION_APPROVE: "approve", ACTION_DENY: "deny"}

# --------------------------------------------------------------------------
# Model hyper-parameters
# --------------------------------------------------------------------------
VALIDATION_SIZE = 0.20

LOGISTIC_PARAMS: dict[str, object] = {"C": 1.0, "max_iter": 1000}

# The professor's first model of choice (no scaling needed).
RANDOM_FOREST_PARAMS: dict[str, object] = {
    "n_estimators": 200,
    "min_samples_leaf": 20,
    "max_features": "sqrt",
    "n_jobs": -1,
}
# Single white-box tree used as a sanity check (professor's advice).
DECISION_TREE_PARAMS: dict[str, object] = {
    "max_depth": 4,
    "min_samples_leaf": 50,
}
# Best configuration found by the randomised search of the original
# submission (3-fold CV AUC = 0.864); used unless ``--tune`` is passed.
HIST_GB_PARAMS: dict[str, object] = {
    "learning_rate": 0.03,
    "max_leaf_nodes": 31,
    "max_depth": 4,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "max_iter": 500,
    "early_stopping": True,
    "validation_fraction": 0.15,
}
HIST_GB_SEARCH_SPACE: dict[str, list[object]] = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [20, 50, 100],
    "l2_regularization": [0.0, 0.1, 1.0],
    "max_depth": [None, 4, 6],
}
HIST_GB_SEARCH_ITERATIONS = 15
XGBOOST_PARAMS: dict[str, object] = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 4,
    "tree_method": "hist",
    "eval_metric": "logloss",
    "n_jobs": -1,
}
# Sign of the expected effect on the default probability, only where the
# credit-risk domain makes it unambiguous (+1 more risk, -1 less risk).
MONOTONIC_SIGNS: dict[str, int] = {
    REVOLVING_UTILIZATION: +1,
    PAST_DUE_30_59: +1,
    PAST_DUE_60_89: +1,
    PAST_DUE_90: +1,
    TOTAL_PAST_DUE: +1,
    SPECIAL_CODE_FLAG: +1,
    DEBT_RATIO: +1,
    AGE: -1,
    MONTHLY_INCOME: -1,
    INCOME_PER_DEPENDENT: -1,
}

# --------------------------------------------------------------------------
# Contextual bandit (reinforcement-learning approach)
# --------------------------------------------------------------------------
BANDIT_PRIOR_PRECISION = 1.0
BANDIT_NOISE_STD = 1.0
BANDIT_ROLLING_WINDOW = 1000

# --------------------------------------------------------------------------
# Explainability
# --------------------------------------------------------------------------
SURROGATE_MAX_DEPTH = 3
SURROGATE_LEAF_GRID: tuple[int, ...] = (2, 3, 4, 6, 8, 12, 16, 24, 32, 64)
SURROGATE_MIN_SAMPLES_LEAF = 20
N_TOP_FEATURES = 4
LIME_NUM_FEATURES = 8
DEPENDENCE_GRID_POINTS = 20
BORDERLINE_MARGIN = 0.03

# Counterfactuals: only features a client can act upon, in the plausible
# direction.  ``None`` as max relative change means "down to zero".
ACTIONABLE_DIRECTIONS: dict[str, str] = {
    REVOLVING_UTILIZATION: "decrease",
    DEBT_RATIO: "decrease",
    MONTHLY_INCOME: "increase",
    OPEN_CREDIT_LINES: "decrease",
}
MAX_INCOME_INCREASE = 0.30
INTEGER_FEATURES: tuple[str, ...] = (
    AGE,
    PAST_DUE_30_59,
    PAST_DUE_60_89,
    PAST_DUE_90,
    OPEN_CREDIT_LINES,
    REAL_ESTATE_LINES,
    DEPENDENTS,
)
COUNTERFACTUALS_PER_CLIENT = 2
COUNTERFACTUAL_CLIENTS_PER_CLASS = 2
# Moderate denials (threshold <= p <= this value) are the realistic
# candidates for an actionable reversal.
COUNTERFACTUAL_MAX_PROBA = 0.20
# Features never quoted to a client as a reason for denial.
NON_COMMUNICABLE_FEATURES: tuple[str, ...] = (AGE,)

# --------------------------------------------------------------------------
# Audit extras
# --------------------------------------------------------------------------
AGE_BINS: tuple[int, ...] = (18, 30, 40, 50, 60, 70, 120)
DISPARATE_IMPACT_LIMIT = 0.80
PSI_BINS = 10
PSI_STABLE_LIMIT = 0.10
PSI_SEVERE_LIMIT = 0.25
REJECT_INFERENCE_ACCEPTANCE_SHARE = 0.80
CONFIDENCE_LEVEL = 0.95
CALIBRATION_BINS = 10
LIFT_BINS = 10

# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
FIGURE_DPI = 110


@dataclass(frozen=True)
class RunSettings:
    """Runtime options of the pipeline.

    Attributes:
        data_dir: Folder with the input CSV files.
        output_dir: Folder where figures and reports are written.
        production_output_dir: Folder where the two delivery files go.
        seed: Global random seed.
        cv_folds: Folds of the out-of-fold model selection.
        tune: Whether to rerun the HistGB randomised search.
        include_xgboost: Whether XGBoost joins the candidate set.
        random_forest_trees: Number of trees of the random forest.
        bandit_steps: Online interactions used to train each bandit.
        random_benchmark_steps: Interactions of the random policy.
        shap_sample_size: Rows explained with SHAP.
        dependence_sample_size: Rows used for PDP / ALE.
        permutation_repeats: Repeats of the permutation importance.
        bootstrap_samples: Resamples of the cost bootstrap.
        counterfactual_candidates: Random candidates per client.
        show_progress: Whether to display progress bars.
    """

    data_dir: Path = DATA_DIR
    output_dir: Path = OUTPUT_DIR
    production_output_dir: Path = DATA_DIR
    seed: int = RANDOM_SEED
    cv_folds: int = 5
    tune: bool = False
    include_xgboost: bool = True
    random_forest_trees: int = 200
    bandit_steps: int = 30_000
    random_benchmark_steps: int = 10_000
    shap_sample_size: int = 2_000
    dependence_sample_size: int = 3_000
    permutation_repeats: int = 5
    bootstrap_samples: int = 2_000
    counterfactual_candidates: int = 4_000
    show_progress: bool = True

    @property
    def figures_dir(self) -> Path:
        """Folder for the generated figures."""
        return self.output_dir / "figures"

    @property
    def reports_dir(self) -> Path:
        """Folder for the generated tables and reports."""
        return self.output_dir / "reports"

    def quick(self) -> RunSettings:
        """Return a lighter copy of the settings for smoke runs.

        Returns:
            A new :class:`RunSettings` with reduced sample sizes.
        """
        return replace(
            self,
            cv_folds=3,
            random_forest_trees=50,
            bandit_steps=3_000,
            random_benchmark_steps=2_000,
            shap_sample_size=300,
            dependence_sample_size=500,
            permutation_repeats=2,
            bootstrap_samples=200,
            counterfactual_candidates=1_000,
        )
