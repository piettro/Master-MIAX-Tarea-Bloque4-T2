"""Figures of the EDA, model evaluation, bandit and XAI audit.

Every public function draws one figure, saves it as PNG and closes it, so
the pipeline can run head-less.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import shap  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402
from sklearn.tree import plot_tree  # noqa: E402

from src.utils import config  # noqa: E402

logger = logging.getLogger(__name__)

PAY_COLOR = "#4c72b0"
DEFAULT_COLOR = "#c44e52"
ACCENT_COLOR = "#55a868"
NEUTRAL_COLOR = "#8172b3"
SCENARIO_COLORS = {
    config.SCENARIO_1_KEY: PAY_COLOR,
    config.SCENARIO_2_KEY: DEFAULT_COLOR,
}
HISTOGRAM_BINS = 50
ROLLING_MIN_PERIODS = 100

sns.set_theme(style="whitegrid", palette="deep")


def _save(fig: plt.Figure, path: Path) -> Path:
    """Save a figure and release its memory.

    Args:
        fig: Figure to save.
        path: Destination PNG file.

    Returns:
        The path written.

    Raises:
        OSError: If the figure cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    except OSError as exc:
        raise OSError(f"Could not save figure {path}") from exc
    finally:
        plt.close(fig)
    logger.debug("Saved figure %s", path)
    return path


def plot_target_distribution(target: pd.Series, path: Path) -> Path:
    """Bar chart of the class balance.

    Args:
        target: Binary target.
        path: Destination file.

    Returns:
        The path written.
    """
    counts = target.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(["Pays (0)", "Defaults (1)"], counts.values,
           color=[PAY_COLOR, DEFAULT_COLOR])
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value:,}\n({value / counts.sum():.1%})",
                ha="center", va="bottom")
    ax.set_ylabel("Clients")
    ax.set_title("Target distribution")
    return _save(fig, path)


def plot_missing_values(
    missing_pct: pd.Series, default_rates: pd.Series, path: Path
) -> Path:
    """Missing percentages and default rate by income availability.

    Args:
        missing_pct: Percentage of NaNs per column.
        default_rates: Default rate with income ``missing`` / ``present``.
        path: Destination file.

    Returns:
        The path written.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].barh(missing_pct.index, missing_pct.values, color="#dd8452")
    axes[0].set_xlabel("% missing")
    axes[0].set_title("Missing values per column")
    axes[1].bar(["Income missing", "Income present"],
                100 * default_rates[["missing", "present"]].values,
                color=["#937860", PAY_COLOR])
    axes[1].set_ylabel("Default rate (%)")
    axes[1].set_title("Is missing income informative?")
    fig.tight_layout()
    return _save(fig, path)


def plot_feature_histograms(frame: pd.DataFrame, path: Path) -> Path:
    """Histograms of the raw features (log scale for heavy tails).

    Args:
        frame: Raw data with the base features.
        path: Destination file.

    Returns:
        The path written.
    """
    features = list(config.BASE_FEATURES)
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    for ax, col in zip(axes.flat, features):
        values = frame[col].dropna()
        if col in config.HEAVY_TAIL_COLUMNS:
            values = np.log10(values + 1)
            ax.set_xlabel("log10(1 + x)", fontsize=8)
        ax.hist(values, bins=HISTOGRAM_BINS, color=PAY_COLOR)
        ax.set_title(col, fontsize=9)
    for ax in axes.flat[len(features):]:
        ax.axis("off")
    fig.suptitle("Distribution of the explanatory variables")
    fig.tight_layout()
    return _save(fig, path)


def plot_default_rate_segments(
    segments: dict[str, pd.Series], path: Path
) -> Path:
    """Default rate by age band, income decile and 90+ day delinquencies.

    Args:
        segments: Output of ``eda.default_rate_by_segments``.
        path: Destination file.

    Returns:
        The path written.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    age = segments["age"]
    axes[0].plot([iv.mid for iv in age.index], 100 * age.values,
                 marker="o", color=DEFAULT_COLOR)
    axes[0].set_xlabel("Age")
    axes[0].set_title("By age")
    income = segments["income_decile"]
    axes[1].bar(income.index.astype(str), 100 * income.values,
                color=PAY_COLOR)
    axes[1].set_xlabel("Monthly income decile (1 = lowest)")
    axes[1].set_title("By income")
    late = segments["late_90"]
    labels = [f"{int(i)}" if i < late.index.max() else f"{int(i)}+"
              for i in late.index]
    axes[2].bar(labels, 100 * late.values, color=ACCENT_COLOR)
    axes[2].set_xlabel("Number of 90+ days delinquencies")
    axes[2].set_title("By severe delinquency history")
    for ax in axes:
        ax.set_ylabel("Default rate (%)")
    fig.tight_layout()
    return _save(fig, path)


def plot_correlation_clustermap(corr: pd.DataFrame, path: Path) -> Path:
    """Clustered correlation map with a diverging palette centred at zero.

    Follows the professor's recipe: ``bwr`` palette, ``vmin=-1`` so that
    white means no correlation, and clustering to reveal blocks.

    Args:
        corr: Correlation matrix.
        path: Destination file.

    Returns:
        The path written.
    """
    grid = sns.clustermap(corr, cmap="bwr", vmin=-1, vmax=1, annot=True,
                          fmt=".2f", annot_kws={"size": 7},
                          figsize=(11, 11))
    grid.fig.suptitle("Spearman correlation (clustered)", y=1.02)
    return _save(grid.fig, path)


def plot_pca_projection(
    projection: np.ndarray,
    explained: np.ndarray,
    target: np.ndarray,
    path: Path,
) -> Path:
    """Scatter of the first two principal components coloured by class.

    Args:
        projection: PCA coordinates.
        explained: Explained variance ratio of the two components.
        target: Binary target.
        path: Destination file.

    Returns:
        The path written.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, color, alpha in ((0, PAY_COLOR, 0.15),
                                (1, DEFAULT_COLOR, 0.3)):
        sel = target == label
        ax.scatter(projection[sel, 0], projection[sel, 1], s=3, alpha=alpha,
                   color=color, label=["Pays", "Defaults"][label])
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} variance)")
    ax.set_title("PCA projection coloured by class")
    ax.legend(markerscale=4)
    return _save(fig, path)


def plot_psi(psi: pd.DataFrame, path: Path) -> Path:
    """PSI per variable against the conventional stability limits.

    Args:
        psi: Output of ``eda.psi_table``.
        path: Destination file.

    Returns:
        The path written.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ordered = psi["psi"].sort_values()
    ax.barh(ordered.index, ordered.values, color=PAY_COLOR)
    ax.axvline(config.PSI_STABLE_LIMIT, color="orange", ls="--",
               label="Moderate shift")
    ax.axvline(config.PSI_SEVERE_LIMIT, color="red", ls="--",
               label="Severe shift")
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_xlabel("PSI (construction vs production)")
    ax.set_title("Population stability")
    ax.legend()
    return _save(fig, path)


def plot_roc_pr(
    target: np.ndarray, scores: dict[str, np.ndarray], path: Path
) -> Path:
    """ROC and precision-recall curves of several models.

    Args:
        target: Binary labels.
        scores: Default probabilities per model name.
        path: Destination file.

    Returns:
        The path written.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, proba in scores.items():
        fpr, tpr, _ = roc_curve(target, proba)
        precision, recall, _ = precision_recall_curve(target, proba)
        axes[0].plot(fpr, tpr, label=name)
        axes[1].plot(recall, precision, label=name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("False-positive rate")
    axes[0].set_ylabel("True-positive rate")
    axes[0].set_title("ROC curve (validation)")
    axes[1].axhline(np.mean(target), color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("Recall (defaults detected)")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curve (validation)")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, path)


def plot_calibration(
    target: np.ndarray, scores: dict[str, np.ndarray], path: Path
) -> Path:
    """Reliability diagram (quantile bins).

    Args:
        target: Binary labels.
        scores: Default probabilities per model name.
        path: Destination file.

    Returns:
        The path written.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
    for name, proba in scores.items():
        observed, predicted = calibration_curve(
            target, proba, n_bins=config.CALIBRATION_BINS,
            strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration (validation)")
    ax.legend(fontsize=7)
    return _save(fig, path)


def plot_cost_curves(
    curves: pd.DataFrame,
    thresholds: dict[str, float],
    theoretical: dict[str, float],
    labels: dict[str, str],
    path: Path,
) -> Path:
    """Mean cost vs threshold per scenario, and resulting denial rate.

    Args:
        curves: Output of ``metrics.cost_curve_frame``.
        thresholds: Chosen threshold per scenario key.
        theoretical: Bayes threshold per scenario key.
        labels: Readable scenario names.
        path: Destination file.

    Returns:
        The path written.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for key, threshold in thresholds.items():
        color = SCENARIO_COLORS.get(key, NEUTRAL_COLOR)
        axes[0].plot(curves.index, curves[key], color=color,
                     label=labels[key])
        axes[0].axvline(threshold, color=color, ls="--", lw=1)
        axes[0].axvline(theoretical[key], color=color, ls=":", lw=1)
        axes[1].axvline(threshold, color=color, ls="--", lw=1,
                        label=f"{labels[key]}: {threshold:.3f}")
    axes[0].set_xlabel("Threshold on default probability")
    axes[0].set_ylabel("Mean cost per client (validation)")
    axes[0].set_title("Cost vs threshold (dashed: chosen, dotted: Bayes)")
    axes[0].legend(fontsize=8)
    axes[1].plot(curves.index, 100 * curves["denial_rate"],
                 color=ACCENT_COLOR)
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("% of applications denied")
    axes[1].set_title("Business impact of the threshold")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_confusion_matrices(
    matrices: dict[str, np.ndarray], labels: dict[str, str], path: Path
) -> Path:
    """Confusion matrices of the scenario policies.

    Args:
        matrices: 2x2 matrix (rows = truth, cols = decision) per scenario.
        labels: Readable scenario names.
        path: Destination file.

    Returns:
        The path written.
    """
    fig, axes = plt.subplots(1, len(matrices), figsize=(5.3 * len(matrices),
                                                        3.8))
    for ax, (key, matrix) in zip(np.atleast_1d(axes), matrices.items()):
        sns.heatmap(matrix, annot=True, fmt=",", cmap="Blues", cbar=False,
                    ax=ax, xticklabels=["Approve", "Deny"],
                    yticklabels=["Pays", "Defaults"])
        ax.set_title(labels[key])
        ax.set_xlabel("Decision")
        ax.set_ylabel("Truth")
    fig.tight_layout()
    return _save(fig, path)


def plot_lift(lift: pd.DataFrame, path: Path) -> Path:
    """Lift per decile and cumulative capture of defaults.

    Args:
        lift: Output of ``metrics.lift_table``.
        path: Destination file.

    Returns:
        The path written.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(lift.index, lift["lift"], color=PAY_COLOR)
    axes[0].axhline(1, color="k", ls="--", lw=0.8)
    axes[0].set_xlabel("Score decile (1 = riskiest)")
    axes[0].set_ylabel("Lift")
    axes[0].set_title("Lift by decile")
    axes[1].plot(lift.index, 100 * lift["cumulative_capture"], marker="o",
                 color=DEFAULT_COLOR)
    axes[1].plot([1, len(lift)], [100 / len(lift), 100], "k--", lw=0.8)
    axes[1].set_xlabel("Cumulative deciles")
    axes[1].set_ylabel("% of defaults captured")
    axes[1].set_title("Cumulative capture")
    fig.tight_layout()
    return _save(fig, path)


def plot_feature_importance(
    importance: pd.Series, title: str, xlabel: str, path: Path
) -> Path:
    """Horizontal bar chart of an importance ranking.

    Args:
        importance: Importance per feature.
        title: Figure title.
        xlabel: Label of the value axis.
        path: Destination file.

    Returns:
        The path written.
    """
    ordered = importance.sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(ordered) + 1.5))
    ax.barh(ordered.index, ordered.values, color=PAY_COLOR)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return _save(fig, path)


def plot_bandit_learning(
    histories: dict[str, np.ndarray],
    benchmarks: dict[str, float],
    labels: dict[str, str],
    path: Path,
) -> Path:
    """Moving average of the bandit reward during online training.

    Args:
        histories: Rewards per interaction, per scenario key.
        benchmarks: Mean reward of the random policy per scenario key.
        labels: Readable scenario names.
        path: Destination file.

    Returns:
        The path written.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    for key, rewards in histories.items():
        color = SCENARIO_COLORS.get(key, NEUTRAL_COLOR)
        rolling = pd.Series(rewards).rolling(
            config.BANDIT_ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS
        ).mean()
        ax.plot(rolling, color=color, label=f"Bandit - {labels[key]}")
        ax.axhline(benchmarks[key], color=color, ls=":",
                   label=f"Random policy - {labels[key]}")
    ax.set_xlabel("Interaction")
    ax.set_ylabel(f"Mean reward ({config.BANDIT_ROLLING_WINDOW} steps)")
    ax.set_title("Contextual bandit learning curve (reward = -cost)")
    ax.legend(fontsize=7)
    return _save(fig, path)


def plot_surrogate_tree(
    tree: object, feature_names: list[str], title: str, path: Path
) -> Path:
    """Draw a surrogate decision tree.

    Args:
        tree: Fitted ``DecisionTreeClassifier``.
        feature_names: Input names.
        title: Figure title.
        path: Destination file.

    Returns:
        The path written.
    """
    fig, ax = plt.subplots(figsize=(18, 7))
    plot_tree(tree, feature_names=feature_names,
              class_names=[config.ACTION_NAMES[int(c)]
                           for c in tree.classes_],
              filled=True, rounded=True, impurity=False, proportion=True,
              fontsize=8, ax=ax)
    ax.set_title(title)
    return _save(fig, path)


def plot_fidelity_curve(
    curves: dict[str, pd.DataFrame], path: Path
) -> Path:
    """Surrogate fidelity vs number of rules for several black boxes.

    Args:
        curves: Output of ``surrogate.fidelity_curve`` per black box.
        path: Destination file.

    Returns:
        The path written.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, curve in curves.items():
        line, = ax.plot(curve["n_leaves"], 100 * curve["fidelity"],
                        marker="o", label=name)
        ax.axhline(100 * curve["majority_baseline"].iloc[0],
                   color=line.get_color(), ls=":", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of rules (leaves)")
    ax.set_ylabel("Fidelity (%)")
    ax.set_title("Explainability trade-off (dotted: majority baseline)")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_shap_beeswarm(explanation: shap.Explanation, path: Path) -> Path:
    """SHAP beeswarm summary plot.

    Args:
        explanation: SHAP values.
        path: Destination file.

    Returns:
        The path written.
    """
    shap.plots.beeswarm(explanation, max_display=16, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP global: impact on the default score")
    return _save(fig, path)


def plot_shap_waterfall(
    explanation: shap.Explanation, title: str, path: Path
) -> Path:
    """SHAP waterfall of a single client.

    Args:
        explanation: SHAP values of one row.
        title: Figure title.
        path: Destination file.

    Returns:
        The path written.
    """
    shap.plots.waterfall(explanation, max_display=10, show=False)
    fig = plt.gcf()
    fig.suptitle(title)
    return _save(fig, path)


def plot_shap_dependence(
    explanation: shap.Explanation, features: list[str], path: Path
) -> Path:
    """SHAP value vs feature value for the top features.

    Args:
        explanation: SHAP values.
        features: Features to plot.
        path: Destination file.

    Returns:
        The path written.
    """
    names = list(explanation.feature_names)
    fig, axes = plt.subplots(1, len(features),
                             figsize=(4 * len(features), 3.5))
    for ax, feature in zip(np.atleast_1d(axes), features):
        j = names.index(feature)
        ax.scatter(explanation.data[:, j], explanation.values[:, j], s=4,
                   alpha=0.3, color=DEFAULT_COLOR)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel(feature, fontsize=8)
        ax.set_ylabel("SHAP value")
    fig.suptitle("SHAP dependence of the most important features")
    fig.tight_layout()
    return _save(fig, path)


def plot_pdp_ale(
    curves: dict[str, tuple[pd.Series, pd.Series]], path: Path
) -> Path:
    """Partial dependence and ALE side by side for several features.

    Args:
        curves: ``(pdp, ale)`` per feature; the PDP is centred for
            comparability with the ALE.
        path: Destination file.

    Returns:
        The path written.
    """
    fig, axes = plt.subplots(1, len(curves), figsize=(4 * len(curves), 3.5))
    for ax, (feature, (pdp, ale)) in zip(np.atleast_1d(axes),
                                         curves.items()):
        ax.plot(pdp.index, pdp - pdp.mean(), marker=".", color=PAY_COLOR,
                label="PDP (centred)")
        ax.plot(ale.index, ale.values, marker=".", color=NEUTRAL_COLOR,
                label="ALE")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(feature, fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("Effect on default probability")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle("Feature effects: PDP vs ALE")
    fig.tight_layout()
    return _save(fig, path)


def plot_fairness(
    with_age: pd.DataFrame, without_age: pd.DataFrame, path: Path
) -> Path:
    """Denial rate per age band, model with vs without age.

    Args:
        with_age: ``fairness.group_metrics`` of the model with age.
        without_age: Same for the model without age.
        path: Destination file.

    Returns:
        The path written.
    """
    x = np.arange(len(with_age))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, 100 * with_age["denial_rate"], 0.4, label="With age",
           color=DEFAULT_COLOR)
    ax.bar(x + 0.2, 100 * without_age["denial_rate"], 0.4,
           label="Without age", color=PAY_COLOR)
    ax.plot(x, 100 * with_age["default_rate"], "k--o", lw=1,
            label="Actual default rate")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in with_age.index], fontsize=8)
    ax.set_xlabel("Age band")
    ax.set_ylabel("%")
    ax.set_title("Denial rate by age band (audited scenario)")
    ax.legend()
    return _save(fig, path)


def plot_bootstrap(
    distributions: dict[str, np.ndarray],
    intervals: dict[str, tuple[float, float]],
    labels: dict[str, str],
    path: Path,
) -> Path:
    """Bootstrap distribution of the mean cost per scenario.

    Args:
        distributions: Bootstrap costs per scenario key.
        intervals: Confidence interval per scenario key.
        labels: Readable scenario names.
        path: Destination file.

    Returns:
        The path written.
    """
    fig, axes = plt.subplots(1, len(distributions),
                             figsize=(6 * len(distributions), 3.8))
    for ax, (key, values) in zip(np.atleast_1d(axes),
                                 distributions.items()):
        low, high = intervals[key]
        ax.hist(values, bins=40, alpha=0.8,
                color=SCENARIO_COLORS.get(key, NEUTRAL_COLOR))
        ax.axvline(low, color="k", ls="--", lw=1)
        ax.axvline(high, color="k", ls="--", lw=1)
        ax.set_title(f"{labels[key]}\n95% CI = [{low:.4f}, {high:.4f}]",
                     fontsize=9)
        ax.set_xlabel("Mean cost per client")
    fig.tight_layout()
    return _save(fig, path)
