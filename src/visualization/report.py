"""Markdown report assembled from the pipeline results."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from src.utils import config
from src.utils.config import RunSettings
from src.utils.reporting import dataframe_to_markdown

COMPARISON_COLUMNS = (
    "description",
    "oof_auc",
    "holdout_auc",
    "holdout_ece",
    f"{config.SCENARIO_1_KEY}_holdout_cost",
    f"{config.SCENARIO_2_KEY}_holdout_cost",
    "selection_score",
)
POLICY_ROWS = (
    "theoretical_threshold",
    "threshold",
    "cost",
    "cost_ci_low",
    "cost_ci_high",
    "denial_rate",
    "default_recall",
    "precision_when_denying",
    "accuracy",
    "approve_all",
    "deny_all",
)


def _figure(name: str, caption: str) -> str:
    """Markdown image pointing to ``outputs/figures``."""
    return f"![{caption}](figures/{name})"


def _section_data(eda: dict[str, Any]) -> list[str]:
    """Data quality and population stability."""
    anomalies = pd.Series(eda["anomalies"], name="value").to_frame()
    missing = eda["default_rate_by_income_missingness"]
    return [
        "## 1. Data and exploratory analysis",
        f"* Construction rows: {eda['rows_construction']:,}; production "
        f"rows: {eda['rows_production']:,}.",
        f"* Default rate: {eda['default_rate']:.2%} -> accuracy of the "
        f"trivial majority classifier: {eda['majority_class_accuracy']:.2%}"
        " (accuracy is therefore not a useful metric).",
        f"* Default rate with income missing: {missing['missing']:.2%} vs "
        f"present: {missing['present']:.2%} -> missingness is "
        "informative, so it is kept as a flag instead of dropping rows.",
        "",
        "Missing values (%):",
        "",
        dataframe_to_markdown(eda["missing_pct"].rename("pct").to_frame(),
                              2),
        "",
        "Data-quality anomalies:",
        "",
        dataframe_to_markdown(anomalies, 3),
        "",
        "Population stability (PSI, construction vs production):",
        "",
        dataframe_to_markdown(eda["psi"], 5),
        "",
        _figure("02_missing_values.png", "Missing values"),
        _figure("04_default_segments.png", "Default rate by segment"),
        _figure("05_correlation_clustermap.png", "Correlation"),
        _figure("06_pca_projection.png", "PCA"),
        "",
    ]


def _section_models(selection: dict[str, Any]) -> list[str]:
    """Model comparison, statistical tests and scenario policies."""
    table = selection["comparison"].loc[:, list(COMPARISON_COLUMNS)]
    policies = pd.DataFrame(selection["policies_holdout"]).loc[
        list(POLICY_ROWS)]
    policies.columns = [config.SCENARIO_LABELS[c] for c in policies.columns]
    delong = pd.DataFrame(selection["delong"]).T
    return [
        "## 2. Supervised models and cost-optimal thresholds",
        "Candidates are ranked by their out-of-fold cost on the training "
        "split (each with its own OOF-optimal thresholds); the validation "
        "split is only used to report unbiased hold-out figures.",
        "",
        dataframe_to_markdown(table),
        "",
        f"**Selected model: `{selection['winner']}`** - "
        f"{selection['winner_description']}.",
        "",
        "DeLong tests on the validation split (AUC difference = A - B):",
        "",
        dataframe_to_markdown(delong),
        "",
        "Hold-out behaviour of the selected policy (validation split):",
        "",
        dataframe_to_markdown(policies),
        "",
        _figure("08_roc_pr.png", "ROC and PR"),
        _figure("09_calibration.png", "Calibration"),
        _figure("10_cost_curves.png", "Cost curves"),
        _figure("11_confusion_matrices.png", "Confusion matrices"),
        _figure("13_cost_bootstrap.png", "Cost bootstrap"),
        "",
    ]


def _section_production(
    production: dict[str, Any], n_folds: int
) -> list[str]:
    """Delivered files."""
    files = pd.DataFrame(production["files"]).T.drop(columns=["path"])
    files.index = [config.SCENARIO_OUTPUT_FILENAMES[k] for k in files.index]
    return [
        "## 3. Production decisions",
        "The selected configuration is refitted on every construction row; "
        f"its thresholds come from {n_folds}-fold out-of-fold predictions "
        "of that same configuration on the full history.",
        "",
        dataframe_to_markdown(files),
        "",
    ]


def _section_bandit(bandit: dict[str, Any]) -> list[str]:
    """Reinforcement-learning comparison."""
    selection = bandit["feature_selection"]
    return [
        "## 4. Multi-armed contextual bandit",
        "Mean cost per client (lower is better). The random policy is the "
        "benchmark used in class.",
        "",
        dataframe_to_markdown(bandit["comparison"]),
        "",
        "XAI-driven feature selection (class exercise): the bandit is "
        f"retrained with the surrogate's variables "
        f"`{selection['selected_features']}` -> validation cost "
        f"{selection['cost_selected_features']:.4f} vs "
        f"{selection['cost_all_features']:.4f} with every variable.",
        "",
        _figure("15_bandit_learning.png", "Bandit learning curve"),
        "",
    ]


def _section_surrogates(surrogates: dict[str, Any]) -> list[str]:
    """Surrogate fidelity and rules."""
    lines = ["## 5. Surrogate models (global rules)"]
    for name, summary in surrogates.items():
        if name == "fidelity_curves":
            continue
        lines += [
            f"### {name}",
            f"Fidelity {summary['fidelity']:.2%} with {summary['n_rules']} "
            f"rules (majority baseline {summary['majority_baseline']:.2%}).",
            "",
            "```text",
            summary["rules_text"],
            "```",
            "",
        ]
    lines += [_figure("18_fidelity_tradeoff.png", "Fidelity trade-off"),
              _figure(f"16_surrogate_supervised_{config.AUDIT_SCENARIO_KEY}"
                      ".png", "Surrogate tree"), ""]
    return lines


def _section_local(local: dict[str, Any]) -> list[str]:
    """SHAP, LIME and feature effects."""
    importance = local["shap_importance"].rename("mean_abs_shap").to_frame()
    cases = pd.DataFrame({
        name: {
            "default_proba": case["default_proba"],
            "top_risk_drivers": ", ".join(case["top_risk_drivers"].index),
        }
        for name, case in local["local_cases"].items()
    }).T
    return [
        "## 6. SHAP, LIME and feature effects (deployed model)",
        dataframe_to_markdown(importance),
        "",
        "Local explanations of three representative production clients:",
        "",
        dataframe_to_markdown(cases),
        "",
        "LIME on the clearest denial (to contrast with SHAP drivers "
        f"`{list(local['shap_top_clear_denial'].index)}`):",
        "",
        dataframe_to_markdown(local["lime_clear_denial"].set_index(
            "condition")),
        "",
        _figure("19_shap_beeswarm.png", "SHAP beeswarm"),
        _figure("22_shap_waterfall_borderline_denial.png",
                "SHAP borderline"),
        _figure("23_pdp_vs_ale.png", "PDP vs ALE"),
        "",
    ]


def _section_counterfactuals(
    counterfactuals: list[dict[str, Any]], local: dict[str, Any]
) -> list[str]:
    """Counterfactuals and the letter for a denied client."""
    lines = [
        "## 7. Counterfactuals and what we tell a denied client",
        "Moderately denied validation clients of both real classes; only "
        "actionable variables may change, in a plausible direction, and "
        "the result must fall below the policy threshold.",
        "",
    ]
    for item in counterfactuals:
        verdict = "found" if item["found"] else (
            "none: the denial cannot be reversed with actionable changes")
        lines += [
            f"**Validation client {item['validation_row']} - real class "
            f"{item['real_class']} - p(default) = "
            f"{item['default_proba']:.3f} - counterfactual {verdict}**",
            "",
            dataframe_to_markdown(item["table"], 3),
            "",
        ]
    withheld = local["withheld_drivers"]
    lines += [
        "Letter generated for the borderline denial:",
        "",
        "```text",
        local["denial_letter"],
        "```",
        "",
        "Drivers withheld from the client and flagged for internal review: "
        f"`{withheld}`." if withheld else
        "No protected attribute among the client's main drivers.",
        "",
    ]
    return lines


def _section_audit(extras: dict[str, Any]) -> list[str]:
    """Fairness, permutation importance and selection bias."""
    return [
        "## 8. Fairness, robustness and selection bias",
        "Fairness by age band (audited scenario, validation split):",
        "",
        dataframe_to_markdown(extras["fairness_summary"], 3),
        "",
        f"Cost without the age variable: {extras['cost_without_age']:.4f}.",
        "",
        dataframe_to_markdown(extras["fairness_by_age_with_age"], 3),
        "",
        "Permutation importance (drop in validation AUC):",
        "",
        dataframe_to_markdown(extras["permutation_importance"]),
        "",
        "Selection bias simulation (validation AUC):",
        "",
        dataframe_to_markdown(extras["reject_inference"].to_frame()),
        "",
        _figure("25_fairness_age.png", "Fairness"),
        "",
    ]


def _section_reflection(results: dict[str, Any]) -> list[str]:
    """Data-driven closing reflection."""
    selection = results["model_selection"]
    audit = results["surrogates"][
        f"supervised_{config.AUDIT_SCENARIO_KEY}"]
    extras = results["audit_extras"]
    di = extras["fairness_summary"].loc["disparate_impact", "with_age"]
    audited_cost = selection["policies_holdout"][
        config.AUDIT_SCENARIO_KEY]["cost"]
    bias = extras["reject_inference"]
    bias_loss = (bias["oracle_full_population"]
                 - bias["biased_accepted_only"])
    inference_gain = (bias["reject_inference_parceling"]
                      - bias["biased_accepted_only"])
    shap_top = list(
        results["local_explanations"]["shap_importance"].index[:3])
    permutation_top = list(extras["permutation_importance"].index[:3])
    thresholds = ", ".join(
        f"{config.SCENARIO_LABELS[k]}: {p['threshold']:.3f} vs Bayes "
        f"{p['theoretical_threshold']:.3f}"
        for k, p in selection["policies_holdout"].items())
    return [
        "## 9. Final reflection",
        f"* **Decision vs model.** One probabilistic model "
        f"(`{selection['winner']}`) serves both scenarios; only the "
        f"threshold changes ({thresholds}). The closer the empirical "
        "threshold is to c_FP / (c_FP + c_FN), the better calibrated the "
        "model.",
        f"* **Global explanation.** A {audit['n_rules']}-rule surrogate "
        f"reproduces {audit['fidelity']:.1%} of the audited decisions "
        f"(majority baseline {audit['majority_baseline']:.1%}). Top SHAP "
        f"drivers: `{shap_top}`; top permutation drivers: "
        f"`{permutation_top}` - agreement between independent techniques "
        "is what makes the audit credible.",
        f"* **Fairness.** Disparate impact across age bands is {di:.2f} "
        f"(legal reference {config.DISPARATE_IMPACT_LIMIT}). Removing age "
        f"moves the audited cost from {audited_cost:.4f}"
        f" to {extras['cost_without_age']:.4f} while redistributing "
        "denials across age bands: the trade-off is between equity and "
        "cost, and it is a business / regulatory decision.",
        "* **Limits.** The history only contains accepted clients: in the "
        f"simulation, training on accepted clients only costs "
        f"{bias_loss:.3f} AUC and parceling reject inference changes it "
        f"by {inference_gain:+.3f} - it creates no information, so the "
        "real fix is operational "
        "(approving a small random sample of rejected applicants). The "
        "validation cost is an estimate: see the bootstrap interval, and "
        "monitor PSI in production.",
        "",
    ]


def build_markdown_report(
    results: dict[str, Any], settings: RunSettings
) -> str:
    """Assemble the full Markdown report.

    Args:
        results: Output of ``pipeline.run_pipeline``.
        settings: Runtime settings used for the run.

    Returns:
        The report as a Markdown string.
    """
    header = [
        "# Explainable credit scoring - results report",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M} | seed "
        f"{settings.seed} | {settings.cv_folds}-fold selection | bandit "
        f"steps {settings.bandit_steps:,}_",
        "",
        "Convention: prediction 1 = expected default = **deny**; "
        "0 = **approve**. Mean cost per client = (c_FP * FP + c_FN * FN) "
        "/ n.",
        "",
    ]
    body = (
        _section_data(results["eda"])
        + _section_models(results["model_selection"])
        + _section_production(results["production"], settings.cv_folds)
        + _section_bandit(results["bandit"])
        + _section_surrogates(results["surrogates"])
        + _section_local(results["local_explanations"])
        + _section_counterfactuals(results["counterfactuals"],
                                   results["local_explanations"])
        + _section_audit(results["audit_extras"])
        + _section_reflection(results)
    )
    return "\n".join(header + body)
