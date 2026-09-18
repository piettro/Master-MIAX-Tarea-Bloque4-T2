"""End-to-end pipeline: data -> models -> production files -> XAI audit.

Each ``run_*`` function is one stage with a single responsibility; they
share a :class:`PreparedData` bundle and return plain dictionaries that
end up in ``outputs/reports/results.json`` and ``report.md``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from src.data import eda
from src.data.cleaning import CreditDataCleaner, drop_invalid_training_rows
from src.data.features import build_feature_matrix
from src.data.loader import (
    load_construction,
    load_data_dictionary,
    load_production,
)
from src.evaluation import fairness
from src.evaluation.costs import (
    CostScenario,
    build_scenarios,
    decisions_from_proba,
    expected_cost,
    policy_summary,
    trivial_policy_costs,
)
from src.evaluation.metrics import (
    bootstrap_cost,
    confidence_interval,
    cost_curve_frame,
    delong_test,
    lift_table,
    majority_class_accuracy,
)
from src.evaluation.model_selection import (
    CandidateResult,
    evaluate_candidates,
    fit_scorer,
    select_best,
)
from src.evaluation.selection_bias import simulate_reject_inference
from src.explainability import surrogate as surrogate_lib
from src.explainability.counterfactuals import (
    CounterfactualSearch,
    default_actionable_features,
    robust_scales,
)
from src.explainability.denial_report import build_denial_letter
from src.explainability.dependence import (
    accumulated_local_effects,
    partial_dependence_curve,
    permutation_auc_importance,
)
from src.explainability.lime_analysis import explain_with_lime
from src.explainability.shap_analysis import (
    compute_shap_values,
    select_local_cases,
    shap_importance,
    top_risk_drivers,
)
from src.models.bandit import (
    BanditDecisionModel,
    ContextEncoder,
    CreditEnvironment,
    LinearThompsonSampling,
    RandomPolicy,
    run_interactions,
)
from src.models.scorer import CreditScorer
from src.models.supervised import build_candidate_specs, tune_hist_gb
from src.utils import config
from src.utils.config import RunSettings
from src.utils.reporting import write_json, write_table, write_text
from src.utils.reproducibility import set_global_seed
from src.visualization import plots
from src.visualization.report import build_markdown_report

logger = logging.getLogger(__name__)

THRESHOLD_GRID = np.linspace(0.005, 0.995, 199)
ROC_MODELS = ("logistic", "random_forest", "decision_tree")


@dataclass
class PreparedData:
    """Cleaned data shared by all stages.

    Attributes:
        construction_raw: Raw labelled data (for the EDA).
        production_raw: Raw production data.
        dictionary: Variable dictionary.
        construction: Cleaned construction features (valid rows only).
        target: Construction labels.
        production: Cleaned production features (every row kept).
        train: Cleaned training split.
        valid: Cleaned validation split.
        y_train: Training labels.
        y_valid: Validation labels.
        scenarios: Cost scenarios.
    """

    construction_raw: pd.DataFrame
    production_raw: pd.DataFrame
    dictionary: pd.DataFrame
    construction: pd.DataFrame
    target: np.ndarray
    production: pd.DataFrame
    train: pd.DataFrame
    valid: pd.DataFrame
    y_train: np.ndarray
    y_valid: np.ndarray
    scenarios: dict[str, CostScenario]


def prepare_data(settings: RunSettings) -> PreparedData:
    """Load, clean and split the data.

    Args:
        settings: Runtime settings.

    Returns:
        A :class:`PreparedData` bundle.
    """
    construction_raw = load_construction(settings.data_dir)
    production_raw = load_production(settings.data_dir)
    dictionary = load_data_dictionary(settings.data_dir)

    valid_rows = drop_invalid_training_rows(construction_raw)
    cleaner = CreditDataCleaner().fit(valid_rows)
    construction = cleaner.transform(valid_rows)
    target = construction.pop(config.TARGET).to_numpy().astype(int)
    production = cleaner.transform(production_raw)
    if len(production) != len(production_raw):
        raise RuntimeError("Cleaning must never drop production rows")

    train, valid, y_train, y_valid = train_test_split(
        construction, target, test_size=config.VALIDATION_SIZE,
        random_state=settings.seed, stratify=target,
    )
    logger.info("Train %d rows / validation %d rows (default rate %.2f%%)",
                len(train), len(valid), 100 * target.mean())
    return PreparedData(
        construction_raw, production_raw, dictionary, construction, target,
        production, train.reset_index(drop=True),
        valid.reset_index(drop=True), y_train, y_valid, build_scenarios(),
    )


def run_eda(data: PreparedData, settings: RunSettings) -> dict[str, Any]:
    """Exploratory analysis and population-stability check.

    Args:
        data: Prepared data.
        settings: Runtime settings.

    Returns:
        EDA results (tables and anomaly counts).
    """
    raw = data.construction_raw
    figures = settings.figures_dir
    missing = eda.missing_value_report(raw)
    missing_vs_default = eda.default_rate_by_missingness(
        raw, config.MONTHLY_INCOME)
    segments = eda.default_rate_by_segments(raw)
    psi = eda.psi_table(data.construction, data.production,
                        config.CLEAN_FEATURES)
    projection, explained = eda.pca_projection(
        data.construction, config.CLEAN_FEATURES, settings.seed)

    plots.plot_target_distribution(raw[config.TARGET],
                                   figures / "01_target_distribution.png")
    plots.plot_missing_values(missing, missing_vs_default,
                              figures / "02_missing_values.png")
    plots.plot_feature_histograms(raw, figures / "03_histograms.png")
    plots.plot_default_rate_segments(segments,
                                     figures / "04_default_segments.png")
    plots.plot_correlation_clustermap(
        eda.spearman_correlation(raw),
        figures / "05_correlation_clustermap.png")
    plots.plot_pca_projection(projection, explained, data.target,
                              figures / "06_pca_projection.png")
    plots.plot_psi(psi, figures / "07_psi.png")

    write_table(eda.describe_features(raw),
                settings.reports_dir / "eda_describe.csv")
    return {
        "rows_construction": len(raw),
        "rows_production": len(data.production_raw),
        "default_rate": float(raw[config.TARGET].mean()),
        "majority_class_accuracy": majority_class_accuracy(
            raw[config.TARGET]),
        "missing_pct": missing,
        "default_rate_by_income_missingness": missing_vs_default,
        "anomalies": eda.anomaly_report(raw),
        "psi": psi,
        "dictionary": data.dictionary,
    }


def run_model_selection(
    data: PreparedData, settings: RunSettings
) -> tuple[dict[str, Any], dict[str, CandidateResult], str]:
    """Compare candidates and evaluate the winner on the hold-out split.

    Args:
        data: Prepared data.
        settings: Runtime settings.

    Returns:
        Tuple ``(results, candidates, winner_name)``.
    """
    hist_gb_params = dict(config.HIST_GB_PARAMS)
    if settings.tune:
        hist_gb_params = tune_hist_gb(
            build_feature_matrix(data.train, config.CLEAN_FEATURES),
            data.y_train, settings)
    specs = build_candidate_specs(settings, hist_gb_params)
    table, candidates = evaluate_candidates(
        specs, data.train, data.y_train, data.valid, data.y_valid,
        data.scenarios, settings.cv_folds, settings.seed)
    winner = select_best(table)
    best = candidates[winner]
    logger.info("Selected model: %s", winner)
    write_table(table, settings.reports_dir / "model_comparison.csv")

    y_valid, holdout = data.y_valid, best.holdout_proba
    runner_up = str(table.index[1]) if len(table) > 1 else winner
    comparisons = {
        f"{winner} vs logistic": candidates["logistic"],
        f"{winner} vs {runner_up}": candidates[runner_up],
    }
    delong = {
        name: delong_test(y_valid, holdout, other.holdout_proba).__dict__
        for name, other in comparisons.items() if other is not best
    }

    policies, matrices, bootstrap, intervals = {}, {}, {}, {}
    for key, scenario in data.scenarios.items():
        threshold = best.scorer.thresholds[key]
        policies[key] = policy_summary(y_valid, holdout, threshold,
                                       scenario)
        policies[key]["theoretical_threshold"] = (
            scenario.theoretical_threshold)
        policies[key].update(trivial_policy_costs(y_valid, scenario))
        matrices[key] = confusion_matrix(
            y_valid, decisions_from_proba(holdout, threshold))
        bootstrap[key] = bootstrap_cost(
            y_valid, holdout, threshold, scenario,
            settings.bootstrap_samples, settings.seed)
        intervals[key] = confidence_interval(bootstrap[key])
        policies[key]["cost_ci_low"], policies[key]["cost_ci_high"] = (
            intervals[key])

    figures = settings.figures_dir
    labels = config.SCENARIO_LABELS
    roc_scores = {winner: holdout}
    roc_scores.update({name: candidates[name].holdout_proba
                       for name in ROC_MODELS
                       if name in candidates and name != winner})
    plots.plot_roc_pr(y_valid, roc_scores, figures / "08_roc_pr.png")
    plots.plot_calibration(y_valid, roc_scores,
                           figures / "09_calibration.png")
    plots.plot_cost_curves(
        cost_curve_frame(y_valid, holdout, data.scenarios, THRESHOLD_GRID),
        best.scorer.thresholds,
        {k: s.theoretical_threshold for k, s in data.scenarios.items()},
        labels, figures / "10_cost_curves.png")
    plots.plot_confusion_matrices(matrices, labels,
                                  figures / "11_confusion_matrices.png")
    lift = lift_table(y_valid, holdout)
    plots.plot_lift(lift, figures / "12_lift.png")
    plots.plot_bootstrap(bootstrap, intervals, labels,
                         figures / "13_cost_bootstrap.png")
    if "random_forest" in candidates:
        forest = candidates["random_forest"].scorer
        plots.plot_feature_importance(
            pd.Series(forest.estimator.feature_importances_,
                      index=forest.features),
            "Random forest impurity importance (professor's first model)",
            "Mean decrease in impurity", figures / "14_rf_importance.png")

    results = {
        "comparison": table,
        "winner": winner,
        "winner_description": best.spec.description,
        "policies_holdout": policies,
        "delong": delong,
        "lift": lift,
        "hist_gb_params": hist_gb_params,
    }
    return results, candidates, winner


def run_production(
    data: PreparedData,
    candidate: CandidateResult,
    settings: RunSettings,
) -> tuple[dict[str, Any], CreditScorer]:
    """Refit the winner on all data and write the two delivery files.

    Args:
        data: Prepared data.
        candidate: Winning candidate (its spec is refitted).
        settings: Runtime settings.

    Returns:
        Tuple ``(results, deployed_scorer)``.

    Raises:
        OSError: If a delivery file cannot be written.
    """
    scorer, _, threshold_results = fit_scorer(
        candidate.spec, data.construction, data.target, data.scenarios,
        settings.cv_folds, settings.seed)
    proba = scorer.predict_proba(data.production)
    results: dict[str, Any] = {"files": {}}
    for key in data.scenarios:
        decisions = decisions_from_proba(proba, scorer.thresholds[key])
        path = (settings.production_output_dir
                / config.SCENARIO_OUTPUT_FILENAMES[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            pd.DataFrame({config.PREDICTION_COLUMN: decisions}).to_csv(
                path, index=False)
        except OSError as exc:
            raise OSError(f"Could not write delivery file {path}") from exc
        results["files"][key] = {
            "path": path,
            "rows": len(decisions),
            "denials": int(decisions.sum()),
            "denial_rate": float(decisions.mean()),
            "threshold": scorer.thresholds[key],
            "oof_cost_full_data": threshold_results[key].cost,
            "holdout_denial_rate": float(
                (candidate.holdout_proba
                 >= candidate.scorer.thresholds[key]).mean()),
        }
        logger.info("%s: %d decisions, %.2f%% denied (threshold %.3f)",
                    path.name, len(decisions), 100 * decisions.mean(),
                    scorer.thresholds[key])
    return results, scorer


def run_bandit(
    data: PreparedData,
    policies_holdout: dict[str, dict[str, float]],
    settings: RunSettings,
) -> tuple[dict[str, Any], dict[str, BanditDecisionModel]]:
    """Reinforcement-learning approach: random benchmark vs linear bandit.

    Args:
        data: Prepared data.
        policies_holdout: Hold-out summaries of the supervised winner.
        settings: Runtime settings.

    Returns:
        Tuple ``(results, trained_bandits_per_scenario)``.
    """
    features = config.CLEAN_FEATURES
    encoder = ContextEncoder(features).fit(data.train)
    train_ctx = encoder.transform(data.train)
    valid_ctx = encoder.transform(data.valid)

    rows, histories, benchmarks, models = [], {}, {}, {}
    for key, scenario in data.scenarios.items():
        rng = np.random.default_rng(settings.seed)
        random_rewards = run_interactions(
            CreditEnvironment(valid_ctx, data.y_valid, scenario, rng),
            RandomPolicy(rng), settings.random_benchmark_steps, learn=False)
        benchmarks[key] = float(random_rewards.mean())

        agent = LinearThompsonSampling(train_ctx.shape[1], rng)
        histories[key] = run_interactions(
            CreditEnvironment(train_ctx, data.y_train, scenario, rng),
            agent, settings.bandit_steps, learn=True,
            show_progress=settings.show_progress,
            description=f"bandit {key}")
        models[key] = BanditDecisionModel(encoder, agent)
        # Exploitation only on validation (no update), as in class.
        greedy = agent.greedy_actions(valid_ctx)
        rows.append({
            "scenario": config.SCENARIO_LABELS[key],
            "random_policy_cost": -benchmarks[key],
            "bandit_online_training_cost": -float(histories[key].mean()),
            "bandit_validation_cost": expected_cost(
                data.y_valid, greedy, scenario),
            "bandit_denial_rate": float(greedy.mean()),
            "supervised_validation_cost": policies_holdout[key]["cost"],
            **trivial_policy_costs(data.y_valid, scenario),
        })
    plots.plot_bandit_learning(
        histories, benchmarks, config.SCENARIO_LABELS,
        settings.figures_dir / "15_bandit_learning.png")

    selection = _bandit_feature_selection(
        data, models[config.AUDIT_SCENARIO_KEY], settings)
    return {"comparison": pd.DataFrame(rows).set_index("scenario"),
            "feature_selection": selection}, models


def _bandit_feature_selection(
    data: PreparedData, model: BanditDecisionModel, settings: RunSettings
) -> dict[str, Any]:
    """Workshop exercise: retrain the bandit on surrogate-selected inputs.

    NOTE: added — required by the class exercise ("retrain the model using
    only the variables of the surrogate and check whether it is as good").

    Args:
        data: Prepared data.
        model: Bandit trained on all features (audited scenario).
        settings: Runtime settings.

    Returns:
        Selected features and validation cost with all / selected inputs.
    """
    scenario = data.scenarios[config.AUDIT_SCENARIO_KEY]
    names = list(model.features)
    decisions = model.predict(data.valid)
    result = surrogate_lib.fit_surrogate(
        data.valid[names], decisions, settings.seed,
        max_depth=config.SURROGATE_MAX_DEPTH)
    selected = surrogate_lib.features_used(result, names)
    full_cost = expected_cost(data.y_valid, decisions, scenario)
    if not selected or len(selected) == len(names):
        return {"selected_features": selected,
                "cost_all_features": full_cost,
                "cost_selected_features": full_cost}

    encoder = ContextEncoder(tuple(selected)).fit(data.train)
    rng = np.random.default_rng(settings.seed)
    agent = LinearThompsonSampling(len(selected) + 1, rng)
    run_interactions(
        CreditEnvironment(encoder.transform(data.train), data.y_train,
                          scenario, rng),
        agent, settings.bandit_steps, learn=True)
    reduced = agent.greedy_actions(encoder.transform(data.valid))
    return {
        "selected_features": selected,
        "cost_all_features": full_cost,
        "cost_selected_features": expected_cost(
            data.y_valid, reduced, scenario),
    }


def run_surrogates(
    data: PreparedData,
    scorer: CreditScorer,
    bandits: dict[str, BanditDecisionModel],
    settings: RunSettings,
) -> dict[str, Any]:
    """Surrogate rules of the deployed model and of the bandit.

    The deployed model is explained on the production clients it actually
    decided; the bandit on the validation clients.

    Args:
        data: Prepared data.
        scorer: Deployed scorer.
        bandits: Trained bandits per scenario.
        settings: Runtime settings.

    Returns:
        Fidelity, baseline and rules per black box and scenario.
    """
    figures, seed = settings.figures_dir, settings.seed
    matrix = scorer.feature_matrix(data.production)
    proba = scorer.proba_from_matrix(matrix)
    results: dict[str, Any] = {}
    curves = {}
    for key in data.scenarios:
        decisions = decisions_from_proba(proba, scorer.thresholds[key])
        fitted = surrogate_lib.fit_surrogate(
            matrix, decisions, seed, max_depth=config.SURROGATE_MAX_DEPTH)
        rules = surrogate_lib.extract_rules(fitted, matrix, decisions,
                                            config.ACTION_NAMES)
        results[f"supervised_{key}"] = _surrogate_summary(fitted, rules)
        plots.plot_surrogate_tree(
            fitted.tree, list(matrix.columns),
            f"Surrogate of the deployed model - "
            f"{config.SCENARIO_LABELS[key]}",
            figures / f"16_surrogate_supervised_{key}.png")
        if key == config.AUDIT_SCENARIO_KEY:
            curves["Deployed supervised model"] = (
                surrogate_lib.fidelity_curve(
                    matrix, decisions, config.SURROGATE_LEAF_GRID, seed))

    bandit = bandits[config.AUDIT_SCENARIO_KEY]
    inputs = data.valid[list(bandit.features)]
    decisions = bandit.predict(data.valid)
    fitted = surrogate_lib.fit_surrogate(
        inputs, decisions, seed, max_depth=config.SURROGATE_MAX_DEPTH)
    rules = surrogate_lib.extract_rules(fitted, inputs, decisions,
                                        config.ACTION_NAMES)
    results["bandit"] = _surrogate_summary(fitted, rules)
    curves["Contextual bandit"] = surrogate_lib.fidelity_curve(
        inputs, decisions, config.SURROGATE_LEAF_GRID, seed)
    plots.plot_surrogate_tree(fitted.tree, list(inputs.columns),
                              "Surrogate of the contextual bandit",
                              figures / "17_surrogate_bandit.png")
    plots.plot_fidelity_curve(curves, figures / "18_fidelity_tradeoff.png")
    results["fidelity_curves"] = curves
    return results


def _surrogate_summary(
    fitted: surrogate_lib.SurrogateResult,
    rules: list[surrogate_lib.SurrogateRule],
) -> dict[str, Any]:
    """Serializable summary of a surrogate."""
    return {
        "fidelity": fitted.fidelity,
        "majority_baseline": fitted.majority_baseline,
        "n_rules": fitted.n_leaves,
        "rules_text": surrogate_lib.format_rules(rules),
    }


def run_local_explanations(
    data: PreparedData, scorer: CreditScorer, settings: RunSettings
) -> dict[str, Any]:
    """SHAP (global and local), PDP/ALE, LIME and the denial letter.

    Args:
        data: Prepared data.
        scorer: Deployed scorer.
        settings: Runtime settings.

    Returns:
        SHAP importance, local cases, LIME table and denial letter.
    """
    figures, seed = settings.figures_dir, settings.seed
    threshold = scorer.thresholds[config.AUDIT_SCENARIO_KEY]
    sample = data.production.sample(
        min(settings.shap_sample_size, len(data.production)),
        random_state=seed)
    matrix = scorer.feature_matrix(sample)
    proba = scorer.proba_from_matrix(matrix)

    explanation = compute_shap_values(scorer, matrix, seed)
    importance = shap_importance(explanation)
    top = list(importance.index[:config.N_TOP_FEATURES])
    plots.plot_shap_beeswarm(explanation, figures / "19_shap_beeswarm.png")
    plots.plot_feature_importance(
        importance, "Global importance (mean |SHAP|)", "mean |SHAP value|",
        figures / "20_shap_importance.png")
    plots.plot_shap_dependence(explanation, top[:3],
                               figures / "21_shap_dependence.png")

    cases = select_local_cases(proba, threshold)
    local = {}
    for name, position in cases.items():
        plots.plot_shap_waterfall(
            explanation[position],
            f"{name.replace('_', ' ')} (p = {proba[position]:.3f}, "
            f"threshold = {threshold:.3f})",
            figures / f"22_shap_waterfall_{name}.png")
        local[name] = {
            "default_proba": float(proba[position]),
            "top_risk_drivers": top_risk_drivers(explanation, position),
        }

    dependence_rows = data.production.sample(
        min(settings.dependence_sample_size, len(data.production)),
        random_state=seed)
    dep_matrix = scorer.feature_matrix(dependence_rows)
    curves = {
        feature: (
            partial_dependence_curve(scorer.proba_from_matrix, dep_matrix,
                                     feature, config.DEPENDENCE_GRID_POINTS),
            accumulated_local_effects(scorer.proba_from_matrix, dep_matrix,
                                      feature,
                                      config.DEPENDENCE_GRID_POINTS),
        )
        for feature in top
    }
    plots.plot_pdp_ale(curves, figures / "23_pdp_vs_ale.png")

    denied_position = cases["clear_denial"]
    lime_table = explain_with_lime(
        scorer.proba_from_matrix, scorer.feature_matrix(data.train),
        matrix.iloc[denied_position], config.LIME_NUM_FEATURES, seed)

    letter_position = cases["borderline_denial"]
    letter, withheld = _denial_letter(
        data, scorer, sample.iloc[letter_position],
        matrix.iloc[letter_position],
        top_risk_drivers(explanation, letter_position), settings)
    return {
        "shap_importance": importance,
        "local_cases": local,
        "lime_clear_denial": lime_table,
        "shap_top_clear_denial": local["clear_denial"]["top_risk_drivers"],
        "denial_letter": letter,
        "withheld_drivers": withheld,
    }


def _counterfactual_search(
    data: PreparedData, scorer: CreditScorer, settings: RunSettings
) -> CounterfactualSearch:
    """Counterfactual search configured for the audited policy."""
    actionable = default_actionable_features()
    return CounterfactualSearch(
        scorer.predict_proba, actionable,
        robust_scales(data.train, [f.name for f in actionable]),
        scorer.thresholds[config.AUDIT_SCENARIO_KEY],
        settings.counterfactual_candidates, settings.seed)


def _denial_letter(
    data: PreparedData,
    scorer: CreditScorer,
    client_clean: pd.Series,
    client_features: pd.Series,
    drivers: pd.Series,
    settings: RunSettings,
) -> tuple[str, list[str]]:
    """Letter for one denied client, with a counterfactual if found."""
    search = _counterfactual_search(data, scorer, settings)
    result = search.generate(client_clean, 1)
    counterfactual = result.table.iloc[1] if result.found else None
    return build_denial_letter(client_features, drivers, counterfactual)


def run_counterfactuals(
    data: PreparedData, scorer: CreditScorer, settings: RunSettings
) -> list[dict[str, Any]]:
    """Counterfactuals for denied validation clients of both real classes.

    The brief asks for "several examples of real class 0 and 1": moderate
    denials (threshold <= p <= ``COUNTERFACTUAL_MAX_PROBA``) are picked
    because they are the ones where an actionable reversal is realistic.

    Args:
        data: Prepared data.
        scorer: Deployed scorer.
        settings: Runtime settings.

    Returns:
        One entry per client with its real class and counterfactual table.
    """
    threshold = scorer.thresholds[config.AUDIT_SCENARIO_KEY]
    proba = scorer.predict_proba(data.valid)
    moderate = (proba >= threshold) & (
        proba <= config.COUNTERFACTUAL_MAX_PROBA)
    rng = np.random.default_rng(settings.seed)
    search = _counterfactual_search(data, scorer, settings)
    outputs = []
    for real_class in (1, 0):
        pool = np.flatnonzero(moderate & (data.y_valid == real_class))
        chosen = rng.choice(pool, size=min(
            config.COUNTERFACTUAL_CLIENTS_PER_CLASS, len(pool)),
            replace=False)
        for position in chosen:
            result = search.generate(data.valid.iloc[position],
                                     config.COUNTERFACTUALS_PER_CLIENT)
            outputs.append({
                "validation_row": int(position),
                "real_class": real_class,
                "default_proba": result.original_proba,
                "found": result.found,
                "table": result.table,
            })
    return outputs


def run_audit_extras(
    data: PreparedData,
    candidate: CandidateResult,
    settings: RunSettings,
) -> dict[str, Any]:
    """Permutation importance, fairness by age and reject inference.

    These analyses need honest labels, so they use the winning
    configuration fitted on the training split and scored on validation.

    Args:
        data: Prepared data.
        candidate: Winning candidate (fitted on the training split).
        settings: Runtime settings.

    Returns:
        Permutation importance, fairness tables and reject-inference AUCs.
    """
    key = config.AUDIT_SCENARIO_KEY
    scorer = candidate.scorer
    matrix = scorer.feature_matrix(data.valid)
    permutation = permutation_auc_importance(
        scorer, matrix, data.y_valid, settings.permutation_repeats,
        settings.seed)
    plots.plot_feature_importance(
        permutation["mean"], "Permutation importance (validation)",
        "Drop in AUC when shuffled",
        settings.figures_dir / "24_permutation_importance.png")

    groups = fairness.age_groups(data.valid[config.AGE])
    with_age = fairness.group_metrics(
        data.y_valid, decisions_from_proba(candidate.holdout_proba,
                                           scorer.thresholds[key]), groups)
    no_age_scorer, _, _ = fit_scorer(
        candidate.spec.without_feature(config.AGE), data.train,
        data.y_train, data.scenarios, settings.cv_folds, settings.seed)
    no_age_proba = no_age_scorer.predict_proba(data.valid)
    without_age = fairness.group_metrics(
        data.y_valid,
        decisions_from_proba(no_age_proba, no_age_scorer.thresholds[key]),
        groups)
    plots.plot_fairness(with_age, without_age,
                        settings.figures_dir / "25_fairness_age.png")
    scenario = data.scenarios[key]
    cost_without_age = policy_summary(
        data.y_valid, no_age_proba, no_age_scorer.thresholds[key],
        scenario)["cost"]

    reject_inference = simulate_reject_inference(
        candidate.spec, data.train, data.y_train, data.valid, data.y_valid,
        config.REJECT_INFERENCE_ACCEPTANCE_SHARE, settings.seed)
    return {
        "permutation_importance": permutation[["mean", "std"]],
        "fairness_by_age_with_age": with_age,
        "fairness_by_age_without_age": without_age,
        "fairness_summary": pd.DataFrame({
            "with_age": fairness.fairness_summary(with_age),
            "without_age": fairness.fairness_summary(without_age),
        }),
        "cost_without_age": cost_without_age,
        "reject_inference": reject_inference,
    }


def run_pipeline(settings: RunSettings) -> dict[str, Any]:
    """Run every stage and write the reports.

    Args:
        settings: Runtime settings.

    Returns:
        Dictionary with the results of every stage.
    """
    set_global_seed(settings.seed)
    settings.figures_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_data(settings)
    results: dict[str, Any] = {"eda": run_eda(data, settings)}

    selection, candidates, winner = run_model_selection(data, settings)
    results["model_selection"] = selection
    production, scorer = run_production(data, candidates[winner], settings)
    results["production"] = production

    bandit_results, bandits = run_bandit(
        data, selection["policies_holdout"], settings)
    results["bandit"] = bandit_results
    results["surrogates"] = run_surrogates(data, scorer, bandits, settings)
    results["local_explanations"] = run_local_explanations(
        data, scorer, settings)
    results["counterfactuals"] = run_counterfactuals(data, scorer, settings)
    results["audit_extras"] = run_audit_extras(
        data, candidates[winner], settings)

    write_json(results, settings.reports_dir / "results.json")
    write_text(build_markdown_report(results, settings),
               settings.output_dir / "report.md")
    return results
