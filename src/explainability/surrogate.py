"""Global surrogate ("digital twin") of a black-box decision policy.

A shallow decision tree is trained to reproduce the *decisions* of the
black box (not the real outcome).  Its fidelity measures how faithful the
replica is; its leaves translate into human-readable rules.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.utils import config

LEAF = -1


@dataclass(frozen=True)
class SurrogateRule:
    """One root-to-leaf path of the surrogate tree.

    Attributes:
        conditions: Conjunction of split conditions.
        action: Decision predicted by the surrogate in the leaf.
        n_samples: Clients reaching the leaf.
        class_counts: Black-box decisions of those clients per action.
        purity: Share of the leaf's clients whose black-box decision
            matches ``action``.
        coverage: Share of all clients reaching the leaf.
    """

    conditions: tuple[str, ...]
    action: str
    n_samples: int
    class_counts: dict[str, int]
    purity: float
    coverage: float


@dataclass
class SurrogateResult:
    """A fitted surrogate and its quality indicators.

    Attributes:
        tree: Fitted decision tree.
        fidelity: Share of black-box decisions reproduced.
        majority_baseline: Fidelity of always predicting the most common
            black-box decision (the bar the surrogate must clear).
        n_leaves: Number of leaves (= number of rules).
    """

    tree: DecisionTreeClassifier
    fidelity: float
    majority_baseline: float
    n_leaves: int


def _majority_share(decisions: np.ndarray) -> float:
    """Share of the most frequent decision."""
    _, counts = np.unique(decisions, return_counts=True)
    return float(counts.max() / counts.sum())


def fit_surrogate(
    features: pd.DataFrame,
    decisions: np.ndarray,
    seed: int,
    max_depth: int | None = None,
    max_leaf_nodes: int | None = None,
) -> SurrogateResult:
    """Fit a decision tree that imitates black-box decisions.

    The tree handles missing values natively, so the surrogate sees the
    same inputs as the black box.

    Args:
        features: Inputs given to the black box (business units).
        decisions: Black-box decisions for those inputs.
        seed: Random seed of the tree.
        max_depth: Optional depth limit.
        max_leaf_nodes: Optional limit on the number of rules.

    Returns:
        A :class:`SurrogateResult`.

    Raises:
        ValueError: If inputs are misaligned or no limit is given.
    """
    decisions = np.asarray(decisions)
    if len(features) != len(decisions):
        raise ValueError("features and decisions must be aligned")
    if max_depth is None and max_leaf_nodes is None:
        raise ValueError("Set max_depth or max_leaf_nodes: an unbounded "
                         "surrogate is as opaque as the black box")
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=config.SURROGATE_MIN_SAMPLES_LEAF,
        random_state=seed,
    )
    tree.fit(features, decisions)
    return SurrogateResult(
        tree=tree,
        fidelity=float(tree.score(features, decisions)),
        majority_baseline=_majority_share(decisions),
        n_leaves=int(tree.get_n_leaves()),
    )


def fidelity_curve(
    features: pd.DataFrame,
    decisions: np.ndarray,
    leaf_grid: tuple[int, ...],
    seed: int,
) -> pd.DataFrame:
    """Fidelity as a function of surrogate size (explainability trade-off).

    The professor stressed that a perfect replica with hundreds of rules
    is no explanation: this curve shows where extra rules stop paying off.

    Args:
        features: Black-box inputs.
        decisions: Black-box decisions.
        leaf_grid: Values of ``max_leaf_nodes`` to try.
        seed: Random seed.

    Returns:
        DataFrame indexed by ``max_leaf_nodes`` with fidelity, actual
        number of leaves and majority baseline.
    """
    rows = []
    for n_leaves in leaf_grid:
        result = fit_surrogate(features, decisions, seed,
                               max_leaf_nodes=n_leaves)
        rows.append({
            "max_leaf_nodes": n_leaves,
            "fidelity": result.fidelity,
            "n_leaves": result.n_leaves,
            "majority_baseline": result.majority_baseline,
        })
    return pd.DataFrame(rows).set_index("max_leaf_nodes")


@dataclass(frozen=True)
class _Split:
    """One raw split condition on a root-to-leaf path."""

    feature: str
    upper: bool          # True for "<= value", False for "> value"
    value: float
    includes_missing: bool


def _simplify(path: tuple[_Split, ...]) -> tuple[str, ...]:
    """Merge repeated splits on a feature into its tightest interval.

    ``x <= 1.5 AND x <= 0.5`` becomes ``x <= 0.5``; a lower and an upper
    bound become ``0.5 < x <= 2.5``.  Missing values satisfy the merged
    condition only if they satisfy every original split.
    """
    conditions = []
    for feature in dict.fromkeys(split.feature for split in path):
        splits = [s for s in path if s.feature == feature]
        uppers = [s.value for s in splits if s.upper]
        lowers = [s.value for s in splits if not s.upper]
        missing = all(s.includes_missing for s in splits)
        if uppers and lowers:
            text = (f"{max(lowers):.4g} < {feature} <= "
                    f"{min(uppers):.4g}")
        elif uppers:
            text = f"{feature} <= {min(uppers):.4g}"
        else:
            text = f"{feature} > {max(lowers):.4g}"
        conditions.append(text + (" (or missing)" if missing else ""))
    return tuple(conditions)


def extract_rules(
    result: SurrogateResult,
    features: pd.DataFrame,
    decisions: np.ndarray,
    action_names: dict[int, str],
) -> list[SurrogateRule]:
    """Convert every leaf of the surrogate into a rule with statistics.

    Equivalent to ``get_rules_from_tree`` used in class: each rule lists
    its conditions and how many clients of each black-box decision reach
    it.  Conditions state explicitly where missing values go.

    Args:
        result: Fitted surrogate.
        features: Data the statistics are computed on.
        decisions: Black-box decisions for ``features``.
        action_names: Readable name of each decision value.

    Returns:
        Rules sorted by coverage (largest first).
    """
    tree = result.tree.tree_
    names = list(features.columns)
    has_missing = features.isna().any().to_dict()
    decisions = np.asarray(decisions)
    leaf_of_row = result.tree.apply(features)
    classes = result.tree.classes_
    missing_left = getattr(tree, "missing_go_to_left",
                           np.zeros(tree.node_count, dtype=bool))

    rules: list[SurrogateRule] = []

    def visit(node: int, path: tuple[_Split, ...]) -> None:
        left, right = tree.children_left[node], tree.children_right[node]
        if left == LEAF:
            in_leaf = leaf_of_row == node
            counts = {
                action_names[int(c)]: int((decisions[in_leaf] == c).sum())
                for c in classes
            }
            predicted = int(classes[np.argmax(tree.value[node][0])])
            n_samples = int(in_leaf.sum())
            action = action_names[predicted]
            rules.append(SurrogateRule(
                conditions=_simplify(path) or ("(all clients)",),
                action=action,
                n_samples=n_samples,
                class_counts=counts,
                purity=counts[action] / n_samples if n_samples else 0.0,
                coverage=n_samples / len(decisions),
            ))
            return
        name = names[tree.feature[node]]
        value = float(tree.threshold[node])
        can_be_missing = bool(has_missing.get(name, False))
        goes_left = bool(missing_left[node])
        visit(left, path + (
            _Split(name, True, value, can_be_missing and goes_left),))
        visit(right, path + (
            _Split(name, False, value, can_be_missing and not goes_left),))

    visit(0, ())
    return sorted(rules, key=lambda rule: rule.coverage, reverse=True)


def format_rules(rules: list[SurrogateRule]) -> str:
    """Render rules as plain text for reports and denial letters.

    Args:
        rules: Output of :func:`extract_rules`.

    Returns:
        Multi-line string, one block per rule.
    """
    blocks = []
    for i, rule in enumerate(rules, start=1):
        counts = ", ".join(f"{k}={v}" for k, v in rule.class_counts.items())
        blocks.append(
            f"Rule {i}: IF " + " AND ".join(rule.conditions)
            + f" THEN {rule.action.upper()}\n"
            f"    clients={rule.n_samples} ({rule.coverage:.1%}), "
            f"black-box decisions: {counts}, purity={rule.purity:.1%}"
        )
    return "\n".join(blocks)


def features_used(result: SurrogateResult, names: list[str]) -> list[str]:
    """Features used by the surrogate, most important first.

    Used for the XAI-driven feature selection exercise of the workshop.

    Args:
        result: Fitted surrogate.
        names: Feature names in model order.

    Returns:
        Names of features with non-zero importance in the surrogate.
    """
    importance = pd.Series(result.tree.feature_importances_, index=names)
    return list(importance[importance > 0].sort_values(
        ascending=False).index)
