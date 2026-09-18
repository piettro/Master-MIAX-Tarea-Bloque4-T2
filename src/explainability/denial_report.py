"""Customer-facing explanation of a credit denial.

Combines three XAI layers: the client's own risk drivers (SHAP), what the
client could change (counterfactual) and the uniform policy applied to
everyone (surrogate rules).
"""
from __future__ import annotations

import pandas as pd

from src.explainability.counterfactuals import CHANGED_COLUMN
from src.utils import config

# NOTE: improved over the original submission — the letter used to
# describe ``age`` as "the length of your credit history", which is both
# inaccurate and a way of disguising a protected attribute.  Age is now
# excluded from customer-facing reasons and reported only internally.
REASON_TEMPLATES: dict[str, str] = {
    config.REVOLVING_UTILIZATION:
        "the use of your credit cards and credit lines "
        "({value:.0%} of the available limit)",
    config.PAST_DUE_90:
        "payments more than 90 days late in your history ({value:.0f})",
    config.PAST_DUE_30_59:
        "payments 30-59 days late in the last two years ({value:.0f})",
    config.PAST_DUE_60_89:
        "payments 60-89 days late in the last two years ({value:.0f})",
    config.TOTAL_PAST_DUE:
        "the total number of late payments recorded ({value:.0f})",
    config.DEBT_RATIO:
        "the share of your income committed to debt payments "
        "({value:.0%})",
    config.MONTHLY_DEBT_AMOUNT:
        "the amount of your monthly debt payments ({value:,.0f})",
    config.MONTHLY_INCOME: "your documented monthly income ({value:,.0f})",
    config.INCOME_PER_DEPENDENT:
        "your income per dependent person ({value:,.0f})",
    config.OPEN_CREDIT_LINES:
        "the number of open loans and credit lines ({value:.0f})",
    config.REAL_ESTATE_LINES:
        "the number of real-estate loans ({value:.0f})",
    config.REAL_ESTATE_SHARE:
        "the share of real-estate loans among your credit lines",
    config.DEPENDENTS: "the number of dependents ({value:.0f})",
    config.MISSING_INCOME_FLAG: "the lack of documented income",
    config.SPECIAL_CODE_FLAG:
        "a special situation recorded in your payment history",
}
ACTION_TEMPLATES: dict[str, str] = {
    config.REVOLVING_UTILIZATION:
        "reduce the use of your credit lines to about {value:.0%} "
        "of the limit",
    config.DEBT_RATIO:
        "reduce the share of income spent on debt to about {value:.0%}",
    config.MONTHLY_INCOME:
        "document a monthly income of about {value:,.0f}",
    config.OPEN_CREDIT_LINES:
        "reduce your open credit lines to {value:.0f}",
}
COUNT_FEATURES: tuple[str, ...] = (
    *config.DELINQUENCY_COLUMNS,
    config.TOTAL_PAST_DUE,
    config.SPECIAL_CODE_FLAG,
    config.MISSING_INCOME_FLAG,
)
GENERIC_ADVICE = (
    "Keeping your payments up to date reduces the weight of past "
    "incidents over time."
)
LINE_WIDTH = 72


def _describe(feature: str, value: float) -> str:
    """Plain-language description of one risk driver."""
    template = REASON_TEMPLATES.get(feature, feature)
    try:
        return template.format(value=value)
    except (ValueError, TypeError):
        return template.split(" ({")[0]


def build_denial_letter(
    client: pd.Series,
    risk_drivers: pd.Series,
    counterfactual: pd.Series | None = None,
    max_reasons: int = 3,
) -> tuple[str, list[str]]:
    """Write the letter sent to a denied client.

    The score and the internal threshold are deliberately not disclosed
    (they would enable gaming of the system).

    Args:
        client: Model inputs of the client (feature space).
        risk_drivers: Positive SHAP contributions, largest first.
        counterfactual: Optional counterfactual row (output of
            ``CounterfactualSearch``) that would lead to approval; only
            the variables listed in its ``changed`` field are quoted.
        max_reasons: Maximum number of reasons quoted.

    Returns:
        Tuple ``(letter, withheld)`` where ``withheld`` lists drivers that
        were not communicated (protected / non-actionable attributes) and
        must be reviewed internally.
    """
    withheld = [f for f in risk_drivers.index
                if f in config.NON_COMMUNICABLE_FEATURES]
    # A zero count ("0 late payments") is never a meaningful reason for
    # a client, even when SHAP attributes risk to it via interactions.
    reasons = [f for f in risk_drivers.index
               if f not in config.NON_COMMUNICABLE_FEATURES
               and not (f in COUNT_FEATURES and client.get(f) == 0)]
    reasons = reasons[:max_reasons]

    lines = [
        "=" * LINE_WIDTH,
        "CREDIT DECISION - EXPLANATION OF THE DENIAL",
        "=" * LINE_WIDTH,
        "Your application was assessed with the same objective criteria",
        "applied to every applicant. The main factors were:",
        "",
    ]
    lines += [f"  {i}. {_describe(f, client[f]).capitalize()}."
              for i, f in enumerate(reasons, start=1)]
    lines += ["", "What you could do for a future application:"]
    if counterfactual is not None:
        changed = str(counterfactual.get(CHANGED_COLUMN, "")).split(",")
        for feature in changed:
            if feature in ACTION_TEMPLATES:
                lines.append("  - " + ACTION_TEMPLATES[feature].format(
                    value=counterfactual[feature]))
        lines.append("    With these changes the application would be "
                     "approved under current criteria.")
    lines += [f"  - {GENERIC_ADVICE}", "",
              "You may request a review of this decision by an analyst.",
              "=" * LINE_WIDTH]
    return "\n".join(lines), withheld
