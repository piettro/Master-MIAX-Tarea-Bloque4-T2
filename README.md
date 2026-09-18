# Explainable Credit Scoring under Asymmetric Costs

> Master's in AI & Quantum Computing Applied to Financial Markets (MIAX)
> Course: Advanced Artificial Intelligence - XAI workshop | Assignment: B4-T2 (XAI)

## Overview

For every applicant, the model decides whether to **grant** or **deny** credit,
under two cost scenarios from the assignment brief:

| Scenario | Cost FP (good client denied) | Cost FN (defaulter approved) | Delivery file |
|---|:-:|:-:|---|
| 1 | 1 | 1 | `data/cs_produccion1.csv` |
| 2 | 1 | 10 | `data/cs_produccion2.csv` |

The target is `SeriousDlqin2yrs` (1 = serious default within two years).
Convention: **prediction 1 = deny**, **0 = approve**. The graded metric is the
mean cost per client, `(c_FP·FP + c_FN·FN) / n`.

The project covers:

* **EDA and data quality.** Informative missingness, 96/98 administrative codes, heavy tails, invalid ages, and population stability (PSI) between construction and production.
* **Supervised models.** Logistic baseline, a single decision tree, random forest, HistGradientBoosting (plain, with derived features, monotonic, class-weighted) and XGBoost. The model is chosen by **out-of-fold cost**.
* **Cost-optimal thresholds.** An exact sweep over every score, compared against the Bayes threshold `c_FP / (c_FP + c_FN)`. Each threshold is always estimated for the same model that uses it.
* **Reinforcement-learning approach.** The credit problem is framed as a game (`CreditEnvironment`), with a random-policy benchmark and a linear contextual bandit (Thompson sampling).
* **XAI audit of the deployed model:**
  * surrogate trees with readable rules and a fidelity-vs-size trade-off curve (also for the bandit)
  * SHAP (global and local)
  * actionable counterfactuals for real class 0 and class 1 clients
  * an automatic denial letter
  * PDP vs ALE
  * LIME
  * permutation importance
  * fairness by age (80% rule, equal opportunity)
  * a bootstrap confidence interval of the cost
  * a selection-bias / reject-inference simulation

## Project Structure

```text
.
├── main.py                        # single entry point (CLI)
├── src/
│   ├── pipeline.py                # stage orchestration
│   ├── data/                      # loader, cleaning, features, eda
│   ├── models/                    # supervised catalogue, CreditScorer, bandit
│   ├── evaluation/                # costs/thresholds, metrics, model selection,
│   │                              # fairness, selection bias
│   ├── explainability/            # surrogate, shap, counterfactuals, PDP/ALE,
│   │                              # LIME, denial letter
│   ├── visualization/             # figures and Markdown report
│   └── utils/                     # config, logging, reproducibility, reporting
├── tests/                         # pytest unit + end-to-end tests
├── docs/architecture.md           # Mermaid diagrams and design decisions
├── data/                          # inputs + the two delivery files
├── notebooks/                     # original (Spanish) submission, for reference
├── professor_solution/            # in-class notebook of the professor
├── materials/                     # brief and class transcripts
├── requirements.txt               # pinned runtime + test dependencies
└── requirements-notebook.txt      # extras to re-run the legacy notebook
```

## Setup & Installation

Requires Python 3.10+.

```bash
# 1. Clone the repo
git clone https://github.com/raulrodriguezlr/B4-T2-XAI
cd B4-T2-XAI

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

The input files (`cs_construccion.csv`, `cs_produccion.csv`,
`DataDictionary.csv`) are in `data/`. If one is missing, the program stops with
a message that says which file to put where.

## Usage

```bash
python main.py
```

A full run takes about 4 minutes on a laptop. It regenerates
`data/cs_produccion1.csv` and `data/cs_produccion2.csv`, which hold 45,000
rows each in a single `prediccion` column. It also writes the following to
`outputs/`:

* `report.md`: every table, the surrogate rules, the counterfactuals, the denial letter and the reflection
* `reports/results.json`, `reports/*.csv`: machine-readable results
* `figures/*.png`: 28 figures (EDA, ROC/PR, calibration, cost curves, bandit, surrogates, SHAP, PDP/ALE, fairness, …)
* `run.log`

| Option | Effect |
|---|---|
| `--quick` | fewer folds and smaller samples (smoke run, ~2 min) |
| `--tune` | reruns the HistGB randomised search instead of using the stored best parameters |
| `--no-xgboost` | drops XGBoost from the candidates |
| `--seed N` | global seed (default 42) |
| `--data-dir`, `--output-dir`, `--production-output-dir` | redirect inputs and outputs |
| `--no-progress`, `--log-level` | console verbosity |

All constants, hyper-parameters and paths live in `src/utils/config.py`.

```bash
pytest                    # 48 tests, including an end-to-end run on synthetic data
flake8 src/ main.py tests/
```

## Results

Seed 42, validation split never used for selection:

| | Scenario 1 (1:1) | Scenario 2 (1:10) |
|---|:-:|:-:|
| Selected model | HistGB + derived features | same model |
| Bayes threshold | 0.500 | 0.091 |
| Deployed threshold (OOF, full history) | 0.522 | 0.091 |
| Hold-out cost per client (95% CI) | 0.0627 [0.059, 0.066] | 0.329 [0.309, 0.349] |
| Approve-all cost (reference) | 0.0669 | 0.669 |
| Production denial rate | 1.85% | 18.6% |

* **Model accuracy.** Hold-out AUC is 0.872, against 0.855 for the logistic baseline (DeLong p < 0.001). The gap to the random forest (0.870) is not significant.
* **Stability.** PSI is below 0.001 for every variable, so construction and production come from the same population.
* **Bandit.** In scenario 2 the bandit costs 0.341, against 0.787 for the random policy and 0.329 for the supervised model.
* **Surrogate.** An 8-rule tree reproduces 94.4% of the scenario-2 decisions, well above the 81.4% majority baseline. The dominant drivers are the number of past-due payments and credit-line utilisation. SHAP, permutation importance and LIME agree on this.
* **Fairness.** Age-band disparate impact is 0.67, below the 0.80 reference. Removing age raises it to 0.76 at a cost of +0.002 per client.

See `outputs/report.md` after a run for the complete, data-driven report.

## Architecture

See [docs/architecture.md](docs/architecture.md) for diagrams.

## References

- Professor's solution: `professor_solution/`
- Assignment materials: `materials/`
- Original submission (Spanish notebook): `notebooks/credit_xai_original_submission.ipynb`
