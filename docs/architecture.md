# Architecture

## 1. Project overview

The project builds, optimises and audits a credit-granting model on the
"Give Me Some Credit" data set (`SeriousDlqin2yrs` target). A single
probabilistic model is turned into two decision policies by cost-optimal
thresholds (FP:FN = 1:1 and 1:10), the decisions for the 45,000 production
clients are written to `data/cs_produccion1.csv` / `data/cs_produccion2.csv`,
and the deployed model is audited with XAI techniques (surrogate rules,
SHAP, counterfactuals, PDP/ALE, LIME, fairness), next to a contextual
multi-armed bandit that approaches the same problem through reinforcement
learning.

## 2. Module dependencies

```mermaid
graph TD
    MAIN[main.py] --> PIPE[src/pipeline.py]
    MAIN --> CFG[src/utils/config.py]
    MAIN --> LOG[src/utils/logging_config.py]

    PIPE --> DATA[src/data<br/>loader · cleaning · features · eda]
    PIPE --> MODELS[src/models<br/>supervised · scorer · bandit]
    PIPE --> EVAL[src/evaluation<br/>costs · metrics · model_selection<br/>fairness · selection_bias]
    PIPE --> XAI[src/explainability<br/>surrogate · shap · counterfactuals<br/>dependence · lime · denial_report]
    PIPE --> VIZ[src/visualization<br/>plots · report]
    PIPE --> UTIL[src/utils<br/>reporting · reproducibility]

    MODELS --> DATA
    EVAL --> MODELS
    EVAL --> DATA
    XAI --> MODELS
    VIZ --> UTIL
    DATA --> CFG
    MODELS --> CFG
    EVAL --> CFG
    XAI --> CFG
```

## 3. Data flow

```mermaid
flowchart LR
    RAW[(cs_construccion.csv<br/>105k labelled)] --> DROP[drop age = 0<br/>training only]
    DROP --> FIT[CreditDataCleaner.fit<br/>caps from construction]
    PROD[(cs_produccion.csv<br/>45k unlabelled)] --> CLEANP[transform<br/>no row ever dropped]
    FIT --> CLEANC[transform<br/>flags · 96/98 codes · caps]
    FIT -.caps.-> CLEANP
    CLEANC --> SPLIT{stratified<br/>80 / 20}
    SPLIT -->|train| OOF[out-of-fold scores<br/>per candidate]
    OOF --> THR[exact cost-optimal<br/>threshold per scenario]
    THR --> SEL[select lowest<br/>normalised OOF cost]
    SPLIT -->|validation| HOLD[unbiased hold-out cost<br/>DeLong · bootstrap · fairness]
    SEL --> HOLD
    SEL --> REFIT[refit on all rows<br/>+ OOF thresholds]
    REFIT --> SCORER[CreditScorer]
    CLEANP --> SCORER
    SCORER --> OUT1[(cs_produccion1.csv)]
    SCORER --> OUT2[(cs_produccion2.csv)]
    SCORER --> AUDIT[XAI audit on production<br/>surrogate · SHAP · PDP/ALE · LIME]
    SCORER --> CF[counterfactuals + denial letter]
    SPLIT -->|train| BANDIT[CreditEnvironment +<br/>Linear Thompson sampling]
    BANDIT --> BSURR[surrogate of the bandit<br/>+ feature-selection exercise]
    AUDIT --> REPORT[(outputs/report.md<br/>results.json · figures)]
    HOLD --> REPORT
    BSURR --> REPORT
    CF --> REPORT
```

## 4. Main classes

```mermaid
classDiagram
    class RunSettings {
        +Path data_dir
        +Path output_dir
        +int seed
        +int cv_folds
        +int bandit_steps
        +quick() RunSettings
    }
    class CreditDataCleaner {
        +dict caps
        +fit(frame) CreditDataCleaner
        +transform(frame) DataFrame
    }
    class ModelSpec {
        +str name
        +builder()
        +tuple features
        +bool class_weighted
        +without_feature(name) ModelSpec
    }
    class CreditScorer {
        +ModelSpec spec
        +estimator
        +dict thresholds
        +predict_proba(clean) ndarray
        +decide(clean, scenario) ndarray
    }
    class CostScenario {
        +float cost_fp
        +float cost_fn
        +theoretical_threshold float
    }
    class CreditEnvironment {
        +new_client()
        +client_context() ndarray
        +act(action) float
    }
    class LinearThompsonSampling {
        +select(context) int
        +update(context, action, reward)
        +greedy_actions(contexts) ndarray
    }
    class RandomPolicy
    class BanditDecisionModel {
        +predict(frame) ndarray
    }
    class SurrogateResult {
        +tree
        +float fidelity
        +float majority_baseline
    }
    class CounterfactualSearch {
        +generate(client, n) CounterfactualResult
    }
    CreditScorer --> ModelSpec
    CreditEnvironment --> CostScenario
    BanditDecisionModel --> LinearThompsonSampling
    CounterfactualSearch ..> CreditScorer : score_fn
```

## 5. Key design decisions

| Decision | Why |
|---|---|
| Cleaning never drops production rows; missing income and 96/98 codes become flags | Every production client needs a decision; missingness is informative (5.5% vs 7.0% default rate). The professor dropped these rows in class, which is fine for training only. |
| Thresholds from out-of-fold predictions of the *same* model | The original submission applied a HistGB threshold to an XGBoost model and tuned it on the validation split used to report cost. |
| Model chosen on OOF cost, validation used once | Avoids the optimistic bias of selecting and reporting on the same split. |
| XAI run on the deployed model and on production clients | The audited model must be the one that produced the delivered decisions. |
| Label-dependent analyses (permutation, fairness, bootstrap) on the hold-out model | They need honest, unseen labels. |
| NumPy Thompson-sampling bandit | Same algorithm as `space_bandits.LinearBandits` used in class, without an unmaintained dependency downloaded from Google Drive. |
| Own counterfactual search | Keeps `DebtRatio` consistent with income changes and rebuilds derived features, which DiCE did not. |

## 6. How to run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python main.py                  # ~4 min, regenerates data/cs_produccion{1,2}.csv
pytest                          # unit + integration tests
```

Outputs: `outputs/report.md`, `outputs/reports/results.json`,
`outputs/reports/*.csv`, `outputs/figures/*.png`, `outputs/run.log`.
