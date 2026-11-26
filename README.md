# Groundwater Predictive Analytics — SML AAT

Course: Statistical Modeling (24AM5PCSML)  
Study: District-level groundwater stress in India using CGWB 2025 assessment data  
SDG alignment: SDG 6 (Target 6.4), with relevance to SDG 13

This repository contains a two-phase statistical modelling workflow. Phase 1 performs classical regression and estimator validation; Phase 2 compares regularised regression and machine-learning regressors.

## Repository structure

- data/ — add official CGWB input CSVs (see data/README.md)
- phase1_regression.py — cleaning, EDA, OLS, diagnostics, bootstrap and CV
- phase2_model_selection.py — nested CV, tuning, model comparison and screening
- outputs/ — generated locally by the scripts
- requirements.txt — Python dependencies
- RESULTS_SUMMARY.md — metrics transcribed from the supplied run

## Data required

Place these files in data/:

| File | Required? | Purpose |
|---|---|---|
| district_gw_resources_2025_annexure_II.csv | Yes | Main district-level CGWB table |
| 07_state_rainfall_gwy_2024_25.csv | Yes | State rainfall join / sensitivity analysis |
| groundwater_district_2024.csv | No | 2024 robustness replication |
| state_gw_resources_2025.csv | No | State-versus-district reconciliation |

Data availability note: the supplied ZIP contained scripts and generated outputs but not the raw CGWB source CSVs. The cleaned dataset in the ZIP was a Phase 1 output, not a substitute for original source files. Raw inputs are not fabricated or silently committed. See data/README.md.

## Setup and execution

Python 3.10+ is recommended.

    python -m venv .venv
    # Windows PowerShell: .venv\\Scripts\\Activate.ps1
    # macOS/Linux: source .venv/bin/activate
    pip install -r requirements.txt
    python phase1_regression.py
    python phase2_model_selection.py

Phase 1 writes outputs/phase1/cleaned_dataset.csv, which Phase 2 consumes. Run Phase 1 first. Full nested tuning in Phase 2 can take several minutes.

Quick smoke test:

    # Windows PowerShell
    $env:AAT_QUICK="1"; python phase2_model_selection.py
    # macOS/Linux
    AAT_QUICK=1 python phase2_model_selection.py

Optional CPU control: AAT_JOBS=-1 uses all available cores for grid search; set a positive integer to limit parallel workers.

## Method overview

### Phase 1 — classical regression and estimator validation
- Validates and cleans the CGWB district table; records each cleaning step.
- Derives recharge-composition predictors and log-transforms the extraction-stage target.
- Performs EDA, simple and multiple OLS regression.
- Checks linearity, residual normality, heteroskedasticity, multicollinearity and influential observations.
- Uses HC3/state-clustered inference, pairs and residual bootstrap, and random/state-grouped cross-validation.
- Runs sensitivity checks and a 2024 replication when optional data is available.

### Phase 2 — model selection and comparative evaluation
- Compares OLS, stepwise AIC/BIC, LASSO and Ridge with Decision Tree, Random Forest, XGBoost and RBF-SVR.
- Uses nested cross-validation so feature selection and hyperparameter tuning occur inside training folds.
- Evaluates random 10-fold CV, state-grouped CV and an untouched 20% holdout.
- Reports RMSE, MAE, R², adjusted R², information criteria where applicable, paired tests and screening metrics.
- Applies a documented one-standard-error/interpretability rule to select a final model.

## Important modelling caveats
- Target: log_stage = ln(stage_of_extraction_pct).
- Extraction and extractable-resource variables are excluded from predictors to avoid direct target leakage.
- State-level rainfall is treated as sensitivity analysis because it is constant within each state and can act as a state identifier.
- Recharge scale and natural-discharge ratio are mathematically related to extractable groundwater availability. Their predictive contribution is partly structural; results must not be interpreted as causal effects.
- OLS residual diagnostics indicate heteroskedasticity and non-normality. Consider robust/clustered inference and bootstrap estimates alongside classical standard errors.
- Predictions are intended for exploratory prioritisation, not allocation/enforcement decisions. Confirm with field measurements.

## Existing run results
See RESULTS_SUMMARY.md for the main metrics transcribed from the uploaded output archive. These are prior-run results, not a fresh execution during repository setup.

## License and data provenance
No license or raw-data redistribution permission was supplied with the archive. Add an appropriate project license and cite/retain the official CGWB data source and its terms before redistribution.
