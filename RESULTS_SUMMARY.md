# Existing execution results

These values are transcribed from the supplied ZIP. They are previous-run results, not a fresh execution performed by this repository commit.

## Phase 1
- Observations: 706 districts across 36 states/UTs
- Bootstrap replications: 2,000
- OLS MLR R²: 0.2122; adjusted R²: 0.2089
- OLS MLR AIC: 2007.6625; BIC: 2025.9010
- Random 10-fold CV RMSE (log stage): 0.9949
- State-grouped CV RMSE (log stage): 0.9628
- Breusch–Pagan p-value: 0.0003815; White test p-value: 8.94e-9
- Jarque–Bera p-value: 5.80e-198

The diagnostics indicate heteroskedastic and non-normal residuals; robust/clustered inference and bootstrap results are reported alongside classical OLS results.

## Phase 2 comparison

| Model | 10-fold CV RMSE (log) | CV R² | Holdout RMSE (log) | Holdout R² | State-grouped RMSE (log) |
|---|---:|---:|---:|---:|---:|
| Phase 1 OLS MLR | 0.9949 | 0.1966 | 1.1891 | 0.0599 | 0.9628 |
| Decision Tree | 0.8841 | 0.3560 | 1.0906 | 0.2092 | 0.9026 |
| Random Forest | 0.8034 | 0.4753 | 0.9763 | 0.3663 | 0.8510 |
| XGBoost | 0.7880 | 0.4953 | 0.9469 | 0.4039 | 0.8531 |
| SVR (RBF) | 0.7866 | 0.5003 | 0.9589 | 0.3887 | 0.8683 |

The one-standard-error rule selected Random Forest as the most interpretable among the competitive models (Random Forest, XGBoost and SVR). It beat the Phase 1 OLS baseline on the untouched holdout and state-grouped CV. Its corrected paired-test p-value was 0.00030.

## Screening
- Random Forest out-of-fold AUC for over-exploited districts: 0.7006
- Flagging the top 30% of districts by predicted stage gave 55.32% recall (52 of 94 over-exploited districts).

## Interpretation and limitations
- State-mean baseline performs strongly under random splits (CV RMSE 0.5513), showing that state identity explains substantial variation.
- State-mean baseline cannot predict unseen states and falls back to the overall mean in grouped validation.
- log_total_recharge and natural_discharge_ratio partly reconstruct the extractable-resource denominator. The model is predictive/associational, not causal.
- Predictions are for prioritisation only; confirm with field measurements before operational decisions.

## Reproducibility status
The supplied ZIP contains Phase 1/Phase 2 output folders and a separate reference_outputs copy. The raw CGWB input CSVs were not included. The summary above preserves the key reported metrics; the full generated output directories and binary final_model.joblib are not included in this initial GitHub upload. To regenerate all artifacts, add the official source CSVs under data/, then run Phase 1 followed by Phase 2.
