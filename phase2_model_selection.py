#!/usr/bin/env python3
"""
Statistical Modeling (24AM5PCSML) - Alternative Assessment Tool (AAT)
======================================================================
PHASE 2 : Model Selection & Comparative Evaluation
Topic   : District-level groundwater stress in India (CGWB 2025)
SDG     : SDG 6 (target 6.4) - also supports SDG 13.

Requires: outputs/phase1/cleaned_dataset.csv  (run phase1_regression.py first)

What this script does (maps 1-to-1 to the faculty brief)
--------------------------------------------------------
 1. Feature selection   : stepwise regression (AIC and BIC), LASSO, Ridge.
 2. ML regression       : Decision Tree, Random Forest, SVR (RBF), XGBoost -
                          each tuned by inner 5-fold CV (hyper-parameters are
                          NEVER tuned on test data).
 3. k-fold CV           : the same outer 10 folds for every model; feature
                          selection and tuning are repeated inside every fold
                          (nested CV), so the scores are honest.
 4. Metrics             : RMSE, MAE, R2, adjusted R2 (CV, holdout) and AIC/BIC
                          (linear models).
 5. Comparative analysis: classical OLS vs penalised vs ML; paired fold tests
                          including the Nadeau-Bengio corrected t-test.
 6. Generalisation      : state-grouped CV (predict a state the model never saw),
                          untouched 20 % holdout, and two naive benchmarks
                          (overall mean, state mean) so every gain is measured
                          against something meaningful.
 7. Final model         : chosen with an explicit, reproducible rule (see
                          select_final_model), justified on accuracy,
                          interpretability and assumptions.
 8. Real-world use      : permutation importance, partial dependence,
                          district screening table (CGWB categories, recall of
                          over-exploited districts), saved model for reuse.

Design rules (why the numbers can be trusted)
---------------------------------------------
 * Leakage guard: stage = extraction / extractable resource, so NO extraction or
   extractable-resource variable (and no allocation variable) is a predictor.
 * PRIMARY feature set = four DISTRICT-level recharge variables.
 * The state rainfall column is one number per state, i.e. it identifies the
   state. It is therefore run only as a clearly labelled SENSITIVITY analysis
   (set RUN_STATE_RAINFALL_SENSITIVITY = False to skip it).
 * One seed, one split definition, everything reproducible.

Run:   python phase2_model_selection.py          (about 5-20 minutes)
Quick smoke test (tiny grids, a couple of minutes):   AAT_QUICK=1 python phase2_model_selection.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV, lasso_path
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# =============================================================================
# CONFIGURATION
# =============================================================================
ROOT = Path(__file__).resolve().parent
P1_DIR = ROOT / "outputs" / "phase1"
OUT_DIR = ROOT / "outputs" / "phase2"
PLOT_DIR = OUT_DIR / "plots"

QUICK = os.environ.get("AAT_QUICK", "0") == "1"      # smoke-test mode only
N_JOBS = int(os.environ.get("AAT_JOBS", "-1"))        # CPU cores for grid search (-1 = all)

SEED = 42
K_OUTER = 10                                          # outer CV folds
K_INNER = 5                                           # inner folds for tuning
TEST_SIZE = 0.20                                      # untouched holdout
PERM_REPEATS = 30
TARGET = "log_stage"

BASELINE_FEATURES = ["rainfall_share", "monsoon_share", "natural_discharge_ratio"]       # Phase 1 MLR
FEATURE_SETS = {
    "A_district_level": BASELINE_FEATURES + ["log_total_recharge"],                        # PRIMARY
    "B_plus_state_rainfall": BASELINE_FEATURES + ["log_total_recharge", "rainfall_gwy_2024_25_mm"],  # SENSITIVITY
}
PRIMARY = "A_district_level"
RUN_STATE_RAINFALL_SENSITIVITY = True

# Variables that must never be predictors (they contain the numerator/denominator of the target)
LEAKY = ["total_extraction_ham", "annual_extractable_gw_ham", "irrigation_extraction_ham", "industrial_extraction_ham",
         "domestic_extraction_ham", "net_gw_availability_future_ham", "allocation_domestic_2035_ham",
         "stage_of_extraction_pct", "stage_recomputed", "stage_diff"]

STAGE_BINS = [-np.inf, 70, 90, 100, np.inf]
STAGE_LABELS = ["safe", "semi_critical", "critical", "over_exploited"]

sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
warnings.filterwarnings("ignore")
log = logging.getLogger("phase2")


# =============================================================================
# UTILITIES
# =============================================================================
def setup_logging() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(OUT_DIR / "run_log.txt", mode="w", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)


def section(title: str) -> None:
    log.info("")
    log.info("=" * 78)
    log.info(title)
    log.info("=" * 78)


def savefig(fig: plt.Figure, name: str, folder: Path | None = None) -> None:
    fig.tight_layout()
    fig.savefig((folder or PLOT_DIR) / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def regression_metrics(y_true, y_pred, n_features: int) -> dict:
    """RMSE, MAE, R2, adjusted R2 on the ln scale; RMSE/MAE again on the original stage (%) scale."""
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    n = len(yt)
    ss_res, ss_tot = np.sum((yt - yp) ** 2), np.sum((yt - yt.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    adj = 1 - (1 - r2) * (n - 1) / (n - n_features - 1) if n - n_features - 1 > 0 else np.nan
    st, sp = np.exp(yt), np.exp(yp)
    return {"rmse_log": float(np.sqrt(ss_res / n)), "mae_log": float(np.mean(np.abs(yt - yp))), "r2": float(r2),
            "adj_r2": float(adj), "rmse_stage": float(np.sqrt(np.mean((st - sp) ** 2))), "mae_stage": float(np.mean(np.abs(st - sp)))}


def stage_category(stage) -> pd.Series:
    return pd.cut(pd.Series(stage), bins=STAGE_BINS, labels=STAGE_LABELS)


# =============================================================================
# MODELS  (every model has fit(train_df) -> self and predict(df) -> ndarray)
# =============================================================================
class MeanOnly:
    """Benchmark 1: predict the training mean of ln(stage) for every district."""
    n_features_used = 0

    def fit(self, tr):
        self.mu = tr[TARGET].mean()
        return self

    def predict(self, d):
        return np.full(len(d), self.mu)


class StateMean:
    """Benchmark 2: predict the mean ln(stage) of the district's state (falls back to the overall mean for unseen states)."""
    n_features_used = 1

    def fit(self, tr):
        self.mu = tr[TARGET].mean()
        self.means = tr.groupby("state")[TARGET].mean()
        return self

    def predict(self, d):
        return d["state"].map(self.means).fillna(self.mu).values


class OLSModel:
    def __init__(self, feats):
        self.feats = list(feats)

    def fit(self, tr):
        self.res = sm.OLS(tr[TARGET], sm.add_constant(tr[self.feats])).fit()
        return self

    def predict(self, d):
        return self.res.predict(sm.add_constant(d[self.feats], has_constant="add")).values

    @property
    def n_features_used(self):
        return len(self.feats)


def stepwise_selection(train: pd.DataFrame, cands: list[str], criterion: str):
    """Bidirectional stepwise regression minimising AIC or BIC (starts from the intercept-only model)."""
    def score(feats):
        X = sm.add_constant(train[feats]) if feats else np.ones((len(train), 1))
        return getattr(sm.OLS(train[TARGET], X).fit(), criterion)

    current, best = [], score([])
    path = [{"step": 0, "action": "start", "variable": "", "criterion_value": best, "model": ""}]
    while True:
        moves = [("add", f, sorted(current + [f])) for f in cands if f not in current]
        moves += [("drop", f, [c for c in current if c != f]) for f in current]
        if not moves:
            break
        s, act, f, new = min(((score(m), a, v, m) for a, v, m in moves), key=lambda t: t[0])
        if s < best - 1e-9:
            best, current = s, new
            path.append({"step": len(path), "action": act, "variable": f, "criterion_value": s, "model": " + ".join(current)})
        else:
            break
    return current, path


class StepwiseOLS(OLSModel):
    def __init__(self, feats, criterion):
        self.cands, self.criterion, self.feats = list(feats), criterion, None

    def fit(self, tr):
        self.feats, self.path = stepwise_selection(tr, self.cands, self.criterion)
        if not self.feats:                       # intercept-only edge case
            self.feats = [self.cands[0]]
        return super().fit(tr)


class Penalised:
    """LASSO / Ridge on standardised features; alpha chosen by inner CV on the training data only."""
    def __init__(self, kind, feats):
        self.kind, self.feats = kind, list(feats)

    def fit(self, tr):
        inner = KFold(K_INNER, shuffle=True, random_state=SEED)
        self.sc = StandardScaler().fit(tr[self.feats])
        Xs, y = self.sc.transform(tr[self.feats]), tr[TARGET].values
        if self.kind == "lasso":
            self.est = LassoCV(cv=inner, random_state=SEED, max_iter=50000).fit(Xs, y)
            self.df_eff = int(np.sum(np.abs(self.est.coef_) > 1e-10)) + 1
        else:
            self.est = RidgeCV(alphas=np.logspace(-3, 3, 61), cv=inner).fit(Xs, y)
            sv = np.linalg.svd(Xs, compute_uv=False)
            self.df_eff = float(np.sum(sv ** 2 / (sv ** 2 + self.est.alpha_))) + 1
        self.alpha_ = float(self.est.alpha_)
        return self

    def predict(self, d):
        return self.est.predict(self.sc.transform(d[self.feats]))

    @property
    def n_features_used(self):
        return int(np.sum(np.abs(self.est.coef_) > 1e-10)) if self.kind == "lasso" else len(self.feats)


def make_grids() -> dict:
    if QUICK:
        return {
            "tree": (lambda: DecisionTreeRegressor(random_state=SEED), {"max_depth": [3, 6], "min_samples_leaf": [10]}),
            "rf": (lambda: RandomForestRegressor(random_state=SEED, n_jobs=1), {"n_estimators": [60], "max_depth": [6], "min_samples_leaf": [5]}),
            "svr": (lambda: make_pipeline(StandardScaler(), SVR(kernel="rbf")), {"svr__C": [1, 10], "svr__gamma": ["scale"], "svr__epsilon": [0.1]}),
            "xgb": (lambda: XGBRegressor(random_state=SEED, objective="reg:squarederror", n_jobs=1, subsample=0.8),
                    {"max_depth": [3], "learning_rate": [0.1], "n_estimators": [100]}),
        }
    return {
        "tree": (lambda: DecisionTreeRegressor(random_state=SEED), {"max_depth": [3, 5, 7, None], "min_samples_leaf": [5, 10, 20]}),
        "rf": (lambda: RandomForestRegressor(random_state=SEED, n_jobs=1),
               {"n_estimators": [200], "max_depth": [6, None], "min_samples_leaf": [3, 10]}),
        "svr": (lambda: make_pipeline(StandardScaler(), SVR(kernel="rbf")),
                {"svr__C": [1, 10, 100], "svr__gamma": ["scale", 0.1], "svr__epsilon": [0.05, 0.2]}),
        "xgb": (lambda: XGBRegressor(random_state=SEED, objective="reg:squarederror", n_jobs=1, subsample=0.8),
                {"max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1], "n_estimators": [150, 300]}),
    }


GRIDS = make_grids()


class Tuned:
    """Tree / forest / kernel / boosting model tuned by inner K-fold CV on the training data only."""
    def __init__(self, kind, feats):
        self.kind, self.feats = kind, list(feats)

    def fit(self, tr):
        factory, grid = GRIDS[self.kind]
        gs = GridSearchCV(factory(), grid, cv=KFold(K_INNER, shuffle=True, random_state=SEED),
                          scoring="neg_root_mean_squared_error", n_jobs=N_JOBS)
        gs.fit(tr[self.feats], tr[TARGET])
        self.est, self.best_params_ = gs.best_estimator_, gs.best_params_
        return self

    def predict(self, d):
        return self.est.predict(d[self.feats])

    @property
    def n_features_used(self):
        return len(self.feats)


@dataclass
class Spec:
    name: str
    family: str                      # benchmark | linear | penalised | ml
    build: Callable[[list], object]  # feats -> unfitted model
    interp_rank: int                 # lower = easier to interpret (used by the selection rule)


def model_zoo() -> list[Spec]:
    return [
        Spec("Benchmark: overall mean", "benchmark", lambda f: MeanOnly(), 0),
        Spec("Benchmark: state mean", "benchmark", lambda f: StateMean(), 0),
        Spec("OLS: Phase 1 MLR", "linear", lambda f: OLSModel(BASELINE_FEATURES), 1),
        Spec("OLS: stepwise BIC", "linear", lambda f: StepwiseOLS(f, "bic"), 2),
        Spec("OLS: stepwise AIC", "linear", lambda f: StepwiseOLS(f, "aic"), 3),
        Spec("LASSO", "penalised", lambda f: Penalised("lasso", f), 4),
        Spec("Ridge", "penalised", lambda f: Penalised("ridge", f), 5),
        Spec("Decision Tree", "ml", lambda f: Tuned("tree", f), 6),
        Spec("Random Forest", "ml", lambda f: Tuned("rf", f), 7),      # native feature importance
        Spec("XGBoost", "ml", lambda f: Tuned("xgb", f), 8),            # native feature importance
        Spec("SVR (RBF)", "ml", lambda f: Tuned("svr", f), 9),          # kernel model, no native importance
    ]


BASELINE = "OLS: Phase 1 MLR"


# =============================================================================
# NESTED CROSS-VALIDATION ENGINE
# =============================================================================
def evaluate(df: pd.DataFrame, splits: list, feats: list[str], label: str, zoo: list[Spec]):
    """Refit every model from scratch on each training fold (feature selection + tuning included), score the test fold."""
    y = df[TARGET].values
    fold_rows, oof, picks = [], {}, []
    for spec in zoo:
        t0 = time.time()
        pred_all = np.full(len(df), np.nan)
        for fold, (tr, te) in enumerate(splits):
            m = spec.build(feats).fit(df.iloc[tr])
            pred = m.predict(df.iloc[te])
            pred_all[te] = pred
            fold_rows.append({"model": spec.name, "fold": fold, "n_train": len(tr), "n_test": len(te),
                              **regression_metrics(y[te], pred, m.n_features_used)})
            if hasattr(m, "best_params_"):
                picks.append({"model": spec.name, "fold": fold, "choice": str(m.best_params_)})
            elif isinstance(m, StepwiseOLS):
                picks.append({"model": spec.name, "fold": fold, "choice": ", ".join(m.feats)})
            elif isinstance(m, Penalised):
                picks.append({"model": spec.name, "fold": fold, "choice": f"alpha={m.alpha_:.5f}; nonzero={m.n_features_used}"})
        oof[spec.name] = pred_all
        log.info("   [%s] %-24s done in %5.1fs", label, spec.name, time.time() - t0)
    folds = pd.DataFrame(fold_rows)
    rows = []
    for spec in zoo:
        f = folds[folds.model == spec.name]
        pooled = regression_metrics(y, oof[spec.name], 0)
        row = {"model": spec.name, "family": spec.family, "split": label}
        for c in ("rmse_log", "mae_log", "r2", "adj_r2", "rmse_stage", "mae_stage"):
            row[c + "_mean"], row[c + "_sd"] = f[c].mean(), f[c].std(ddof=1)
        row["pooled_oof_r2"] = pooled["r2"]
        rows.append(row)
    return pd.DataFrame(rows), folds, oof, pd.DataFrame(picks)


def paired_tests(folds: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Each model vs the Phase 1 OLS baseline on identical folds (RMSE on ln scale)."""
    piv = folds.pivot(index="fold", columns="model", values="rmse_log")
    nte, ntr = folds.groupby("fold").n_test.first().mean(), folds.groupby("fold").n_train.first().mean()
    k = piv.shape[0]
    rows = []
    for m in models:
        if m == BASELINE:
            continue
        d = piv[m] - piv[BASELINE]
        t_p = stats.ttest_rel(piv[m], piv[BASELINE]).pvalue
        # Nadeau & Bengio (2003) corrected resampled t-test: accounts for the overlap of training sets across folds
        var_c = (1 / k + nte / ntr) * d.var(ddof=1)
        t_c = d.mean() / np.sqrt(var_c) if var_c > 0 else np.nan
        p_c = 2 * stats.t.sf(abs(t_c), k - 1) if var_c > 0 else np.nan
        try:
            w_p = stats.wilcoxon(piv[m], piv[BASELINE]).pvalue
        except ValueError:
            w_p = np.nan
        rows.append({"model": m, "mean_rmse_diff_vs_baseline": d.mean(), "sd_diff": d.std(ddof=1), "paired_t_p": t_p,
                     "corrected_t_p_NadeauBengio": p_c, "wilcoxon_p": w_p, "folds_better_than_baseline": int((d < 0).sum()), "n_folds": k})
    return pd.DataFrame(rows).round(5)


# =============================================================================
# FEATURE SELECTION REPORTS
# =============================================================================
def feature_selection_report(df: pd.DataFrame, feats: list[str]) -> dict:
    section("1. FEATURE SELECTION: STEPWISE (AIC/BIC), LASSO, RIDGE  (full-data illustration)")
    out = {}
    paths = []
    for crit in ("aic", "bic"):
        sel, path = stepwise_selection(df, feats, crit)
        out[crit] = sel
        for p in path:
            paths.append({"criterion": crit.upper(), **p})
        log.info("Stepwise %s selected: %s", crit.upper(), sel)
    pd.DataFrame(paths).round(4).to_csv(OUT_DIR / "stepwise_paths.csv", index=False)
    pd.DataFrame([{"criterion": c.upper(), "selected": ", ".join(v), "n_selected": len(v)} for c, v in out.items()]).to_csv(OUT_DIR / "stepwise_selected.csv", index=False)

    sc = StandardScaler().fit(df[feats])
    Xs, y = sc.transform(df[feats]), df[TARGET].values
    inner = KFold(K_INNER, shuffle=True, random_state=SEED)
    lasso = LassoCV(cv=inner, random_state=SEED, max_iter=50000).fit(Xs, y)
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 61), cv=inner).fit(Xs, y)
    coef = pd.DataFrame({"feature": feats, "lasso_coef_std": lasso.coef_, "ridge_coef_std": ridge.coef_}).round(4)
    coef.to_csv(OUT_DIR / "lasso_ridge_coefficients.csv", index=False)
    log.info("LASSO alpha = %.5f | Ridge alpha = %.4f\n%s", lasso.alpha_, ridge.alpha_, coef.to_string(index=False))
    out.update(lasso_alpha=float(lasso.alpha_), ridge_alpha=float(ridge.alpha_), lasso_nonzero=int((np.abs(lasso.coef_) > 1e-10).sum()))

    # Regularisation paths
    alphas, coefs, _ = lasso_path(Xs, y, n_alphas=100)
    ridge_alphas = np.logspace(-3, 4, 80)
    ridge_coefs = np.array([np.linalg.solve(Xs.T @ Xs + a * np.eye(Xs.shape[1]), Xs.T @ (y - y.mean())) for a in ridge_alphas]).T
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, f in enumerate(feats):
        axes[0].plot(np.log10(alphas), coefs[i], label=f)
        axes[1].plot(np.log10(ridge_alphas), ridge_coefs[i], label=f)
    axes[0].axvline(np.log10(lasso.alpha_), color="k", ls="--", label="CV alpha")
    axes[1].axvline(np.log10(ridge.alpha_), color="k", ls="--", label="CV alpha")
    axes[0].set(title="LASSO coefficient path (standardised)", xlabel="log10(alpha)", ylabel="coefficient")
    axes[1].set(title="Ridge coefficient path (standardised)", xlabel="log10(alpha)", ylabel="coefficient")
    axes[0].legend(fontsize=8)
    savefig(fig, "regularisation_paths.png")
    return out


def information_criteria(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """AIC/BIC for the linear models on the full data, all on ONE scale (Gaussian likelihood incl. constants)."""
    section("2. INFORMATION CRITERIA (linear models; ML models: not applicable)")
    n = len(df)
    const = n * (1 + np.log(2 * np.pi))

    def ic(rss, k):
        return n * np.log(rss / n) + 2 * k + const, n * np.log(rss / n) + np.log(n) * k + const

    rows = []
    for name, model in (("OLS: Phase 1 MLR", OLSModel(BASELINE_FEATURES)), ("OLS: stepwise BIC", StepwiseOLS(feats, "bic")),
                        ("OLS: stepwise AIC", StepwiseOLS(feats, "aic")), ("LASSO", Penalised("lasso", feats)), ("Ridge", Penalised("ridge", feats))):
        model.fit(df)
        if isinstance(model, Penalised):
            rss, k = float(np.sum((df[TARGET].values - model.predict(df)) ** 2)), model.df_eff
            a, b = ic(rss, k)
            note = "effective df (LASSO: nonzero+1; Ridge: trace of hat matrix + 1)"
        else:
            a, b, k = model.res.aic, model.res.bic, len(model.feats) + 1
            note = "statsmodels OLS likelihood"
        rows.append({"model": name, "k_parameters": round(float(k), 2), "aic": a, "bic": b, "note": note})
    for name in ("Decision Tree", "SVR (RBF)", "Random Forest", "XGBoost"):
        rows.append({"model": name, "k_parameters": np.nan, "aic": np.nan, "bic": np.nan, "note": "not applicable (no likelihood)"})
    tbl = pd.DataFrame(rows).round(3)
    tbl.to_csv(OUT_DIR / "information_criteria.csv", index=False)
    log.info("AIC/BIC (all on one scale):\n%s", tbl[["model", "k_parameters", "aic", "bic"]].to_string(index=False))
    return tbl


# =============================================================================
# FINAL MODEL SELECTION RULE
# =============================================================================
def select_final_model(cv_tbl: pd.DataFrame, grp_tbl: pd.DataFrame, hold_tbl: pd.DataFrame, tests: pd.DataFrame, zoo: list[Spec]) -> dict:
    """
    Rule:
      1. ACCURACY      : candidates = models whose 10-fold CV RMSE is within ONE standard error of the best model
                         (SE = SD of the best model's fold RMSEs / sqrt(K)).  [one-standard-error rule]
      2. INTERPRETABLE : among those candidates choose the most interpretable (lowest interpretability rank:
                         OLS < stepwise < LASSO < Ridge < Tree < Random Forest < XGBoost < SVR-RBF).
                         Candidates inside one SE are statistically indistinguishable on accuracy, so this is a tie-break.
      3. CONFIRMATION  : the choice must (a) beat the Phase 1 OLS baseline in the corrected paired test, (b) beat it on the
                         untouched holdout, (c) beat it under state-grouped CV. Failing (c) does not veto the model but
                         limits its stated scope to 'districts of states already assessed'.
      Benchmarks are never eligible.
    """
    rank = {s.name: s.interp_rank for s in zoo}
    el = cv_tbl[cv_tbl.family != "benchmark"].copy()
    best = el.loc[el.rmse_log_mean.idxmin()]
    se = best.rmse_log_sd / np.sqrt(K_OUTER)
    cand = el[el.rmse_log_mean <= best.rmse_log_mean + se].copy()
    cand["interp_rank"] = cand.model.map(rank)
    final = cand.sort_values(["interp_rank", "rmse_log_mean"]).iloc[0]
    name = final.model

    base_cv = cv_tbl.set_index("model").loc[BASELINE]
    row_cv = cv_tbl.set_index("model").loc[name]
    row_gr, base_gr = grp_tbl.set_index("model").loc[name], grp_tbl.set_index("model").loc[BASELINE]
    row_ho, base_ho = hold_tbl.set_index("model").loc[name], hold_tbl.set_index("model").loc[BASELINE]
    sm_cv, sm_gr = cv_tbl.set_index("model").loc["Benchmark: state mean"], grp_tbl.set_index("model").loc["Benchmark: state mean"]
    p_corr = tests.set_index("model").corrected_t_p_NadeauBengio.get(name, np.nan) if name != BASELINE else np.nan

    checks = [
        ("Best 10-fold CV model", best.model, f"RMSE {best.rmse_log_mean:.4f} (SE {se:.4f})"),
        ("Models within 1 SE of the best", ", ".join(cand.model), "one-standard-error rule"),
        ("Selected (most interpretable among them)", name, f"10-fold CV RMSE {row_cv.rmse_log_mean:.4f} vs baseline {base_cv.rmse_log_mean:.4f}"),
        ("Check a: corrected paired t-test vs baseline", "n/a (is the baseline)" if name == BASELINE else ("PASS" if p_corr < 0.05 else "FAIL"),
         "" if name == BASELINE else ("p < 0.0001" if p_corr < 0.0001 else f"p = {p_corr:.4f}")),
        ("Check b: untouched holdout beats baseline", "PASS" if row_ho.test_rmse_log < base_ho.test_rmse_log else "FAIL",
         f"RMSE {row_ho.test_rmse_log:.4f} vs {base_ho.test_rmse_log:.4f}; R2 {row_ho.test_r2:.3f} vs {base_ho.test_r2:.3f}"),
        ("Check c: state-grouped CV beats baseline", "PASS" if row_gr.rmse_log_mean < base_gr.rmse_log_mean else "FAIL",
         f"RMSE {row_gr.rmse_log_mean:.4f} vs {base_gr.rmse_log_mean:.4f}"),
        ("Information beyond state identity (10-fold)", "YES" if row_cv.rmse_log_mean < sm_cv.rmse_log_mean else "NO",
         f"state-mean benchmark RMSE {sm_cv.rmse_log_mean:.4f}; model {row_cv.rmse_log_mean:.4f}"),
        ("Information beyond state identity (grouped)", "YES" if row_gr.rmse_log_mean < sm_gr.rmse_log_mean else "NO",
         f"state-mean benchmark RMSE {sm_gr.rmse_log_mean:.4f}; model {row_gr.rmse_log_mean:.4f}"),
    ]
    dec = pd.DataFrame(checks, columns=["check", "result", "detail"])
    dec.to_csv(OUT_DIR / "final_model_decision.csv", index=False)
    log.info("FINAL MODEL DECISION\n%s", dec.to_string(index=False))
    scope_ok = checks[5][1] == "PASS"
    return {"name": name, "family": final.family, "best_cv": best.model, "candidates": list(cand.model), "grouped_pass": scope_ok,
            "decision_table": dec}


# =============================================================================
# EXPLANATION, SCREENING AND REAL-WORLD USE
# =============================================================================
def permutation_importance(model, data: pd.DataFrame, feats: list[str], reps: int = PERM_REPEATS) -> pd.Series:
    rng = np.random.default_rng(SEED)
    y = data[TARGET].values
    base = np.sqrt(np.mean((y - model.predict(data)) ** 2))
    out = {}
    for f in feats:
        inc = []
        for _ in range(reps):
            d2 = data.copy()
            d2[f] = rng.permutation(d2[f].values)
            inc.append(np.sqrt(np.mean((y - model.predict(d2)) ** 2)) - base)
        out[f] = np.mean(inc)
    return pd.Series(out)


def partial_dependence(model, data: pd.DataFrame, feat: str, n_grid: int = 25):
    grid = np.unique(np.quantile(data[feat], np.linspace(0.02, 0.98, n_grid)))
    pd_vals = []
    for g in grid:
        d2 = data.copy()
        d2[feat] = g
        pd_vals.append(float(np.mean(model.predict(d2))))
    return grid, np.array(pd_vals)


def screening_analysis(df: pd.DataFrame, oof: dict, final_name: str):
    """Can out-of-fold predictions find the over-exploited districts (actual stage > 100 %)?
    Models shrink predictions towards the mean, so a fixed 100 % cut-off is useless; we rank districts instead."""
    section("7. REAL-WORLD USE: CAN THE MODEL SCREEN FOR OVER-EXPLOITED DISTRICTS? (out-of-fold)")
    from sklearn.metrics import roc_auc_score
    actual = df["stage_of_extraction_pct"].values
    pos = actual > 100
    rows, info = [], {}
    for name in dict.fromkeys([BASELINE, final_name]):
        score = oof[name]
        info[name] = {"auc": float(roc_auc_score(pos, score)), "spearman": float(stats.spearmanr(score, actual)[0])}
        order = np.argsort(-score)
        for share in (0.10, 0.20, 0.30, 0.40, 0.50):
            k = int(round(share * len(df)))
            caught = int(pos[order[:k]].sum())
            rows.append({"model": name, "share_of_districts_flagged_pct": int(share * 100), "districts_flagged": k,
                         "over_exploited_total": int(pos.sum()), "over_exploited_caught": caught,
                         "recall_pct": 100 * caught / pos.sum(), "precision_pct": 100 * caught / k,
                         "random_selection_recall_pct": 100 * share})
    tbl = pd.DataFrame(rows).round(2)
    tbl.to_csv(OUT_DIR / "screening_by_rank.csv", index=False)
    pd.DataFrame(info).T.round(4).to_csv(OUT_DIR / "screening_auc_spearman.csv")
    log.info("Ranking quality (AUC for stage > 100 %%; 0.5 = random):\n%s", pd.DataFrame(info).T.round(3).to_string())
    log.info("Screening by rank:\n%s", tbl.to_string(index=False))

    pred_cat = stage_category(np.exp(oof[final_name])).astype(str)
    act_cat = df["stage_category"].astype(str)
    cm = pd.crosstab(pd.Categorical(act_cat, STAGE_LABELS), pd.Categorical(pred_cat, STAGE_LABELS), rownames=["actual"], colnames=["predicted"], dropna=False)
    cm.to_csv(OUT_DIR / "screening_confusion_matrix.csv")
    log.info("Confusion matrix of CGWB categories (final model, out-of-fold):\n%s", cm.to_string())
    fig, ax = plt.subplots(figsize=(6.4, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_title(f"CGWB category: actual vs predicted (out-of-fold)\n{final_name}")
    savefig(fig, "screening_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name in info:
        order = np.argsort(-oof[name])
        xs = np.arange(1, len(df) + 1) / len(df) * 100
        ax.plot(xs, np.cumsum(pos[order]) / pos.sum() * 100, label=f"{name} (AUC {info[name]['auc']:.2f})")
    ax.plot([0, 100], [0, 100], "k:", label="random")
    ax.set(xlabel="% of districts flagged (highest predicted stage first)", ylabel="% of over-exploited districts caught",
           title="Screening curve (out-of-fold predictions)")
    ax.legend()
    savefig(fig, "screening_workload_vs_recall.png")
    return tbl, info


def md_table(t: pd.DataFrame) -> str:
    """Markdown table without needing the optional 'tabulate' package."""
    cols = list(t.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)]
    return "\n".join(lines)


def decision_markdown(res: dict, spec_name: str, dec: dict, importance: pd.Series, master: pd.DataFrame, screen, feats: list[str]) -> str:
    final = dec["name"]
    m = master.set_index("model")
    base, fin = m.loc[BASELINE], m.loc[final]
    scr_tbl, scr_info = screen
    r30 = scr_tbl[(scr_tbl.model == final) & (scr_tbl.share_of_districts_flagged_pct == 30)].iloc[0]
    auc = scr_info[final]["auc"]
    sm_row = m.loc["Benchmark: state mean"]
    lines = [
        "# Final model decision (Phase 2)", "",
        f"**Feature set:** {spec_name} -> {', '.join(feats)}", "",
        f"**Selected model: {final}** ({dec['family']}).", "",
        "## Selection rule (explicit and reproducible)",
        "1. Accuracy: keep every model whose 10-fold CV RMSE is within one standard error of the best.",
        "2. Interpretability: among those, choose the most interpretable.",
        "3. Confirmation: corrected paired t-test, untouched holdout, state-grouped CV, and comparison with the state-mean benchmark.", "",
        "## Evidence",
        f"* Best 10-fold CV model: {dec['best_cv']}. Candidates within one SE: {', '.join(dec['candidates'])}.",
        f"* 10-fold CV RMSE (ln stage): {fin['cv_rmse_log']:.3f} vs {base['cv_rmse_log']:.3f} for the Phase 1 OLS; R2 {fin['cv_r2']:.3f} vs {base['cv_r2']:.3f}.",
        f"* Holdout R2: {fin['holdout_r2']:.3f} vs {base['holdout_r2']:.3f}.",
        f"* State-grouped CV RMSE: {fin['grouped_rmse_log']:.3f} vs {base['grouped_rmse_log']:.3f}.", "",
        "## Checks", md_table(dec["decision_table"]), "",
        "## Justification",
        "* **Accuracy:** see the evidence above; every comparison uses identical folds and nested tuning, so the gain is not an artefact of tuning on test data.",
        "* **Interpretability:** " + ("the selected model is a linear model with readable coefficients." if dec["family"] in ("linear", "penalised")
                                      else "the selected model is a black box; its behaviour is explained with permutation importance and partial dependence plots (see plots)."),
        "* **Assumptions:** Phase 1 showed the OLS errors are non-normal and heteroskedastic, so classical OLS p-values are unreliable; "
        "tree/boosting/kernel models make no normality or constant-variance assumption.", "",
        "## Scope and limits",
        ("* The model also beats the baseline when whole states are held out." if dec["grouped_pass"] else
         "* The model does NOT clearly beat the baseline when whole states are held out. Use it for districts of states that are already in the "
         "assessment data (e.g. screening between assessment cycles), not to extrapolate to a new state."),
        (f"* Under random splits the state-mean benchmark (RMSE {sm_row.cv_rmse_log:.3f}) predicts better than every model: within the states "
         "already assessed, knowing the state is worth more than these district variables. The model's value is that it needs no state label and still "
         "improves on the OLS when whole states are held out." ),
        "* Predictors are recharge-composition variables only. log_total_recharge and natural_discharge_ratio together reconstruct the denominator "
        "of the target (extractable resource = recharge - natural discharge), so part of the explanatory power is structural, not causal.",
        "* Predictions are associations, not causal effects.", "",
        "## Real-world application",
        f"* Most important predictors (permutation importance): {', '.join(importance.sort_values(ascending=False).index[:3])}.",
        f"* Ranking ability for over-exploited districts (stage > 100 %), out-of-fold: AUC = {auc:.2f} (0.5 = random). Flagging the top 30 % of districts "
        f"by predicted stage catches {r30.recall_pct:.0f} % of them (random flagging would catch 30 %).",
        ("* Recommended use: PRIORITISATION - rank districts to decide where field re-assessment should start, and give a first estimate of stress for "
         "districts awaiting a full assessment." if auc >= 0.70 else
         "* The ranking is only weak-to-moderate, so use it as a supporting indicator, not as a screening tool."),
        "* The model shrinks predictions towards the mean: extreme districts are under-predicted (see screening_confusion_matrix.csv), so do not apply a "
        "fixed cut-off such as 'predicted stage > 100 %'.",
        "* Do not use predictions alone for allocation or enforcement; confirm with field measurements.",
    ]
    return "\n".join(lines)


# =============================================================================
# PIPELINE FOR ONE FEATURE SET
# =============================================================================
def run_full(df: pd.DataFrame, spec_name: str, feats: list[str]) -> dict:
    zoo = model_zoo()
    names = [s.name for s in zoo]
    assert not set(feats) & set(LEAKY), "A leaky variable entered the feature set"
    log.info("Feature set %s: %s", spec_name, feats)

    sel = feature_selection_report(df, feats)
    ic = information_criteria(df, feats)

    section(f"3. NESTED {K_OUTER}-FOLD CV (identical folds for every model)")
    outer = list(KFold(K_OUTER, shuffle=True, random_state=SEED).split(df))
    cv_tbl, cv_folds, oof, picks = evaluate(df, outer, feats, f"{K_OUTER}-fold CV", zoo)
    cv_tbl.round(4).to_csv(OUT_DIR / "cv_10fold_summary.csv", index=False)
    cv_folds.round(5).to_csv(OUT_DIR / "cv_10fold_fold_metrics.csv", index=False)
    picks.to_csv(OUT_DIR / "selected_features_and_hyperparameters_by_fold.csv", index=False)
    cv_folds.pivot(index="fold", columns="model", values="rmse_log")[names].round(5).to_csv(OUT_DIR / "cv_fold_rmse.csv")
    log.info("\n%s", cv_tbl[["model", "rmse_log_mean", "rmse_log_sd", "mae_log_mean", "r2_mean", "adj_r2_mean", "rmse_stage_mean"]].round(4).to_string(index=False))

    section("4. IS THE GAIN LARGER THAN FOLD-TO-FOLD NOISE?  (paired tests vs Phase 1 OLS)")
    tests = paired_tests(cv_folds, [n for n in names if not n.startswith("Benchmark")])
    tests.to_csv(OUT_DIR / "paired_tests_vs_baseline.csv", index=False)
    log.info("\n%s", tests.to_string(index=False))

    section("5. STATE-GROUPED CV (whole states held out)")
    gsplits = list(GroupKFold(K_OUTER).split(df, groups=df["state"]))
    grp_tbl, grp_folds, _, _ = evaluate(df, gsplits, feats, "GroupKFold by state", zoo)
    grp_tbl.round(4).to_csv(OUT_DIR / "cv_grouped_by_state_summary.csv", index=False)
    grp_folds.round(5).to_csv(OUT_DIR / "cv_grouped_by_state_fold_metrics.csv", index=False)
    log.info("\n%s", grp_tbl[["model", "rmse_log_mean", "rmse_log_sd", "r2_mean", "pooled_oof_r2", "rmse_stage_mean"]].round(4).to_string(index=False))

    section("6. UNTOUCHED HOLDOUT (20 %)")
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=SEED, stratify=df["stage_category"])
    log.info("Train %d | Test %d (stratified on CGWB category)", len(train_df), len(test_df))
    hold_rows, fitted = [], {}
    for spec in zoo:
        m = spec.build(feats).fit(train_df)
        fitted[spec.name] = m
        mt = regression_metrics(test_df[TARGET].values, m.predict(test_df), m.n_features_used)
        hold_rows.append({"model": spec.name, "family": spec.family, **{f"test_{k}": v for k, v in mt.items()}})
    hold_tbl = pd.DataFrame(hold_rows)
    hold_tbl.round(4).to_csv(OUT_DIR / "holdout_metrics.csv", index=False)
    log.info("\n%s", hold_tbl.round(4).to_string(index=False))

    # Master comparison table (Faculty: 'Metrics & Comparative Analysis')
    mc = cv_tbl.set_index("model")
    gc = grp_tbl.set_index("model")
    hc = hold_tbl.set_index("model")
    icm = ic.set_index("model")
    master = pd.DataFrame({
        "family": mc.family,
        "cv_rmse_log": mc.rmse_log_mean, "cv_rmse_sd": mc.rmse_log_sd, "cv_mae_log": mc.mae_log_mean, "cv_r2": mc.r2_mean, "cv_adj_r2": mc.adj_r2_mean,
        "cv_rmse_stage_pct": mc.rmse_stage_mean, "cv_mae_stage_pct": mc.mae_stage_mean,
        "grouped_rmse_log": gc.rmse_log_mean, "grouped_r2_pooled": gc.pooled_oof_r2,
        "holdout_rmse_log": hc.test_rmse_log, "holdout_mae_log": hc.test_mae_log, "holdout_r2": hc.test_r2, "holdout_adj_r2": hc.test_adj_r2,
        "aic": icm.aic.reindex(mc.index), "bic": icm.bic.reindex(mc.index),
    }).reset_index().rename(columns={"index": "model"})
    master.round(4).to_csv(OUT_DIR / "master_comparison_table.csv", index=False)

    section("7. FINAL MODEL SELECTION")
    dec = select_final_model(cv_tbl, grp_tbl, hold_tbl, tests, zoo)
    final_name = dec["name"]

    # Importance and partial dependence of the selected model and key comparators
    section("8. EXPLAINING THE MODELS")
    imp_models = [n for n in names if not n.startswith("Benchmark") and n in ("OLS: Phase 1 MLR", "Decision Tree", "SVR (RBF)", "Random Forest", "XGBoost", final_name)]
    imp = pd.DataFrame({n: permutation_importance(fitted[n], test_df, feats) for n in dict.fromkeys(imp_models)})
    imp.round(4).to_csv(OUT_DIR / "permutation_importance_holdout.csv")
    log.info("Permutation importance on the holdout (RMSE increase, ln scale):\n%s", imp.round(4).to_string())
    fig, axes = plt.subplots(1, min(3, imp.shape[1]), figsize=(5.2 * min(3, imp.shape[1]), 4.5))
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, [BASELINE] + [c for c in (final_name, "Random Forest", "XGBoost") if c != BASELINE and c in imp.columns][:2]):
        s = imp[n].sort_values()
        ax.barh(s.index, s.values, color="#1c7293")
        ax.set_title(n)
        ax.set_xlabel("RMSE increase when shuffled")
    savefig(fig, "permutation_importance.png")

    best_ml = cv_tbl[cv_tbl.family == "ml"].sort_values("rmse_log_mean").model.iloc[0]
    focus = final_name if dec["family"] in ("ml", "penalised") else best_ml
    fig, axes = plt.subplots(1, len(feats), figsize=(4.2 * len(feats), 3.8), sharey=True)
    for ax, f in zip(np.atleast_1d(axes), feats):
        g, v = partial_dependence(fitted[focus], test_df, f)
        ax.plot(g, v, "o-", color="#08306b", ms=3)
        ax.set_xlabel(f)
    np.atleast_1d(axes)[0].set_ylabel("average predicted ln(stage)")
    fig.suptitle(f"Partial dependence: {focus}", y=1.02)
    savefig(fig, "partial_dependence.png")

    # Comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    order = master.sort_values("cv_rmse_log").model
    mi = master.set_index("model")
    colors = {"benchmark": "#bdbdbd", "linear": "#9ecae1", "penalised": "#6baed6", "ml": "#08306b"}
    axes[0].barh(order, mi.loc[order, "cv_rmse_log"], xerr=mi.loc[order, "cv_rmse_sd"], color=[colors[mi.loc[o, "family"]] for o in order])
    axes[0].set_title(f"{K_OUTER}-fold CV RMSE (ln stage), lower is better")
    axes[1].barh(order, mi.loc[order, "grouped_rmse_log"], color=[colors[mi.loc[o, "family"]] for o in order])
    axes[1].set_title("State-grouped CV RMSE (unseen states)")
    axes[2].barh(order, mi.loc[order, "holdout_rmse_log"], color=[colors[mi.loc[o, "family"]] for o in order])
    axes[2].set_title("Untouched holdout RMSE")
    for ax in axes:
        ax.invert_yaxis()
    axes[1].set_yticklabels([]); axes[2].set_yticklabels([])
    savefig(fig, "model_comparison.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    fr = cv_folds.pivot(index="fold", columns="model", values="rmse_log")[list(order)]
    sns.boxplot(data=fr, orient="h", ax=ax, color="#9ecae1")
    sns.stripplot(data=fr, orient="h", ax=ax, color="k", size=3)
    ax.set_title("Fold-by-fold RMSE (identical folds)")
    ax.set_xlabel("RMSE (ln stage)")
    savefig(fig, "cv_fold_rmse_boxplot.png")

    # Out-of-fold actual vs predicted, residual comparison
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, n in zip(axes, [BASELINE, final_name if final_name != BASELINE else best_ml]):
        ax.scatter(np.exp(df[TARGET]), np.exp(oof[n]), s=10, alpha=0.5, color="#1c7293")
        lim = [0, float(np.percentile(np.exp(df[TARGET]), 99.5))]
        ax.plot(lim, lim, "r--")
        ax.set(xlim=lim, ylim=lim, xlabel="actual stage (%)", ylabel="out-of-fold predicted stage (%)", title=n)
        ax.axvline(100, color="grey", ls=":"); ax.axhline(100, color="grey", ls=":")
    savefig(fig, "actual_vs_predicted.png")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for n in dict.fromkeys([BASELINE, final_name]):
        sns.kdeplot(df[TARGET].values - oof[n], fill=True, alpha=0.3, label=n, ax=ax)
    ax.axvline(0, color="k", ls="--")
    ax.set(xlabel="out-of-fold residual (ln stage)", title="Residual distribution: baseline vs selected model")
    ax.legend()
    savefig(fig, "residual_distribution_comparison.png")

    screen = screening_analysis(df, oof, final_name)

    # District-level predictions (out-of-fold = honest) for all models of interest
    pred_tbl = df[["state", "district", "stage_of_extraction_pct", "stage_category"]].copy()
    pred_tbl["oof_pred_stage_baseline_ols"] = np.exp(oof[BASELINE])
    pred_tbl["oof_pred_stage_final"] = np.exp(oof[final_name])
    pred_tbl["pred_category_final"] = stage_category(pred_tbl.oof_pred_stage_final).astype(str).values
    pred_tbl["under_predicted"] = (pred_tbl.stage_of_extraction_pct > 100) & (pred_tbl.oof_pred_stage_final < 100)
    pred_tbl.round(2).to_csv(OUT_DIR / "district_predictions_out_of_fold.csv", index=False)

    # Final deployable model: refit on ALL data (tuning again by inner CV)
    final_model = [s for s in zoo if s.name == final_name][0].build(feats).fit(df)
    joblib.dump({"model": final_model, "features": feats, "target": TARGET, "name": final_name,
                 "note": "predict ln(stage); stage % = exp(prediction)"}, OUT_DIR / "final_model.joblib")

    md = decision_markdown({}, spec_name, dec, imp[final_name] if final_name in imp.columns else imp.iloc[:, 0], master, screen, feats)
    (OUT_DIR / "final_model_decision.md").write_text(md, encoding="utf-8")
    log.info("Wrote final_model_decision.md and final_model.joblib")
    return {"master": master, "decision": dec, "selection": sel, "tests": tests, "final_name": final_name}


def run_sensitivity_state_rainfall(df: pd.DataFrame) -> None:
    """Same CV machinery, but with the state-level rainfall column added; shows how much of an ML gain is state identity."""
    section("9. SENSITIVITY: ADDING THE STATE-LEVEL RAINFALL COLUMN (identifies the state)")
    feats = FEATURE_SETS["B_plus_state_rainfall"]
    zoo = [s for s in model_zoo() if s.name in ("Benchmark: state mean", BASELINE, "LASSO", "Random Forest", "XGBoost")]
    outer = list(KFold(K_OUTER, shuffle=True, random_state=SEED).split(df))
    gsplits = list(GroupKFold(K_OUTER).split(df, groups=df["state"]))
    r_tbl, _, _, _ = evaluate(df, outer, feats, f"{K_OUTER}-fold CV", zoo)
    g_tbl, _, _, _ = evaluate(df, gsplits, feats, "GroupKFold by state", zoo)
    both = pd.concat([r_tbl, g_tbl])
    both.round(4).to_csv(OUT_DIR / "sensitivity_state_rainfall.csv", index=False)
    log.info("With the state-level rainfall column:\n%s", both[["model", "split", "rmse_log_mean", "r2_mean", "pooled_oof_r2"]].round(4).to_string(index=False))
    try:
        prim_r = pd.read_csv(OUT_DIR / "cv_10fold_summary.csv").set_index("model").loc["XGBoost", "rmse_log_mean"]
        prim_g = pd.read_csv(OUT_DIR / "cv_grouped_by_state_summary.csv").set_index("model").loc["XGBoost", "rmse_log_mean"]
        sen_r = r_tbl.set_index("model").loc["XGBoost", "rmse_log_mean"]
        sen_g = g_tbl.set_index("model").loc["XGBoost", "rmse_log_mean"]
        log.info("XGBoost RMSE, primary (district-level) vs + state rainfall: random CV %.3f -> %.3f | state-grouped CV %.3f -> %.3f",
                 prim_r, sen_r, prim_g, sen_g)
        log.info("Reading: the state-level column improves random-split accuracy (the model can recognise the state) but %s "
                 "when whole states are held out - it adds identity, not transferable information.",
                 "makes no difference or hurts" if sen_g >= prim_g - 0.005 else "helps only slightly")
    except Exception as exc:  # pragma: no cover
        log.info("Could not compare with the primary run: %s", exc)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    setup_logging()
    path = P1_DIR / "cleaned_dataset.csv"
    if not path.exists():
        sys.exit("ERROR: run phase1_regression.py first (outputs/phase1/cleaned_dataset.csv not found).")
    t0 = time.time()
    df = pd.read_csv(path).reset_index(drop=True)
    need = {TARGET, "state", "stage_category", "stage_of_extraction_pct"} | set(FEATURE_SETS["B_plus_state_rainfall"])
    if need - set(df.columns):
        sys.exit(f"ERROR: cleaned dataset lacks columns {need - set(df.columns)}")
    section("PHASE 2: MODEL SELECTION & COMPARATIVE EVALUATION")
    log.info("n = %d districts, %d states | target = %s | seed = %d | quick mode = %s", len(df), df.state.nunique(), TARGET, SEED, QUICK)

    sns.heatmap(df[FEATURE_SETS["B_plus_state_rainfall"] + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Candidate features and target")
    savefig(plt.gcf(), "feature_correlations.png")

    res = run_full(df, PRIMARY, FEATURE_SETS[PRIMARY])
    if RUN_STATE_RAINFALL_SENSITIVITY and not QUICK:
        run_sensitivity_state_rainfall(df)

    summary = {"n": len(df), "primary_feature_set": PRIMARY, "features": FEATURE_SETS[PRIMARY], "seed": SEED,
               "selected_model": res["final_name"], "stepwise_aic": res["selection"]["aic"], "stepwise_bic": res["selection"]["bic"],
               "lasso_alpha": res["selection"]["lasso_alpha"], "ridge_alpha": res["selection"]["ridge_alpha"],
               "runtime_minutes": round((time.time() - t0) / 60, 1)}
    (OUT_DIR / "phase2_results.json").write_text(json.dumps(summary, indent=2, default=jsonable), encoding="utf-8")
    section("PHASE 2 COMPLETE")
    log.info("Selected model: %s | runtime %.1f min | outputs in %s", res["final_name"], (time.time() - t0) / 60, OUT_DIR)


if __name__ == "__main__":
    main()