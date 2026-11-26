#!/usr/bin/env python3
"""
Statistical Modeling (24AM5PCSML) - Alternative Assessment Tool (AAT)
======================================================================
PHASE 1 : Classical Regression & Estimator Validation
Topic   : District-level groundwater stress in India (CGWB Dynamic Ground Water
          Resources Assessment 2025)
SDG     : SDG 6 - Clean Water and Sanitation (target 6.4, water-use efficiency
          and sustainable withdrawals); also supports SDG 13 (Climate Action).

What this script does (maps 1-to-1 to the faculty brief)
--------------------------------------------------------
 1. Data preparation   : load raw CGWB district table, validate, clean, log every
                         change, derive features, join state rainfall.
 2. EDA                : descriptive statistics, histograms, correlation heatmap,
                         scatterplots with LOWESS, state-wise distribution.
 3. Simple linear reg. : exactly one predictor (chosen objectively).
 4. Multiple linear reg: three predictors (recharge-composition variables).
 5. Assumption checks  : linearity (RESET), residual normality (JB, Shapiro-Wilk,
                         Q-Q), homoscedasticity (Breusch-Pagan, White),
                         multicollinearity (VIF, condition number), influence
                         (leverage, Cook's D), independence (state clustering).
 6. Estimator validation: bootstrap (pairs and residual) of the OLS coefficients;
                         empirical mean and variance compared with the OLS theory;
                         histograms with normal overlays; k-fold CV.
 7. Sensitivity checks : outliers, raw target, state rainfall, state fixed effects,
                         2024 replication.

Outputs (all written under outputs/phase1/)
-------------------------------------------
 cleaned_dataset.csv (input for Phase 2), cleaning_log.csv, data_dictionary.csv,
 descriptive_stats.csv, coefficient tables, diagnostics, bootstrap tables, CV
 tables, sensitivity tables, results.json, run_log.txt and plots/*.png

Folder layout expected
----------------------
 SML_AAT/
   data/   district_gw_resources_2025_annexure_II.csv
           07_state_rainfall_gwy_2024_25.csv
           groundwater_district_2024.csv          (robustness only)
           state_gw_resources_2025.csv            (reconciliation only, optional)
   phase1_regression.py   <- this file
   phase2_model_selection.py

Run:  python phase1_regression.py
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to disk; no GUI window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import GroupKFold, KFold
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =============================================================================
# CONFIGURATION  (the only block you should need to edit)
# =============================================================================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "phase1"
PLOT_DIR = OUT_DIR / "plots"

FILES = {
    "district": "district_gw_resources_2025_annexure_II.csv",   # main data
    "rainfall": "07_state_rainfall_gwy_2024_25.csv",           # state rainfall (mm)
    "prev_year": "groundwater_district_2024.csv",              # robustness only
    "state_totals": "state_gw_resources_2025.csv",             # reconciliation only
}

SEED = 42                 # one seed for every random operation
N_BOOT = 2000             # bootstrap replications (faculty range: 1000-2000+)
K_FOLDS = 10              # k for k-fold cross-validation
HAM_PER_BCM = 1e5         # 1 bcm = 1e9 m3 ; 1 ham = 1e4 m3  ->  1 bcm = 1e5 ham
STAGE_TOL = 0.5           # tolerance (percentage points) for recomputed stage
GAP_TOL_HAM = 1.0         # tolerance for recharge component reconciliation

TARGET = "log_stage"                      # ln(stage of extraction %)
MLR_FEATURES = ["rainfall_share", "monsoon_share", "natural_discharge_ratio"]
SLR_FEATURE = None                        # None -> pick the feature with the largest |r| with the target
# Step sizes used to translate coefficients into "% change in stage" statements.
EFFECT_STEP = {"rainfall_share": 0.10, "monsoon_share": 0.10, "natural_discharge_ratio": 0.01}

# CGWB categorisation of an assessment unit by stage of extraction (%)
STAGE_BINS = [-np.inf, 70, 90, 100, np.inf]
STAGE_LABELS = ["safe", "semi_critical", "critical", "over_exploited"]

REQUIRED_COLS = [
    "state_ut", "district", "monsoon_recharge_rainfall_ham", "monsoon_recharge_other_sources_ham",
    "nonmonsoon_recharge_rainfall_ham", "nonmonsoon_recharge_other_sources_ham",
    "total_annual_recharge_ham", "total_natural_discharges_ham", "annual_extractable_gw_ham",
    "total_extraction_ham", "stage_of_extraction_pct",
]

sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
warnings.filterwarnings("ignore", category=FutureWarning)
log = logging.getLogger("phase1")


# =============================================================================
# SMALL UTILITIES
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


def norm_state(name: str) -> str:
    """One canonical key for state labels so that all files join cleanly."""
    s = str(name).upper().replace("&", "AND").replace("CHHATISGARH", "CHHATTISGARH")
    return " ".join(s.split())


def savefig(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(PLOT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def to_jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Series,)):
        return o.to_dict()
    return str(o)


def stage_category(stage: pd.Series) -> pd.Series:
    return pd.cut(stage, bins=STAGE_BINS, labels=STAGE_LABELS)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> dict:
    """RMSE, MAE, R2 and adjusted R2 on the modelling (ln) scale and RMSE/MAE on the stage (%) scale."""
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    n = len(yt)
    ss_res, ss_tot = np.sum((yt - yp) ** 2), np.sum((yt - yt.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    adj = 1 - (1 - r2) * (n - 1) / (n - n_features - 1) if n - n_features - 1 > 0 else np.nan
    st, sp = np.exp(yt), np.exp(yp)
    return {
        "rmse_log": float(np.sqrt(ss_res / n)), "mae_log": float(np.mean(np.abs(yt - yp))),
        "r2": float(r2), "adj_r2": float(adj),
        "rmse_stage": float(np.sqrt(np.mean((st - sp) ** 2))), "mae_stage": float(np.mean(np.abs(st - sp))),
    }


class CleaningLog:
    """Audit trail: every change to the data is recorded (faculty/viva evidence)."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, step: str, action: str, before: int, after: int, detail: str) -> None:
        self.rows.append(dict(step=step, action=action, rows_before=before, rows_after=after, detail=detail))
        log.info("[%s] %s | rows %d -> %d | %s", step, action, before, after, detail)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


# =============================================================================
# 1. DATA PREPARATION
# =============================================================================
def check_inputs() -> None:
    missing = [f for k, f in FILES.items() if k in ("district", "rainfall") and not (DATA_DIR / f).exists()]
    if missing:
        sys.exit(f"ERROR: put these files in {DATA_DIR}: {missing}")
    for k in ("prev_year", "state_totals"):
        if not (DATA_DIR / FILES[k]).exists():
            log.info("NOTE: optional file %s not found - the related check will be skipped.", FILES[k])


def load_and_clean() -> tuple[pd.DataFrame, CleaningLog]:
    section("1. DATA PREPARATION")
    cl = CleaningLog()
    raw = pd.read_csv(DATA_DIR / FILES["district"])
    miss_cols = [c for c in REQUIRED_COLS if c not in raw.columns]
    if miss_cols:
        sys.exit(f"ERROR: district file is missing columns: {miss_cols}")
    cl.add("S1.1", "load raw district table", len(raw), len(raw), FILES["district"])

    df = raw.copy()
    df["district"] = df["district"].astype(str).str.strip()
    df["state"] = df["state_ut"].map(norm_state)
    dup = int(df.duplicated(["state", "district"]).sum())
    cl.add("S1.2", "duplicate (state, district) check", len(df), len(df), f"{dup} duplicates found")
    if dup:
        df = df.drop_duplicates(["state", "district"], keep="first")

    # Rows whose recharge is not published ('-' in the PDF) cannot be modelled.
    na_mask = df["total_annual_recharge_ham"].isna()
    dropped = "; ".join(f"{r.district} ({r.state_ut})" for r in df[na_mask].itertuples())
    n0 = len(df)
    df = df.loc[~na_mask].copy()
    cl.add("S1.3", "drop districts with unpublished recharge", n0, len(df), f"dropped: {dropped or 'none'}")

    other_na = int(df[REQUIRED_COLS].isna().sum().sum())
    cl.add("S1.4", "remaining missing values in required columns", len(df), len(df), f"{other_na} cells")
    if other_na:
        sys.exit("ERROR: unexpected missing values remain; inspect the raw file.")

    # Validity checks
    for c in ("total_annual_recharge_ham", "annual_extractable_gw_ham", "stage_of_extraction_pct"):
        if (df[c] <= 0).any():
            sys.exit(f"ERROR: non-positive values in {c}; the log transform would fail.")

    # Validation: stage recomputed from its definition = extraction / extractable resource * 100
    df["stage_recomputed"] = 100 * df["total_extraction_ham"] / df["annual_extractable_gw_ham"]
    df["stage_diff"] = df["stage_recomputed"] - df["stage_of_extraction_pct"]
    bad = int((df["stage_diff"].abs() > STAGE_TOL).sum())
    cl.add("S1.5", "recompute stage and compare with printed value", len(df), len(df),
           f"max |diff| = {df['stage_diff'].abs().max():.4f} pp; rows > {STAGE_TOL} pp: {bad}")

    # Validation: published recharge total vs sum of its four components
    comp = (df["monsoon_recharge_rainfall_ham"] + df["monsoon_recharge_other_sources_ham"]
            + df["nonmonsoon_recharge_rainfall_ham"] + df["nonmonsoon_recharge_other_sources_ham"])
    df["recharge_component_gap_ham"] = comp - df["total_annual_recharge_ham"]
    df["flag_component_gap"] = (df["recharge_component_gap_ham"].abs() > GAP_TOL_HAM).astype(int)
    gap_states = df.loc[df.flag_component_gap == 1, "state"].value_counts().to_dict()
    cl.add("S1.6", "flag rows where components differ from printed total", len(df), len(df),
           f"{int(df.flag_component_gap.sum())} rows {gap_states}; printed totals are kept (they reconcile to the "
           "state table) and the component columns are not used as the denominator of any feature")

    # Derived variables (denominator = printed total recharge)
    tot = df["total_annual_recharge_ham"]
    df["rainfall_share"] = (df["monsoon_recharge_rainfall_ham"] + df["nonmonsoon_recharge_rainfall_ham"]) / tot
    df["monsoon_share"] = (df["monsoon_recharge_rainfall_ham"] + df["monsoon_recharge_other_sources_ham"]) / tot
    df["natural_discharge_ratio"] = df["total_natural_discharges_ham"] / tot
    df["log_total_recharge"] = np.log(tot)
    df["log_stage"] = np.log(df["stage_of_extraction_pct"])
    df["stage_category"] = stage_category(df["stage_of_extraction_pct"]).astype(str)
    for c in ("rainfall_share", "monsoon_share"):
        if not df[c].between(0, 1.0001).all():
            log.warning("WARNING: %s has values outside [0, 1]", c)
    cl.add("S1.7", "derive features and ln(stage)", len(df), len(df),
           "rainfall_share=(monsoon+non-monsoon rainfall recharge)/total; monsoon_share=(monsoon rainfall+monsoon other)/total; "
           "natural_discharge_ratio=natural discharges/total; log_total_recharge=ln(total recharge); log_stage=ln(stage %)")
    cl.add("S1.8", "skewness of the target before/after log transform", len(df), len(df),
           f"stage skew = {stats.skew(df.stage_of_extraction_pct):.2f}; ln(stage) skew = {stats.skew(df.log_stage):.2f}")

    # Flag (do not delete) extreme districts
    df["flag_stage_gt250"] = (df["stage_of_extraction_pct"] > 250).astype(int)
    cl.add("S1.9", "flag districts with stage > 250 %", len(df), len(df),
           "kept in the model; effect tested in the sensitivity analysis: "
           + ", ".join(df.loc[df.flag_stage_gt250 == 1, "district"]))

    # Join state-level rainfall (used only as a sensitivity / Phase 2 candidate; it is shared by all districts of a state)
    rain = pd.read_csv(DATA_DIR / FILES["rainfall"])
    rain["state"] = rain["state_ut"].map(norm_state)
    n_before = len(df)
    df = df.merge(rain[["state", "rainfall_gwy_2024_25_mm"]], on="state", how="left", validate="m:1")
    unmatched = int(df["rainfall_gwy_2024_25_mm"].isna().sum())
    cl.add("S1.10", "join state rainfall (mm, GWY 2024-25)", n_before, len(df), f"unmatched districts: {unmatched}")
    if unmatched:
        sys.exit("ERROR: state names did not match between district file and rainfall file.")

    df = df.reset_index(drop=True)
    cl.add("S1.11", "final analysis table", len(df), len(df), f"{len(df)} districts x {df.shape[1]} columns, {df.state.nunique()} states/UTs")
    return df, cl


def reconcile_with_state_table(df: pd.DataFrame) -> pd.DataFrame | None:
    """Optional QA: do district rows add up to the published state totals?"""
    path = DATA_DIR / FILES["state_totals"]
    if not path.exists():
        return None
    st = pd.read_csv(path)
    st = st[~st["state_ut"].str.contains("Grand Total", case=False, na=False)].copy()
    st["state"] = st["state_ut"].map(norm_state)
    agg = df.groupby("state").agg(rech_ham=("total_annual_recharge_ham", "sum"),
                                  extr_ham=("total_extraction_ham", "sum")).reset_index()
    m = st.merge(agg, on="state", how="left")
    m["recharge_bcm_districts"] = m["rech_ham"] / HAM_PER_BCM
    m["extraction_bcm_districts"] = m["extr_ham"] / HAM_PER_BCM
    m["recharge_diff_bcm"] = m["recharge_bcm_districts"] - m["total_annual_gw_recharge_bcm"]
    m["extraction_diff_bcm"] = m["extraction_bcm_districts"] - m["total_extraction_bcm"]
    out = m[["state_ut", "total_annual_gw_recharge_bcm", "recharge_bcm_districts", "recharge_diff_bcm",
             "total_extraction_bcm", "extraction_bcm_districts", "extraction_diff_bcm"]].round(3)
    out.to_csv(OUT_DIR / "reconciliation_state_vs_district.csv", index=False)
    worst = out.loc[out["recharge_diff_bcm"].abs().idxmax()]
    log.info("Reconciliation district-sum vs state table: largest recharge gap %.3f bcm (%s). "
             "Gaps are expected for states with dropped districts (West Bengal, Tamil Nadu).",
             worst["recharge_diff_bcm"], worst["state_ut"])
    return out


def write_data_dictionary() -> None:
    rows = [
        ("state / district", "CGWB assessment unit identifiers (state labels harmonised to upper case, '&'->'AND')."),
        ("stage_of_extraction_pct", "Stage of groundwater extraction (%) = total extraction / annual extractable resource x 100."),
        ("log_stage", "TARGET. ln(stage_of_extraction_pct); removes the strong right skew."),
        ("rainfall_share", "(monsoon + non-monsoon rainfall recharge) / total annual recharge."),
        ("monsoon_share", "(monsoon rainfall + monsoon other-source recharge) / total annual recharge."),
        ("natural_discharge_ratio", "total natural discharges / total annual recharge."),
        ("log_total_recharge", "ln(total annual recharge in ham) - scale of the district's resource (used in Phase 2)."),
        ("rainfall_gwy_2024_25_mm", "State-level rainfall for the groundwater year 2024-25 (identical for all districts of a state)."),
        ("flag_component_gap", "1 if the four recharge components differ from the printed total by > 1 ham."),
        ("flag_stage_gt250", "1 if stage of extraction > 250 %."),
        ("stage_category", "CGWB class: safe (<=70), semi_critical (70-90], critical (90-100], over_exploited (>100)."),
    ]
    pd.DataFrame(rows, columns=["variable", "definition"]).to_csv(OUT_DIR / "data_dictionary.csv", index=False)


# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================================
def run_eda(df: pd.DataFrame) -> list[str]:
    section("2. EXPLORATORY DATA ANALYSIS")
    eda_cols = ["stage_of_extraction_pct", "log_stage", "rainfall_share", "monsoon_share",
                "natural_discharge_ratio", "log_total_recharge", "rainfall_gwy_2024_25_mm"]
    desc = df[eda_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc["skew"] = [stats.skew(df[c]) for c in eda_cols]
    desc["excess_kurtosis"] = [stats.kurtosis(df[c]) for c in eda_cols]
    desc.round(4).to_csv(OUT_DIR / "descriptive_stats.csv")
    log.info("Descriptive statistics:\n%s", desc[["count", "mean", "std", "min", "50%", "max", "skew", "excess_kurtosis"]].round(3).to_string())

    cat = df["stage_category"].value_counts().reindex(STAGE_LABELS).fillna(0).astype(int)
    cat_tbl = pd.DataFrame({"districts": cat, "percent": (100 * cat / len(df)).round(1)})
    cat_tbl.to_csv(OUT_DIR / "stage_category_counts.csv")
    log.info("CGWB categories:\n%s", cat_tbl.to_string())

    # Histograms
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, c in zip(axes.ravel(), eda_cols[:6]):
        sns.histplot(df[c], kde=True, bins=40, ax=ax, color="#1c7293")
        ax.set_title(f"{c}  (skew {stats.skew(df[c]):.2f})")
        ax.set_xlabel("")
    fig.suptitle("Distributions of the target and candidate predictors (n = %d districts)" % len(df), y=1.01)
    savefig(fig, "eda_histograms.png")

    # Correlation heatmap (Pearson) + Spearman table
    corr = df[eda_cols[1:]].corr()
    corr.round(4).to_csv(OUT_DIR / "correlation_pearson.csv")
    df[eda_cols[1:]].corr(method="spearman").round(4).to_csv(OUT_DIR / "correlation_spearman.csv")
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation matrix (Pearson)")
    savefig(fig, "eda_correlation_heatmap.png")

    # Scatterplots target vs predictors with LOWESS (visual linearity check)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, c in zip(axes, MLR_FEATURES):
        ax.scatter(df[c], df[TARGET], s=11, alpha=0.45, color="#1c7293")
        lw = lowess(df[TARGET], df[c], frac=0.4)
        ax.plot(lw[:, 0], lw[:, 1], color="crimson", lw=2, label="LOWESS")
        r = df[c].corr(df[TARGET])
        ax.set_xlabel(c)
        ax.set_ylabel("ln(stage %)")
        ax.set_title(f"r = {r:.2f}")
        ax.legend()
    savefig(fig, "eda_scatter_target_vs_predictors.png")

    # Distribution of stage by state
    order = df.groupby("state")["stage_of_extraction_pct"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(9, 10))
    sns.boxplot(data=df, y="state", x="stage_of_extraction_pct", order=order, ax=ax, color="#9ecae1", fliersize=2)
    ax.axvline(100, color="crimson", ls="--", label="100 % (extraction = resource)")
    ax.set_xscale("log")
    ax.set_xlabel("stage of extraction (%, log scale)")
    ax.set_ylabel("")
    ax.set_title("Stage of extraction by state / UT (districts)")
    ax.legend(loc="lower right")
    savefig(fig, "eda_stage_by_state.png")

    # Between-state share of variance (motivates the clustering and grouped-CV checks)
    eta2 = float(df.groupby("state")[TARGET].transform("mean").var() / df[TARGET].var())
    log.info("Share of ln(stage) variance that lies BETWEEN states (eta^2): %.3f", eta2)
    top = df.nlargest(10, "stage_of_extraction_pct")[["state", "district", "stage_of_extraction_pct"]]
    top.to_csv(OUT_DIR / "top10_stage_districts.csv", index=False)
    return eda_cols


# =============================================================================
# 3-4. OLS MODELS (SLR and MLR)
# =============================================================================
def fit_ols(df: pd.DataFrame, xcols: list[str], target: str = TARGET, cov: str | None = None, groups=None):
    X = sm.add_constant(df[xcols])
    mdl = sm.OLS(df[target], X)
    if cov is None:
        return mdl.fit()
    if cov == "cluster":
        return mdl.fit(cov_type="cluster", cov_kwds={"groups": groups})
    return mdl.fit(cov_type=cov)


def coefficient_table(df: pd.DataFrame, xcols: list[str]) -> tuple[pd.DataFrame, dict]:
    groups = pd.factorize(df["state"])[0]
    m, m_hc3, m_cl = fit_ols(df, xcols), fit_ols(df, xcols, cov="HC3"), fit_ols(df, xcols, cov="cluster", groups=groups)
    ci = m_hc3.conf_int()
    t = pd.DataFrame({
        "coef": m.params, "se_classical": m.bse, "t_classical": m.tvalues, "p_classical": m.pvalues,
        "se_HC3": m_hc3.bse, "p_HC3": m_hc3.pvalues, "ci95_low_HC3": ci[0], "ci95_high_HC3": ci[1],
        "se_state_cluster": m_cl.bse, "p_state_cluster": m_cl.pvalues,
    })
    t["pct_change_in_stage_per_step"] = np.nan
    t["step"] = np.nan
    for f in xcols:
        t.loc[f, "step"] = EFFECT_STEP.get(f, 0.1)
        t.loc[f, "pct_change_in_stage_per_step"] = 100 * (np.exp(m.params[f] * EFFECT_STEP.get(f, 0.1)) - 1)
    return t.round(5), {"classical": m, "hc3": m_hc3, "cluster": m_cl}


def fit_summary_row(name: str, m, xcols: list[str]) -> dict:
    return dict(model=name, n=int(m.nobs), k_predictors=len(xcols), r2=m.rsquared, adj_r2=m.rsquared_adj,
                f_stat=m.fvalue, f_pvalue=m.f_pvalue, aic=m.aic, bic=m.bic, resid_std_err=np.sqrt(m.scale))


def run_models(df: pd.DataFrame) -> dict:
    section("3-4. SIMPLE AND MULTIPLE LINEAR REGRESSION (OLS)")
    global SLR_FEATURE
    cors = {f: abs(df[f].corr(df[TARGET])) for f in MLR_FEATURES}
    if SLR_FEATURE is None:
        SLR_FEATURE = max(cors, key=cors.get)
    log.info("|r| with %s: %s -> SLR predictor = %s", TARGET, {k: round(float(v), 3) for k, v in cors.items()}, SLR_FEATURE)

    slr_tbl, slr_m = coefficient_table(df, [SLR_FEATURE])
    mlr_tbl, mlr_m = coefficient_table(df, MLR_FEATURES)
    slr_tbl.to_csv(OUT_DIR / "ols_slr_coefficients.csv")
    mlr_tbl.to_csv(OUT_DIR / "ols_mlr_coefficients.csv")
    fits = pd.DataFrame([fit_summary_row("SLR", slr_m["classical"], [SLR_FEATURE]),
                         fit_summary_row("MLR", mlr_m["classical"], MLR_FEATURES)]).round(5)
    fits.to_csv(OUT_DIR / "ols_fit_summary.csv", index=False)
    (OUT_DIR / "ols_slr_summary.txt").write_text(str(slr_m["classical"].summary()) + "\n\nHC3:\n" + str(slr_m["hc3"].summary()), encoding="utf-8")
    (OUT_DIR / "ols_mlr_summary.txt").write_text(str(mlr_m["classical"].summary()) + "\n\nHC3:\n" + str(mlr_m["hc3"].summary()), encoding="utf-8")

    log.info("SLR  ln(stage) ~ %s\n%s", SLR_FEATURE, slr_tbl[["coef", "se_classical", "p_classical", "se_HC3", "p_HC3"]].to_string())
    log.info("MLR  ln(stage) ~ %s\n%s", " + ".join(MLR_FEATURES), mlr_tbl[["coef", "se_classical", "p_classical", "se_HC3", "p_HC3", "p_state_cluster", "pct_change_in_stage_per_step"]].to_string())
    log.info("Fit summary:\n%s", fits.to_string(index=False))

    # F-test: does the MLR improve on the SLR? (nested models)
    f_stat, f_p, df_diff = mlr_m["classical"].compare_f_test(slr_m["classical"])
    log.info("Nested F-test MLR vs SLR: F = %.3f, p = %.3g", f_stat, f_p)
    return {"slr": slr_m, "mlr": mlr_m, "slr_tbl": slr_tbl, "mlr_tbl": mlr_tbl, "fits": fits,
            "nested_f": (float(f_stat), float(f_p))}


# =============================================================================
# 5. ASSUMPTION CHECKS
# =============================================================================
def run_diagnostics(df: pd.DataFrame, mods: dict) -> dict:
    section("5. ASSUMPTION CHECKS (MLR)")
    m = mods["mlr"]["classical"]
    res, fitted = m.resid, m.fittedvalues
    n, p = int(m.nobs), len(MLR_FEATURES)

    bp = het_breuschpagan(res, m.model.exog)
    wh = het_white(res, m.model.exog)
    jb = stats.jarque_bera(res)
    sw = stats.shapiro(res)
    try:
        rs = linear_reset(m, power=2, test_type="fitted", use_f=True)
        reset_stat, reset_p = float(rs.fvalue), float(rs.pvalue)
    except Exception:  # pragma: no cover - older statsmodels
        reset_stat, reset_p = np.nan, np.nan

    Xv = sm.add_constant(df[MLR_FEATURES])
    vif = pd.DataFrame({"variable": MLR_FEATURES,
                        "VIF": [variance_inflation_factor(Xv.values, i + 1) for i in range(p)]})
    vif.round(3).to_csv(OUT_DIR / "vif.csv", index=False)
    cond = float(m.condition_number)

    infl = m.get_influence()
    lev, cooks, stud = infl.hat_matrix_diag, infl.cooks_distance[0], infl.resid_studentized_external
    lev_cut, cook_cut = 2 * (p + 1) / n, 4 / n
    out = df[["state", "district", "stage_of_extraction_pct"] + MLR_FEATURES].copy()
    out["leverage"], out["cooks_d"], out["studentized_resid"] = lev, cooks, stud
    flag = (out.leverage > lev_cut) | (out.cooks_d > cook_cut) | (out.studentized_resid.abs() > 3) | (out.stage_of_extraction_pct > 250)
    infl_tbl = out[flag].sort_values("cooks_d", ascending=False).round(4)
    infl_tbl.to_csv(OUT_DIR / "influential_districts.csv", index=False)

    rows = [
        ("Linearity", "Ramsey RESET (fitted^2)", reset_stat, reset_p, reset_p < 0.05, "p < 0.05: functional form is incomplete (non-linearity)"),
        ("Normality of residuals", "Jarque-Bera", float(jb.statistic), float(jb.pvalue), jb.pvalue < 0.05, f"skew={stats.skew(res):.2f}, excess kurtosis={stats.kurtosis(res):.2f}"),
        ("Normality of residuals", "Shapiro-Wilk", float(sw.statistic), float(sw.pvalue), sw.pvalue < 0.05, "p < 0.05: reject normality"),
        ("Homoscedasticity", "Breusch-Pagan", float(bp[0]), float(bp[1]), bp[1] < 0.05, "p < 0.05: reject constant variance"),
        ("Homoscedasticity", "White", float(wh[0]), float(wh[1]), wh[1] < 0.05, "p < 0.05: reject constant variance"),
        ("Multicollinearity", "max VIF", float(vif.VIF.max()), np.nan, vif.VIF.max() > 5, "VIF > 5 would be a concern"),
        ("Multicollinearity", "condition number", cond, np.nan, cond > 30, "> 30 would be a concern"),
        ("Influence", "districts flagged", float(len(infl_tbl)), np.nan, False,
         f"leverage > {lev_cut:.4f}, Cook's D > {cook_cut:.4f}, |studentized| > 3 or stage > 250"),
    ]
    tests = pd.DataFrame(rows, columns=["assumption", "test", "statistic", "p_value", "violated", "reading"])
    tests["statistic"] = tests["statistic"].astype(float).round(5)
    tests["p_value"] = tests["p_value"].astype(float).round(8)
    tests.to_csv(OUT_DIR / "assumption_tests.csv", index=False)
    log.info("Assumption tests:\n%s", tests[["assumption", "test", "statistic", "p_value", "violated"]].to_string(index=False))
    log.info("VIF:\n%s", vif.round(3).to_string(index=False))
    log.info("Consequence: classical standard errors/p-values are unreliable where heteroskedasticity is present; "
             "HC3 and state-clustered standard errors are reported alongside, and the bootstrap (section 6) is used for inference.")

    # Residual diagnostic panel
    std_res = stud
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax = axes[0, 0]
    ax.scatter(fitted, res, s=11, alpha=0.5, color="#1c7293")
    lw = lowess(res, fitted, frac=0.4)
    ax.plot(lw[:, 0], lw[:, 1], color="crimson", lw=2)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set(title="Residuals vs fitted", xlabel="fitted ln(stage)", ylabel="residual")
    sm.qqplot(res, line="s", ax=axes[0, 1], markersize=3, alpha=0.6)
    axes[0, 1].set_title("Normal Q-Q plot of residuals")
    ax = axes[0, 2]
    sns.histplot(res, bins=40, stat="density", ax=ax, color="#9ecae1")
    xs = np.linspace(res.min(), res.max(), 300)
    ax.plot(xs, stats.norm.pdf(xs, res.mean(), res.std()), color="crimson", lw=2, label="normal fit")
    ax.set(title=f"Residual distribution (skew {stats.skew(res):.2f})", xlabel="residual")
    ax.legend()
    ax = axes[1, 0]
    ax.scatter(fitted, np.sqrt(np.abs(std_res)), s=11, alpha=0.5, color="#1c7293")
    lw = lowess(np.sqrt(np.abs(std_res)), fitted, frac=0.4)
    ax.plot(lw[:, 0], lw[:, 1], color="crimson", lw=2)
    ax.set(title="Scale-location (heteroskedasticity)", xlabel="fitted ln(stage)", ylabel="sqrt |studentized residual|")
    ax = axes[1, 1]
    ax.vlines(np.arange(n), 0, cooks, lw=0.8, color="#1c7293")
    ax.axhline(cook_cut, color="crimson", ls="--", label=f"4/n = {cook_cut:.4f}")
    ax.set(title="Cook's distance", xlabel="district index", ylabel="Cook's D")
    ax.legend()
    ax = axes[1, 2]
    ax.scatter(lev, stud, s=11, alpha=0.5, color="#1c7293")
    ax.axvline(lev_cut, color="crimson", ls="--", label=f"2(p+1)/n = {lev_cut:.4f}")
    ax.axhline(3, color="grey", ls=":")
    ax.axhline(-3, color="grey", ls=":")
    ax.set(title="Leverage vs studentized residual", xlabel="leverage", ylabel="studentized residual")
    ax.legend()
    fig.suptitle("MLR residual diagnostics", y=1.0)
    savefig(fig, "diagnostics_mlr_residuals.png")

    # Partial-regression (added-variable) plots for each predictor
    fig = sm.graphics.plot_partregress_grid(m, fig=plt.figure(figsize=(14, 4.6)))
    savefig(fig, "diagnostics_partial_regression.png")

    return {"bp_p": float(bp[1]), "white_p": float(wh[1]), "jb_p": float(jb.pvalue), "sw_p": float(sw.pvalue),
            "reset_p": reset_p, "resid_skew": float(stats.skew(res)), "resid_excess_kurtosis": float(stats.kurtosis(res)),
            "vif": vif.set_index("variable")["VIF"].round(3).to_dict(), "condition_number": cond,
            "n_influential": int(len(infl_tbl)), "leverage_cut": float(lev_cut), "cooks_cut": float(cook_cut),
            "cooks": cooks}


# =============================================================================
# 6. ESTIMATOR VALIDATION: BOOTSTRAP AND CROSS-VALIDATION
# =============================================================================
def bootstrap_ols(X: np.ndarray, y: np.ndarray, kind: str, rng: np.random.Generator, b: int) -> np.ndarray:
    """kind='pairs'    : resample whole districts (rows) - valid under heteroskedasticity.
       kind='residual' : resample residuals around the fitted line - mimics the textbook (homoskedastic) model."""
    n, k = X.shape
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta_hat
    out = np.empty((b, k))
    for i in range(b):
        if kind == "pairs":
            idx = rng.integers(0, n, n)
            out[i] = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
        else:
            ystar = X @ beta_hat + rng.choice(resid, n, replace=True)
            out[i] = np.linalg.lstsq(X, ystar, rcond=None)[0]
    return out


def run_bootstrap(df: pd.DataFrame, mods: dict) -> pd.DataFrame:
    section(f"6a. BOOTSTRAP VALIDATION OF THE OLS ESTIMATOR (B = {N_BOOT}, seed {SEED})")
    rng = np.random.default_rng(SEED)
    rows, store = [], {}
    for label, xcols, key in (("MLR", MLR_FEATURES, "mlr"), ("SLR", [SLR_FEATURE], "slr")):
        m, m_hc3 = mods[key]["classical"], mods[key]["hc3"]
        X, y = sm.add_constant(df[xcols]).values, df[TARGET].values
        names = ["const"] + xcols
        for kind in ("pairs", "residual"):
            if label == "SLR" and kind == "residual":
                continue
            bs = bootstrap_ols(X, y, kind, rng, N_BOOT)
            store[(label, kind)] = pd.DataFrame(bs, columns=names)
            for j, nm in enumerate(names):
                col = bs[:, j]
                rows.append(dict(
                    model=label, bootstrap=kind, term=nm, ols_estimate=m.params.iloc[j], boot_mean=col.mean(),
                    bias=col.mean() - m.params.iloc[j], bias_pct_of_se=100 * (col.mean() - m.params.iloc[j]) / m.bse.iloc[j],
                    theory_sd_classical=m.bse.iloc[j], theory_sd_HC3=m_hc3.bse.iloc[j], boot_sd=col.std(ddof=1),
                    sd_ratio_vs_classical=col.std(ddof=1) / m.bse.iloc[j], var_ratio_vs_classical=col.var(ddof=1) / m.bse.iloc[j] ** 2,
                    sd_ratio_vs_HC3=col.std(ddof=1) / m_hc3.bse.iloc[j],
                    boot_ci95_low=np.quantile(col, 0.025), boot_ci95_high=np.quantile(col, 0.975),
                    boot_skew=stats.skew(col), boot_excess_kurtosis=stats.kurtosis(col), shapiro_p=stats.shapiro(col).pvalue))
    summ = pd.DataFrame(rows).round(5)
    summ.to_csv(OUT_DIR / "bootstrap_summary.csv", index=False)
    store[("MLR", "pairs")].to_csv(OUT_DIR / "bootstrap_coefficients_mlr_pairs.csv", index=False)
    store[("MLR", "residual")].to_csv(OUT_DIR / "bootstrap_coefficients_mlr_residual.csv", index=False)
    store[("SLR", "pairs")].to_csv(OUT_DIR / "bootstrap_coefficients_slr_pairs.csv", index=False)
    log.info("Bootstrap vs theory (MLR):\n%s", summ[summ.model == "MLR"][["bootstrap", "term", "ols_estimate", "boot_mean", "theory_sd_classical", "theory_sd_HC3", "boot_sd", "sd_ratio_vs_classical", "sd_ratio_vs_HC3"]].to_string(index=False))
    log.info("Reading: means agree (OLS is unbiased); the residual bootstrap reproduces the classical SD, but the pairs "
             "bootstrap is larger and matches HC3 - the classical SEs understate the real sampling variability here.")

    # Figures: histogram + KDE + two normal overlays (classical theory and HC3 theory)
    def hist_panel(ax, col, est, sd_c, sd_h, title):
        sns.histplot(col, stat="density", bins=40, ax=ax, color="#9ecae1", edgecolor="white")
        sns.kdeplot(col, ax=ax, color="#08306b", lw=1.5, label="bootstrap KDE")
        xs = np.linspace(col.min(), col.max(), 400)
        ax.plot(xs, stats.norm.pdf(xs, est, sd_c), color="crimson", lw=2, label="theory N(b, classical SE)")
        ax.plot(xs, stats.norm.pdf(xs, est, sd_h), color="darkgreen", lw=2, ls="--", label="theory N(b, HC3 SE)")
        ax.axvline(est, color="k", lw=1, ls=":")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("density")

    for label, kind, fname in (("MLR", "pairs", "bootstrap_mlr_pairs_normal_overlay.png"),
                               ("MLR", "residual", "bootstrap_mlr_residual_normal_overlay.png"),
                               ("SLR", "pairs", "bootstrap_slr_pairs_normal_overlay.png")):
        d = store[(label, kind)]
        key = label.lower()
        m, m_hc3 = mods[key]["classical"], mods[key]["hc3"]
        cols = list(d.columns)
        ncols = 2
        nrows = int(np.ceil(len(cols) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 3.6 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for i, c in enumerate(cols):
            row = summ[(summ.model == label) & (summ.bootstrap == kind) & (summ.term == c)].iloc[0]
            hist_panel(axes[i], d[c], m.params[c], m.bse[c], m_hc3.bse[c],
                       f"{c}: boot SD / classical SD = {row.sd_ratio_vs_classical:.2f}; / HC3 SD = {row.sd_ratio_vs_HC3:.2f}")
        for j in range(len(cols), len(axes)):
            axes[j].axis("off")
        axes[0].legend(fontsize=7.5)
        fig.suptitle(f"{label}: {kind} bootstrap of OLS coefficients (B = {N_BOOT}) vs theoretical sampling distribution", y=1.0)
        savefig(fig, fname)

    # SD comparison bar chart
    mm = summ[(summ.model == "MLR") & (summ.term != "const")]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    terms = mm.term.unique()
    w = 0.2
    xs = np.arange(len(terms))
    ax.bar(xs - 1.5 * w, mm[mm.bootstrap == "pairs"].theory_sd_classical, w, label="theory: classical SE", color="crimson")
    ax.bar(xs - 0.5 * w, mm[mm.bootstrap == "pairs"].theory_sd_HC3, w, label="theory: HC3 SE", color="darkgreen")
    ax.bar(xs + 0.5 * w, mm[mm.bootstrap == "residual"].boot_sd, w, label="residual bootstrap SD", color="#9ecae1")
    ax.bar(xs + 1.5 * w, mm[mm.bootstrap == "pairs"].boot_sd, w, label="pairs bootstrap SD", color="#08306b")
    ax.set_xticks(xs)
    ax.set_xticklabels(terms, rotation=10)
    ax.set_ylabel("standard deviation of coefficient")
    ax.set_title("Sampling variability of the MLR coefficients: theory vs bootstrap")
    ax.legend(fontsize=8)
    savefig(fig, "bootstrap_sd_comparison.png")
    return summ


def cv_ols(df: pd.DataFrame, xcols: list[str] | None, splits: list, label: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Fit OLS (or the mean-only baseline when xcols is None) on each training fold and score the test fold."""
    y = df[TARGET].values
    oof = np.full(len(df), np.nan)
    rows = []
    for fold, (tr, te) in enumerate(splits):
        if xcols is None:
            pred = np.full(len(te), y[tr].mean())
            k = 0
        else:
            mod = sm.OLS(y[tr], sm.add_constant(df.iloc[tr][xcols])).fit()
            pred = mod.predict(sm.add_constant(df.iloc[te][xcols], has_constant="add")).values
            k = len(xcols)
        oof[te] = pred
        rows.append({"model": label, "fold": fold, **regression_metrics(y[te], pred, k)})
    return pd.DataFrame(rows), oof


def run_cv(df: pd.DataFrame) -> dict:
    section(f"6b. {K_FOLDS}-FOLD CROSS-VALIDATION OF THE OLS MODELS")
    kf = list(KFold(K_FOLDS, shuffle=True, random_state=SEED).split(df))
    gkf = list(GroupKFold(K_FOLDS).split(df, groups=df["state"]))
    specs = [("Mean-only baseline", None), ("SLR", [SLR_FEATURE]), ("MLR", MLR_FEATURES)]
    out = {}
    for tag, splits in (("random_kfold", kf), ("grouped_by_state", gkf)):
        fold_frames, rows = [], []
        for name, xc in specs:
            fm, oof = cv_ols(df, xc, splits, name)
            fold_frames.append(fm)
            pooled = regression_metrics(df[TARGET].values, oof, 0 if xc is None else len(xc))
            row = {"model": name, "split": tag}
            for c in ("rmse_log", "mae_log", "r2", "adj_r2", "rmse_stage", "mae_stage"):
                row[c + "_mean"], row[c + "_sd"] = fm[c].mean(), fm[c].std(ddof=1)
            row["pooled_oof_r2"] = pooled["r2"]
            rows.append(row)
            out[(tag, name)] = oof
        pd.concat(fold_frames).round(5).to_csv(OUT_DIR / f"cv_fold_metrics_{tag}.csv", index=False)
        tbl = pd.DataFrame(rows).round(4)
        tbl.to_csv(OUT_DIR / f"cv_summary_{tag}.csv", index=False)
        log.info("CV (%s):\n%s", tag, tbl[["model", "rmse_log_mean", "rmse_log_sd", "mae_log_mean", "r2_mean", "adj_r2_mean", "rmse_stage_mean", "pooled_oof_r2"]].to_string(index=False))
        out[tag] = tbl
    log.info("Note: under state-grouped CV the R2 is much lower than under random CV - a model fitted to some states does "
             "not transfer well to unseen states. This is why both are reported.")

    # Predicted vs actual (out-of-fold) for the MLR
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, tag in zip(axes, ("random_kfold", "grouped_by_state")):
        ax.scatter(df[TARGET], out[(tag, "MLR")], s=10, alpha=0.45, color="#1c7293")
        lim = [df[TARGET].min() - 0.2, df[TARGET].max() + 0.2]
        ax.plot(lim, lim, "r--")
        ax.set(xlabel="actual ln(stage)", ylabel="out-of-fold prediction", title=f"MLR, {tag.replace('_', ' ')}")
    savefig(fig, "cv_mlr_actual_vs_predicted.png")
    return out


# =============================================================================
# 7. SENSITIVITY AND ROBUSTNESS
# =============================================================================
def run_sensitivity(df: pd.DataFrame, mods: dict, diag: dict) -> pd.DataFrame:
    section("7a. SENSITIVITY OF THE MLR TO MODELLING CHOICES")
    rows = []

    def add(name: str, d: pd.DataFrame, xcols: list[str], target: str = TARGET, note: str = "") -> None:
        m = fit_ols(d, xcols, target)
        h = fit_ols(d, xcols, target, cov="HC3")
        row = dict(spec=name, note=note, n=int(m.nobs), r2=m.rsquared, adj_r2=m.rsquared_adj)
        for f in MLR_FEATURES:
            row[f"b_{f}"] = m.params.get(f, np.nan)
            row[f"pHC3_{f}"] = h.pvalues.get(f, np.nan)
        rows.append(row)

    add("Main: ln(stage) ~ 3 shares", df, MLR_FEATURES)
    add("Drop stage > 250 %", df[df.stage_of_extraction_pct <= 250], MLR_FEATURES, note="removes the extreme districts")
    keep = diag["cooks"] <= diag["cooks_cut"]
    add("Drop Cook's D > 4/n", df[keep], MLR_FEATURES, note=f"removes {int((~keep).sum())} influential districts")
    d2 = df.copy()
    add("Raw stage (no log)", d2, MLR_FEATURES, target="stage_of_extraction_pct", note="coefficients on a different scale")
    add("+ state rainfall (mm)", df, MLR_FEATURES + ["rainfall_gwy_2024_25_mm"], note="state-level variable shared by all districts of a state")
    add("+ log total recharge", df, MLR_FEATURES + ["log_total_recharge"], note="adds district resource scale")
    # State fixed effects
    fe = pd.get_dummies(df["state"], drop_first=True, dtype=float)
    d_fe = pd.concat([df[[TARGET] + MLR_FEATURES], fe], axis=1)
    add("+ state fixed effects", d_fe, MLR_FEATURES + list(fe.columns), note="absorbs all between-state differences")
    tbl = pd.DataFrame(rows).round(5)
    tbl.to_csv(OUT_DIR / "sensitivity_models.csv", index=False)
    log.info("Sensitivity:\n%s", tbl[["spec", "n", "r2", "adj_r2"] + [f"b_{f}" for f in MLR_FEATURES]].to_string(index=False))
    fe_row = tbl[tbl.spec == "+ state fixed effects"].iloc[0]
    log.info("Reading: dropping outliers or adding scale variables changes magnitudes but not signs. Adding state fixed "
             "effects lifts R2 to %.2f and the monsoon_share and natural_discharge_ratio coefficients change sign: most of "
             "the national association is BETWEEN states, and the three shares carry little WITHIN-state information.", fe_row.r2)
    return tbl


def run_replication_2024(df: pd.DataFrame, mods: dict) -> dict | None:
    """Robustness only (never mixed into the main model): does the 2024 assessment tell the same story?"""
    path = DATA_DIR / FILES["prev_year"]
    if not path.exists():
        return None
    section("7b. ROBUSTNESS: 2024 ASSESSMENT vs 2025")
    o = pd.read_csv(path)
    o["state"] = o["state"].map(norm_state)
    o["key"] = o["district"].str.lower().str.strip()
    d25 = df.copy()
    d25["key"] = d25["district"].str.lower().str.strip()
    mm = d25.merge(o[["state", "key", "stage_of_extraction_pct"]].rename(columns={"stage_of_extraction_pct": "stage_2024"}),
                   on=["state", "key"], how="inner")
    mm["change"] = mm["stage_of_extraction_pct"] - mm["stage_2024"]
    info = dict(matched=len(mm), pearson=mm[["stage_2024", "stage_of_extraction_pct"]].corr().iloc[0, 1],
                spearman=mm[["stage_2024", "stage_of_extraction_pct"]].corr(method="spearman").iloc[0, 1],
                mean_change=mm.change.mean(), median_change=mm.change.median(),
                share_over_100_2025=(mm.stage_of_extraction_pct > 100).mean(), share_over_100_2024=(mm.stage_2024 > 100).mean())

    # Re-estimate the same MLR on the 2024 table
    tot = o["total_annual_recharge_ham"]
    o["rainfall_share"] = (o["monsoon_recharge_rainfall_ham"] + o["nonmonsoon_recharge_rainfall_ham"]) / tot
    o["monsoon_share"] = (o["monsoon_recharge_rainfall_ham"] + o["monsoon_recharge_other_ham"]) / tot
    o["natural_discharge_ratio"] = o["total_natural_discharges_ham"] / tot
    o = o[(o.stage_of_extraction_pct > 0) & (tot > 0)].replace([np.inf, -np.inf], np.nan).dropna(subset=MLR_FEATURES + ["stage_of_extraction_pct"])
    o[TARGET] = np.log(o["stage_of_extraction_pct"])
    m24 = fit_ols(o, MLR_FEATURES)
    m25 = mods["mlr"]["classical"]
    comp = pd.DataFrame({"coef_2025": m25.params, "coef_2024": m24.params})
    comp["same_sign"] = np.sign(comp.coef_2025) == np.sign(comp.coef_2024)
    comp.round(4).to_csv(OUT_DIR / "robustness_coefficients_2024_vs_2025.csv")
    pd.Series(info).round(4).to_csv(OUT_DIR / "robustness_2024_vs_2025.csv", header=["value"])
    info["r2_2024_model"] = float(m24.rsquared)
    info["n_2024_model"] = int(m24.nobs)
    log.info("Stage 2024 vs 2025: matched %d districts, Pearson r = %.3f, Spearman = %.3f", info["matched"], info["pearson"], info["spearman"])
    log.info("MLR coefficients 2025 vs 2024:\n%s", comp.round(4).to_string())
    return {k: float(v) for k, v in info.items()}


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    setup_logging()
    check_inputs()
    df, cl = load_and_clean()
    reconcile_with_state_table(df)
    write_data_dictionary()
    df.to_csv(OUT_DIR / "cleaned_dataset.csv", index=False)
    cl.frame().to_csv(OUT_DIR / "cleaning_log.csv", index=False)

    run_eda(df)
    mods = run_models(df)
    diag = run_diagnostics(df, mods)
    boot = run_bootstrap(df, mods)
    cv = run_cv(df)
    sens = run_sensitivity(df, mods, diag)
    rep24 = run_replication_2024(df, mods)

    m = mods["mlr"]["classical"]
    results = {
        "n_districts": len(df), "n_states": int(df.state.nunique()), "seed": SEED, "n_boot": N_BOOT, "k_folds": K_FOLDS,
        "slr_feature": SLR_FEATURE, "slr_r2": float(mods["slr"]["classical"].rsquared),
        "mlr_r2": float(m.rsquared), "mlr_adj_r2": float(m.rsquared_adj), "mlr_aic": float(m.aic), "mlr_bic": float(m.bic),
        "mlr_params": m.params.round(4).to_dict(), "mlr_p_classical": m.pvalues.round(5).to_dict(),
        "mlr_p_HC3": mods["mlr"]["hc3"].pvalues.round(5).to_dict(), "mlr_p_cluster": mods["mlr"]["cluster"].pvalues.round(5).to_dict(),
        "nested_F_test_mlr_vs_slr": {"F": mods["nested_f"][0], "p": mods["nested_f"][1]},
        "diagnostics": {k: v for k, v in diag.items() if k != "cooks"},
        "cv_random": cv["random_kfold"].set_index("model")[["rmse_log_mean", "r2_mean"]].to_dict("index"),
        "cv_grouped": cv["grouped_by_state"].set_index("model")[["rmse_log_mean", "r2_mean"]].to_dict("index"),
        "robustness_2024": rep24,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=to_jsonable), encoding="utf-8")

    section("PHASE 1 COMPLETE")
    log.info("Final MLR: ln(stage) = %.4f %+.4f*rainfall_share %+.4f*monsoon_share %+.4f*natural_discharge_ratio   (R2 = %.3f)",
             m.params["const"], m.params["rainfall_share"], m.params["monsoon_share"], m.params["natural_discharge_ratio"], m.rsquared)
    log.info("All outputs are in: %s", OUT_DIR)
    log.info("Next step: run  python phase2_model_selection.py")


if __name__ == "__main__":
    main()