#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSBC Synthetic Customer Segmentation & Persona Engine
=====================================================
End-to-end, reproducible pipeline:

  1. Load data from Excel (path + sheet from .env)
  2. Data-quality audit: dtypes, missing tokens, dirty values, invalid codes,
     duplicates, logical-consistency checks
  3. Missing-value handling: benchmark of 3 imputers (median/mode, KNN,
     Iterative/MICE) via masked-cell simulation -> best one applied
  4. Outlier analysis: IQR (raw & log), robust MAD z-score, Isolation Forest
  5. Outlier-handling / preprocessing benchmark (7 strategies) evaluated by
     silhouette, Davies-Bouldin and bootstrap stability -> best one chosen
  6. Clustering tendency (Hopkins statistic)
  7. K-Means: k selection with Elbow, Silhouette, Calinski-Harabasz,
     Davies-Bouldin, Gap statistic, bootstrap stability -> rank aggregation
  8. Hierarchical (Ward / average / complete): cophenetic correlation,
     dendrogram, k selection, kNN propagation to the full population
  9. Advanced composite method: FAMD (Factor Analysis of Mixed Data)
     embedding + Gaussian Mixture Model (BIC/AIC, soft probabilities)
 10. Ensemble: Evidence-Accumulation co-association consensus
     (average / ward) + one-hot meta-clustering, selected by ANMI
 11. Statistical validation: internal indices, permutation null test,
     cluster-wise bootstrap Jaccard, ARI/NMI agreement, Random-Forest
     separability, Kruskal-Wallis / Chi-square with effect sizes + BH-FDR
 12. Profiling, automatic persona generation, figures, reports
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from scipy import stats
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (adjusted_rand_score, balanced_accuracy_score,
                             calinski_harabasz_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_samples,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import (PowerTransformer, QuantileTransformer,
                                   RobustScaler, StandardScaler)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# CONSTANTS
# =============================================================================
AGE_BANDS = ["18-24", "25-29", "30-34"]
AGE_MAP = {b: i for i, b in enumerate(AGE_BANDS)}
NUM_COLS = ["INCOME", "TRB"]
CHANNEL_COLS = ["DIG_ACTIVE", "FX_TRANS", "PAYME"]
PRODUCT_COLS = ["CC", "LN", "MT", "TD", "SC", "SP", "BD", "MPF"]
BIN_COLS = CHANNEL_COLS + PRODUCT_COLS
EXPECTED_COLS = ["AGE"] + NUM_COLS + BIN_COLS
IMP_COLS = ["AGE_ORD", "LOG_INCOME", "LOG_TRB"] + BIN_COLS
DISCRETE_IMP = ["AGE_ORD"] + BIN_COLS
MODEL_FEATURES = ["AGE_ORD", "LOG_INCOME", "LOG_TRB"] + BIN_COLS

MISSING_TOKENS = ["", "nan", "NaN", "NAN", "None", "none", "NULL", "null", "N/A",
                  "n/a", "NA", "na", "-", "--", "?", "#N/A", "#VALUE!", "missing"]
TRUE_TOKENS = {"Y", "YES", "TRUE", "T"}
FALSE_TOKENS = {"N", "NO", "FALSE", "F"}

FEATURE_LABELS = {
    "DIG_ACTIVE": "Digital banking", "FX_TRANS": "FX transaction", "PAYME": "PayMe",
    "CC": "Credit card", "LN": "Personal loan", "MT": "Mortgage", "TD": "Term deposit",
    "SC": "Securities", "SP": "Structured products", "BD": "Bonds", "MPF": "MPF",
    "INVESTOR": "Any investment (SC/SP/BD)", "CREDIT_HOLDER": "Any loan/mortgage",
}

FIG_DIR: Path = Path("outputs/figures")
REP_DIR: Path = Path("outputs/reports")
DATA_DIR: Path = Path("outputs/data")
log = logging.getLogger("segmentation")


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class Config:
    data_path: str
    sheet_name: str
    output_dir: Path
    random_state: int
    k_min: int
    k_max: int
    min_cluster_share: float
    sample_size: int
    hc_sample: int
    stability_sample: int
    consensus_sample: int
    n_bootstrap: int
    n_null: int
    binary_weight: float
    famd_var_target: float
    n_jobs: int

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        dp = os.getenv("DATA_PATH")
        if not dp:
            raise ValueError("DATA_PATH is not set in .env")
        g = os.getenv
        return cls(
            data_path=dp,
            sheet_name=g("SHEET_NAME", "Synthetic Data"),
            output_dir=Path(g("OUTPUT_DIR", "outputs")),
            random_state=int(g("RANDOM_STATE", 42)),
            k_min=max(2, int(g("K_MIN", 3))),
            k_max=int(g("K_MAX", 10)),
            min_cluster_share=float(g("MIN_CLUSTER_SHARE", 0.03)),
            sample_size=int(g("SAMPLE_SIZE", 8000)),
            hc_sample=int(g("HC_SAMPLE", 6000)),
            stability_sample=int(g("STABILITY_SAMPLE", 20000)),
            consensus_sample=int(g("CONSENSUS_SAMPLE", 5000)),
            n_bootstrap=int(g("N_BOOTSTRAP", 20)),
            n_null=int(g("N_NULL", 20)),
            binary_weight=float(g("BINARY_WEIGHT", 1.0)),
            famd_var_target=float(g("FAMD_VAR_TARGET", 0.85)),
            n_jobs=int(g("N_JOBS", -1)),
        )


def setup(cfg: Config) -> None:
    global FIG_DIR, REP_DIR, DATA_DIR
    FIG_DIR = cfg.output_dir / "figures"
    REP_DIR = cfg.output_dir / "reports"
    DATA_DIR = cfg.output_dir / "data"
    for d in (FIG_DIR, REP_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(cfg.output_dir / "run.log", mode="w", encoding="utf-8")],
        force=True,
    )
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 110


# =============================================================================
# UTILITIES
# =============================================================================
def savefig(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"   figure saved -> {name}.png")


def jsonify(o):
    if isinstance(o, dict):
        return {str(k): jsonify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonify(v) for v in o]
    if isinstance(o, pd.Series):
        return jsonify(o.to_dict())
    if isinstance(o, pd.DataFrame):
        return jsonify(o.to_dict(orient="records"))
    if isinstance(o, np.ndarray):
        return jsonify(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (np.isnan(o) or np.isinf(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


def save_json(obj, name: str) -> None:
    with open(REP_DIR / name, "w", encoding="utf-8") as f:
        json.dump(jsonify(obj), f, indent=2, ensure_ascii=False)


def relabel_by_size(lab: np.ndarray) -> np.ndarray:
    counts = pd.Series(lab).value_counts()
    mapping = {old: new for new, old in enumerate(counts.index)}
    return pd.Series(lab).map(mapping).values.astype(int)


def min_share(lab: np.ndarray) -> float:
    _, c = np.unique(lab, return_counts=True)
    return float(c.min() / len(lab))


def knee_point(ks, values, direction: str = "decreasing") -> int:
    """Kneedle-style knee: max distance between normalised curve and diagonal."""
    x = np.asarray(ks, float)
    y = np.asarray(values, float)
    xn = (x - x.min()) / (x.max() - x.min() + 1e-12)
    yn = (y - y.min()) / (y.max() - y.min() + 1e-12)
    if direction == "decreasing":
        yn = 1 - yn
    return int(x[np.argmax(yn - xn)])


def composite_rank(df: pd.DataFrame, higher: list, lower: list, valid_col: str | None = None) -> pd.Series:
    r = pd.DataFrame(index=df.index)
    for c in higher:
        r[c] = df[c].rank(ascending=False)
    for c in lower:
        r[c] = df[c].rank(ascending=True)
    score = r.mean(axis=1)
    if valid_col is not None:
        score[~df[valid_col].astype(bool)] = np.inf
    return score


def bh_adjust(p) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


# =============================================================================
# 1. LOAD
# =============================================================================
def load_data(cfg: Config) -> pd.DataFrame:
    path = Path(cfg.data_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    log.info(f"Loading {path}")
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        xls = pd.ExcelFile(path)
        sheets = {s.strip().lower(): s for s in xls.sheet_names}
        key = cfg.sheet_name.strip().lower()
        if key not in sheets:
            raise ValueError(f"Sheet '{cfg.sheet_name}' not found. Available: {xls.sheet_names}")
        df = pd.read_excel(xls, sheet_name=sheets[key], dtype=object)
        log.info(f"   sheet used: '{sheets[key]}' (available: {xls.sheet_names})")
    else:
        df = pd.read_csv(path, dtype=object)
    df.columns = [re.sub(r"\s+", "_", str(c).strip().upper()) for c in df.columns]
    df = df.dropna(axis=1, how="all")
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Found: {list(df.columns)}")
    extra = [c for c in df.columns if c not in EXPECTED_COLS]
    if extra:
        log.info(f"   ignoring extra columns: {extra}")
    log.info(f"   raw shape: {df.shape}")
    return df[EXPECTED_COLS].copy()


# =============================================================================
# 2. AUDIT & CLEAN
# =============================================================================
def normalise_age(s: pd.Series) -> pd.Series:
    def f(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        t = str(v).strip().lower()
        t = re.sub(r"\s*(to|–|—|~|_)\s*", "-", t).replace(" ", "")
        if t in AGE_MAP:
            return t
        try:
            a = float(t)
            if 18 <= a <= 24:
                return "18-24"
            if 25 <= a <= 29:
                return "25-29"
            if 30 <= a <= 34:
                return "30-34"
        except ValueError:
            pass
        return np.nan
    return s.map(f)


def coerce_numeric(s: pd.Series) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    needs = num.isna() & s.notna()
    if needs.any():
        cleaned = (s[needs].astype(str).str.strip()
                   .str.replace(r"(?i)hkd|hk\$|\$|,|\s", "", regex=True)
                   .str.replace(r"^\((.*)\)$", r"-\1", regex=True))
        num.loc[needs] = pd.to_numeric(cleaned, errors="coerce")
    return num.astype(float)


def coerce_binary(s: pd.Series) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[num == 1] = 1.0
    out[num == 0] = 0.0
    txt = s.astype(str).str.strip().str.upper()
    out[out.isna() & txt.isin(TRUE_TOKENS)] = 1.0
    out[out.isna() & txt.isin(FALSE_TOKENS)] = 0.0
    return out


def audit_and_clean(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log.info("STEP 2 | Data-quality audit & cleaning")
    rep: dict = {"shape_raw": list(df_raw.shape),
                 "missing_native": df_raw.isna().sum().to_dict()}
    raw = df_raw.copy()
    for c in raw.columns:
        raw[c] = raw[c].map(lambda v: v.strip() if isinstance(v, str) else v)
        raw[c] = raw[c].replace(MISSING_TOKENS, np.nan)
    rep["missing_after_token_normalisation"] = raw.isna().sum().to_dict()

    df = pd.DataFrame(index=raw.index)
    dirty: dict = {}
    df["AGE"] = normalise_age(raw["AGE"])
    dirty["AGE"] = {"invalid_codes_set_to_nan": int((raw["AGE"].notna() & df["AGE"].isna()).sum()),
                    "raw_unique_sample": list(map(str, pd.unique(raw["AGE"].dropna())[:15]))}
    for c in NUM_COLS:
        v = coerce_numeric(raw[c])
        unparsable = int((raw[c].notna() & v.isna()).sum())
        n_inf = int(np.isinf(v).sum())
        v = v.replace([np.inf, -np.inf], np.nan)
        n_neg = int((v < 0).sum())
        v = v.mask(v < 0)
        dirty[c] = {"unparsable_to_nan": unparsable, "infinite_to_nan": n_inf,
                    "negative_to_nan": n_neg, "zeros": int((v == 0).sum())}
        df[c] = v
    for c in BIN_COLS:
        v = coerce_binary(raw[c])
        dirty[c] = {"invalid_codes_set_to_nan": int((raw[c].notna() & v.isna()).sum())}
        df[c] = v
    rep["dirty_values"] = dirty

    all_na = df.isna().all(axis=1)
    heavy_na = df.isna().mean(axis=1) > 0.5
    rep["rows_all_missing_dropped"] = int(all_na.sum())
    rep["rows_gt50pct_missing_dropped"] = int((heavy_na & ~all_na).sum())
    df = df[~heavy_na]

    n_dup = int(df.duplicated().sum())
    rep["exact_duplicates_dropped"] = n_dup
    df = df.drop_duplicates().reset_index(drop=True)
    df["AGE_ORD"] = df["AGE"].map(AGE_MAP)
    rep["missing_after_coercion"] = df.isna().sum().to_dict()
    rep["missing_pct_after_coercion"] = (df.isna().mean() * 100).round(3).to_dict()

    # logical-consistency checks (flag only, not modified)
    inv = df[["TD", "SC", "SP", "BD"]].max(axis=1)
    rep["consistency_flags"] = {
        "TRB_zero_but_holds_deposit_or_investment": int(((df["TRB"] == 0) & (inv == 1)).sum()),
        "mortgage_holder_aged_18_24": int(((df["AGE"] == "18-24") & (df["MT"] == 1)).sum()),
        "zero_income_with_loan_or_mortgage": int(((df["INCOME"] == 0) & (df[["LN", "MT"]].max(axis=1) == 1)).sum()),
    }
    rep["shape_clean_pre_imputation"] = list(df.shape)
    for k, v in rep["dirty_values"].items():
        log.info(f"   {k:<10} {v if k != 'AGE' else {kk: vv for kk, vv in v.items() if kk != 'raw_unique_sample'}}")
    log.info(f"   duplicates dropped: {n_dup} | consistency flags: {rep['consistency_flags']}")
    return df, rep


def plot_missing(df_raw: pd.DataFrame, df: pd.DataFrame) -> None:
    m = pd.DataFrame({"native NaN (%)": df_raw.isna().mean() * 100,
                      "after dirty-value coercion (%)": df[EXPECTED_COLS].isna().mean() * 100})
    fig, ax = plt.subplots(figsize=(11, 4.5))
    m.plot.bar(ax=ax, color=["#9ecae1", "#de2d26"])
    ax.set_ylabel("% missing")
    ax.set_title("Missing values per field (raw vs after cleaning dirty values)")
    savefig(fig, "01_missing_values")


# =============================================================================
# 3. IMPUTATION BENCHMARK
# =============================================================================
def make_imp_matrix(df: pd.DataFrame) -> pd.DataFrame:
    M = pd.DataFrame(index=df.index)
    M["AGE_ORD"] = df["AGE_ORD"]
    M["LOG_INCOME"] = np.log1p(df["INCOME"])
    M["LOG_TRB"] = np.log1p(df["TRB"])
    for c in BIN_COLS:
        M[c] = df[c]
    return M.astype(float)


def impute_matrix(train: pd.DataFrame, apply: pd.DataFrame, method: str, rs: int) -> pd.DataFrame:
    if method == "median_mode":
        fill = {c: (train[c].mode(dropna=True).iloc[0] if c in DISCRETE_IMP else train[c].median())
                for c in train.columns}
        return apply.fillna(fill)
    scaler = StandardScaler().fit(train)  # NaN-aware
    ztr, zap = scaler.transform(train), scaler.transform(apply)
    imp = (KNNImputer(n_neighbors=10) if method == "knn"
           else IterativeImputer(max_iter=15, random_state=rs))
    imp.fit(ztr)
    out = scaler.inverse_transform(imp.transform(zap))
    return pd.DataFrame(out, columns=apply.columns, index=apply.index)


def postprocess_imputed(M: pd.DataFrame) -> pd.DataFrame:
    M = M.copy()
    M["AGE_ORD"] = M["AGE_ORD"].round().clip(0, 2)
    for c in BIN_COLS:
        M[c] = M[c].round().clip(0, 1)
    M["LOG_INCOME"] = M["LOG_INCOME"].clip(lower=0)
    M["LOG_TRB"] = M["LOG_TRB"].clip(lower=0)
    return M


def imputation_benchmark(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, str]:
    """Mask 5% of known cells in complete rows, impute, and score the recovery."""
    log.info("STEP 3 | Imputation benchmark (masked-cell simulation)")
    rng = np.random.default_rng(cfg.random_state)
    complete = make_imp_matrix(df).dropna()
    n = min(4000, len(complete))
    rows = []
    for rep in range(3):
        S = complete.sample(n, random_state=cfg.random_state + rep)
        mask = rng.random(S.shape) < 0.05
        Sm = S.mask(mask)
        for method in ["median_mode", "knn", "iterative"]:
            t0 = time.time()
            imp = postprocess_imputed(impute_matrix(Sm, Sm, method, cfg.random_state))
            row = {"repeat": rep, "method": method}
            nrmse, bacc = [], []
            for j, c in enumerate(S.columns):
                m = mask[:, j]
                if m.sum() < 5:
                    continue
                truth, pred = S[c].values[m], imp[c].values[m]
                if c in DISCRETE_IMP:
                    val = balanced_accuracy_score(truth, pred) if len(np.unique(truth)) > 1 else np.mean(truth == pred)
                    row[f"bal_acc_{c}"] = val
                    bacc.append(val)
                else:
                    val = np.sqrt(np.mean((truth - pred) ** 2)) / (S[c].std() + 1e-12)
                    row[f"nrmse_{c}"] = val
                    nrmse.append(val)
            row["mean_nrmse"] = np.mean(nrmse)
            row["mean_balanced_acc"] = np.mean(bacc)
            row["score_lower_better"] = row["mean_nrmse"] + (1 - row["mean_balanced_acc"])
            row["seconds"] = time.time() - t0
            rows.append(row)
    res = pd.DataFrame(rows)
    summ = res.groupby("method")[["mean_nrmse", "mean_balanced_acc", "score_lower_better", "seconds"]].agg(["mean", "std"])
    best = summ[("score_lower_better", "mean")].idxmin()
    res.to_csv(REP_DIR / "imputation_benchmark_detail.csv", index=False)
    summ.to_csv(REP_DIR / "imputation_benchmark_summary.csv")
    log.info(f"\n{summ.round(4)}\n   -> best imputer: {best}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.barplot(data=res, x="method", y="mean_nrmse", ax=ax[0], errorbar="sd", palette="Blues")
    ax[0].set_title("Continuous recovery: NRMSE (lower = better)")
    sns.barplot(data=res, x="method", y="mean_balanced_acc", ax=ax[1], errorbar="sd", palette="Greens")
    ax[1].set_title("Categorical recovery: balanced accuracy (higher = better)")
    ax[1].set_ylim(0, 1)
    savefig(fig, "02_imputation_benchmark")
    return summ, best


def apply_imputation(df: pd.DataFrame, method: str, cfg: Config) -> tuple[pd.DataFrame, dict]:
    M = make_imp_matrix(df)
    miss_rows = M.isna().any(axis=1)
    info = {"method": method, "rows_with_missing": int(miss_rows.sum()),
            "cells_imputed": M.isna().sum().to_dict()}
    out = df.copy()
    if miss_rows.any():
        train = M.sample(min(10000, len(M)), random_state=cfg.random_state) if method == "knn" else M
        filled = postprocess_imputed(impute_matrix(train, M.loc[miss_rows], method, cfg.random_state))
        for c in IMP_COLS:
            na_idx = M.index[M[c].isna()]
            if len(na_idx) == 0:
                continue
            vals = filled.loc[na_idx, c]
            if c == "LOG_INCOME":
                out.loc[na_idx, "INCOME"] = np.expm1(vals)
            elif c == "LOG_TRB":
                out.loc[na_idx, "TRB"] = np.expm1(vals)
            elif c == "AGE_ORD":
                out.loc[na_idx, "AGE_ORD"] = vals
                out.loc[na_idx, "AGE"] = vals.astype(int).map(dict(enumerate(AGE_BANDS)))
            else:
                out.loc[na_idx, c] = vals
        log.info(f"   imputed {int(miss_rows.sum())} rows with '{method}'")
    else:
        log.info("   no missing values present -> nothing imputed (benchmark kept for documentation)")
    out["AGE_ORD"] = out["AGE_ORD"].astype(int)
    for c in BIN_COLS:
        out[c] = out[c].astype(int)
    assert out[EXPECTED_COLS + ["AGE_ORD"]].isna().sum().sum() == 0
    return out, info


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["LOG_INCOME"] = np.log1p(df["INCOME"])
    df["LOG_TRB"] = np.log1p(df["TRB"])
    df["PRODUCT_COUNT"] = df[PRODUCT_COLS].sum(axis=1)
    df["DIGITAL_SCORE"] = df[CHANNEL_COLS].sum(axis=1)
    df["INVESTOR"] = df[["SC", "SP", "BD"]].max(axis=1)
    df["CREDIT_HOLDER"] = df[["LN", "MT"]].max(axis=1)
    df["TRB_TO_ANNUAL_INCOME"] = np.where(df["INCOME"] > 0, df["TRB"] / (12 * df["INCOME"]), np.nan)
    return df


# =============================================================================
# 4. OUTLIERS & EDA
# =============================================================================
def outlier_analysis(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    log.info("STEP 4 | Outlier analysis")
    rep: dict = {}
    for c in NUM_COLS:
        x = df[c]
        lx = np.log1p(x)
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        lq1, lq3 = lx.quantile([0.25, 0.75])
        liqr = lq3 - lq1
        mad = stats.median_abs_deviation(lx, scale="normal")
        rz = (lx - lx.median()) / (mad + 1e-12)
        rep[c] = {
            "skew_raw": float(stats.skew(x)), "kurtosis_raw": float(stats.kurtosis(x)),
            "skew_log": float(stats.skew(lx)), "kurtosis_log": float(stats.kurtosis(lx)),
            "iqr_raw_outliers_pct": float(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean() * 100),
            "iqr_log_outliers_pct": float(((lx < lq1 - 1.5 * liqr) | (lx > lq3 + 1.5 * liqr)).mean() * 100),
            "robust_z_log_gt3.5_pct": float((rz.abs() > 3.5).mean() * 100),
            "p99": float(x.quantile(0.99)), "max": float(x.max()),
        }
        df[f"OUT_MAD_{c}"] = (rz.abs() > 3.5).astype(int)
    iso = IsolationForest(n_estimators=300, contamination=0.01, random_state=cfg.random_state, n_jobs=cfg.n_jobs)
    df["OUTLIER_IFOREST"] = (iso.fit_predict(df[MODEL_FEATURES]) == -1).astype(int)
    rep["isolation_forest_pct"] = float(df["OUTLIER_IFOREST"].mean() * 100)
    rep["decision"] = ("Extreme INCOME/TRB values are plausible (affluent customers), not errors, so they "
                       "are KEPT. Their effect is controlled by transformations benchmarked in Step 5.")
    for c in NUM_COLS:
        log.info(f"   {c}: {({k: round(v, 2) for k, v in rep[c].items()})}")
    log.info(f"   IsolationForest multivariate outliers: {rep['isolation_forest_pct']:.2f}%")

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for i, c in enumerate(NUM_COLS):
        sns.histplot(df[c], bins=100, ax=ax[i, 0], color="#3182bd")
        ax[i, 0].set_title(f"{c} raw (skew={rep[c]['skew_raw']:.1f})")
        sns.histplot(np.log1p(df[c]), bins=100, ax=ax[i, 1], color="#31a354")
        ax[i, 1].set_title(f"log1p({c}) (skew={rep[c]['skew_log']:.2f})")
    savefig(fig, "03_distributions_raw_vs_log")

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.5))
    for i, c in enumerate(NUM_COLS):
        sns.boxplot(y=df[c], ax=ax[2 * i], color="#fdae6b")
        ax[2 * i].set_title(f"{c} raw")
        sns.boxplot(y=np.log1p(df[c]), ax=ax[2 * i + 1], color="#9e9ac8")
        ax[2 * i + 1].set_title(f"log1p({c})")
    savefig(fig, "04_outlier_boxplots")
    return df, rep


def eda_plots(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [3, 1]})
    pen = df[BIN_COLS].mean().sort_values(ascending=False) * 100
    sns.barplot(x=pen.index, y=pen.values, ax=ax[0], palette="viridis")
    ax[0].set_ylabel("% customers")
    ax[0].set_title("Channel & product penetration")
    for i, v in enumerate(pen.values):
        ax[0].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
    df["AGE"].value_counts().reindex(AGE_BANDS).plot.bar(ax=ax[1], color="#6baed6")
    ax[1].set_title("Age bands")
    savefig(fig, "05_penetration_and_age")

    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[MODEL_FEATURES + ["PRODUCT_COUNT"]].corr(method="spearman")
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)
    ax.set_title("Spearman correlation matrix")
    savefig(fig, "06_correlation_spearman")


# =============================================================================
# 5. FEATURE SPACE / PREPROCESSING BENCHMARK
# =============================================================================
STRATEGIES = ["raw_standard", "log_standard", "winsor_log_standard", "yeo_johnson",
              "quantile_normal", "log_robust", "iforest_trimmed"]


def build_space(df: pd.DataFrame, strategy: str, cfg: Config) -> tuple[np.ndarray, list]:
    age = df[["AGE_ORD"]].astype(float).values
    inc = df["INCOME"].astype(float).values
    trb = df["TRB"].astype(float).values
    if strategy == "raw_standard":
        cont = StandardScaler().fit_transform(np.column_stack([age, inc, trb]))
    elif strategy in ("log_standard", "iforest_trimmed"):
        cont = StandardScaler().fit_transform(np.column_stack([age, np.log1p(inc), np.log1p(trb)]))
    elif strategy == "winsor_log_standard":
        ci = np.clip(inc, *np.quantile(inc, [0.005, 0.995]))
        ct = np.clip(trb, *np.quantile(trb, [0.005, 0.995]))
        cont = StandardScaler().fit_transform(np.column_stack([age, np.log1p(ci), np.log1p(ct)]))
    elif strategy == "yeo_johnson":
        pt = PowerTransformer(method="yeo-johnson", standardize=True).fit_transform(
            np.column_stack([inc / 1000.0, trb / 1000.0]))
        cont = np.column_stack([StandardScaler().fit_transform(age), pt])
    elif strategy == "quantile_normal":
        qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                 random_state=cfg.random_state).fit_transform(np.column_stack([inc, trb]))
        cont = StandardScaler().fit_transform(np.column_stack([age, qt]))
    elif strategy == "log_robust":
        cont = RobustScaler().fit_transform(np.column_stack([age, np.log1p(inc), np.log1p(trb)]))
    else:
        raise ValueError(strategy)
    B = df[BIN_COLS].astype(float).values * cfg.binary_weight
    return np.hstack([cont, B]), ["AGE_ORD", "INCOME_T", "TRB_T"] + BIN_COLS


def preprocessing_benchmark(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, str, str]:
    log.info("STEP 5 | Outlier-handling / preprocessing benchmark")
    rng = np.random.default_rng(cfg.random_state)
    n = len(df)
    bidx = rng.choice(n, min(20000, n), replace=False)
    sidx = rng.choice(len(bidx), min(cfg.sample_size, len(bidx)), replace=False)
    ks = list(range(cfg.k_min, min(cfg.k_max, 8) + 1))
    rows = []
    for strat in STRATEGIES:
        X_full, _ = build_space(df, strat, cfg)
        X = X_full[bidx]
        fit_mask = (df["OUTLIER_IFOREST"].values[bidx] == 0) if strat == "iforest_trimmed" else np.ones(len(X), bool)
        Xfit = X[fit_mask]
        for k in ks:
            km = KMeans(n_clusters=k, n_init=5, random_state=cfg.random_state).fit(Xfit)
            lab = km.predict(X)
            aris = []
            for b in range(5):
                ii = rng.choice(len(Xfit), int(0.8 * len(Xfit)), replace=False)
                lb = KMeans(n_clusters=k, n_init=3, random_state=cfg.random_state + b + 1).fit(Xfit[ii]).predict(X)
                aris.append(adjusted_rand_score(lab, lb))
            rows.append({"strategy": strat, "k": k,
                         "silhouette": silhouette_score(X[sidx], lab[sidx]),
                         "davies_bouldin": davies_bouldin_score(X, lab),
                         "stability_ari": np.mean(aris), "min_share": min_share(lab)})
        log.info(f"   {strat:<22} done")
    res = pd.DataFrame(rows)
    res["valid"] = res["min_share"] >= cfg.min_cluster_share
    agg = (res[res.valid].groupby("strategy")
           .agg(mean_silhouette=("silhouette", "mean"), max_silhouette=("silhouette", "max"),
                mean_db=("davies_bouldin", "mean"), mean_stability=("stability_ari", "mean"),
                n_valid_k=("k", "count")))
    agg["rank_score"] = composite_rank(agg, ["mean_silhouette", "mean_stability"], ["mean_db"])
    agg = agg.sort_values("rank_score")
    best, second = agg.index[0], (agg.index[1] if len(agg) > 1 else agg.index[0])
    res.to_csv(REP_DIR / "preprocessing_benchmark_detail.csv", index=False)
    agg.to_csv(REP_DIR / "preprocessing_benchmark_summary.csv")
    log.info(f"\n{agg.round(4)}\n   -> best: {best} | runner-up: {second}")

    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    for m, a, t in [("silhouette", ax[0], "Silhouette (↑)"), ("davies_bouldin", ax[1], "Davies-Bouldin (↓)"),
                    ("stability_ari", ax[2], "Bootstrap stability ARI (↑)")]:
        sns.lineplot(data=res, x="k", y=m, hue="strategy", marker="o", ax=a)
        a.set_title(t)
        a.legend(fontsize=7)
    fig.suptitle(f"Preprocessing / outlier-handling benchmark – selected: {best}", fontweight="bold")
    savefig(fig, "07_preprocessing_benchmark")
    return agg, best, second


# =============================================================================
# 6. CLUSTERING TENDENCY
# =============================================================================
def hopkins_statistic(X: np.ndarray, m: int, rs: int) -> float:
    """H ≈ 0.5 -> random; H -> 1 -> clusterable."""
    rng = np.random.default_rng(rs)
    n, d = X.shape
    m = min(m, n - 1)
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    idx = rng.choice(n, m, replace=False)
    w = nn.kneighbors(X[idx], n_neighbors=2)[0][:, 1]
    u_pts = rng.uniform(X.min(0), X.max(0), size=(m, d))
    u = nn.kneighbors(u_pts, n_neighbors=1)[0].ravel()
    return float(u.sum() / (u.sum() + w.sum()))


# =============================================================================
# 7. K-MEANS + K SELECTION
# =============================================================================
def bootstrap_stability(X: np.ndarray, k: int, B: int, rs: int, ref_labels: np.ndarray | None = None,
                        frac: float = 0.8) -> tuple[float, float, np.ndarray]:
    """Prediction-based bootstrap stability: ARI and Hennig cluster-wise Jaccard."""
    rng = np.random.default_rng(rs)
    r = KMeans(n_clusters=k, n_init=10, random_state=rs).fit(X).labels_ if ref_labels is None else ref_labels
    ref_ids = np.unique(r)
    aris, jac = [], np.zeros((B, len(ref_ids)))
    for b in range(B):
        idx = rng.choice(len(X), int(frac * len(X)), replace=False)
        lb = KMeans(n_clusters=k, n_init=3, random_state=rs + 101 + b).fit(X[idx]).predict(X)
        aris.append(adjusted_rand_score(r, lb))
        for ci, c in enumerate(ref_ids):
            A = r == c
            best = 0.0
            for c2 in np.unique(lb[A]):
                Bm = lb == c2
                best = max(best, (A & Bm).sum() / (A | Bm).sum())
            jac[b, ci] = best
    return float(np.mean(aris)), float(np.std(aris)), jac.mean(axis=0)


def gap_statistic(X: np.ndarray, ks: list, B: int, rs: int) -> tuple[pd.DataFrame, int]:
    """Tibshirani et al. (2001) gap statistic with PCA-aligned uniform reference."""
    rng = np.random.default_rng(rs)
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    Xp = (X - mu) @ Vt.T
    lo, hi = Xp.min(0), Xp.max(0)
    rows = []
    for k in ks:
        logW = np.log(KMeans(n_clusters=k, n_init=5, random_state=rs).fit(X).inertia_)
        ref = [np.log(KMeans(n_clusters=k, n_init=3, random_state=rs + b).fit(
            rng.uniform(lo, hi, size=Xp.shape) @ Vt + mu).inertia_) for b in range(B)]
        rows.append({"k": k, "gap": np.mean(ref) - logW, "gap_sd": np.std(ref) * np.sqrt(1 + 1 / B)})
    g = pd.DataFrame(rows)
    k_gap = int(g["k"].iloc[-1])
    for i in range(len(g) - 1):
        if g.gap.iloc[i] >= g.gap.iloc[i + 1] - g.gap_sd.iloc[i + 1]:
            k_gap = int(g.k.iloc[i])
            break
    return g, k_gap


def kmeans_selection(X: np.ndarray, cfg: Config, sil_idx: np.ndarray) -> tuple[pd.DataFrame, int, dict]:
    log.info("STEP 7 | K-Means & optimal-k search")
    rng = np.random.default_rng(cfg.random_state)
    n = len(X)
    Xst = X[rng.choice(n, min(cfg.stability_sample, n), replace=False)]
    ks = list(range(cfg.k_min, cfg.k_max + 1))
    rows = []
    for k in ks:
        lab = KMeans(n_clusters=k, n_init=10, random_state=cfg.random_state).fit(X)
        sm, ssd, _ = bootstrap_stability(Xst, k, cfg.n_bootstrap, cfg.random_state)
        rows.append({"k": k, "inertia": lab.inertia_,
                     "silhouette": silhouette_score(X[sil_idx], lab.labels_[sil_idx]),
                     "calinski_harabasz": calinski_harabasz_score(X, lab.labels_),
                     "davies_bouldin": davies_bouldin_score(X, lab.labels_),
                     "stability_ari": sm, "stability_sd": ssd, "min_share": min_share(lab.labels_)})
        log.info(f"   k={k:>2} sil={rows[-1]['silhouette']:.3f} CH={rows[-1]['calinski_harabasz']:.0f} "
                 f"DB={rows[-1]['davies_bouldin']:.3f} stab={sm:.3f}±{ssd:.3f} min_share={rows[-1]['min_share']:.3f}")
    tbl = pd.DataFrame(rows)
    gap_tbl, k_gap = gap_statistic(X[rng.choice(n, min(3000, n), replace=False)], ks, 10, cfg.random_state)
    tbl = tbl.merge(gap_tbl, on="k")
    k_elbow = knee_point(tbl.k, tbl.inertia)
    tbl["valid"] = tbl.min_share >= cfg.min_cluster_share
    tbl["composite_rank"] = composite_rank(tbl, ["silhouette", "calinski_harabasz", "stability_ari", "gap"],
                                           ["davies_bouldin"], "valid")
    if np.isinf(tbl.composite_rank).all():
        log.warning("   no k satisfies MIN_CLUSTER_SHARE – ignoring the constraint")
        tbl["composite_rank"] = composite_rank(tbl, ["silhouette", "calinski_harabasz", "stability_ari", "gap"],
                                               ["davies_bouldin"])
    k_star = int(tbl.loc[tbl.composite_rank.idxmin(), "k"])
    votes = {"elbow": k_elbow, "gap_tibshirani": k_gap,
             "silhouette": int(tbl.loc[tbl.silhouette.idxmax(), "k"]),
             "calinski_harabasz": int(tbl.loc[tbl.calinski_harabasz.idxmax(), "k"]),
             "davies_bouldin": int(tbl.loc[tbl.davies_bouldin.idxmin(), "k"]),
             "stability": int(tbl.loc[tbl.stability_ari.idxmax(), "k"]),
             "composite_rank_selected": k_star}
    tbl.to_csv(REP_DIR / "kmeans_k_selection.csv", index=False)
    log.info(f"   votes: {votes}  -> K* = {k_star}")

    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    ax = ax.ravel()
    ax[0].plot(tbl.k, tbl.inertia, "o-")
    ax[0].axvline(k_elbow, ls=":", c="grey", label=f"elbow={k_elbow}")
    ax[0].set_title("Elbow (inertia)")
    ax[1].plot(tbl.k, tbl.silhouette, "o-", c="green")
    ax[1].set_title("Silhouette (↑)")
    ax[2].plot(tbl.k, tbl.calinski_harabasz, "o-", c="purple")
    ax[2].set_title("Calinski-Harabasz (↑)")
    ax[3].plot(tbl.k, tbl.davies_bouldin, "o-", c="orange")
    ax[3].set_title("Davies-Bouldin (↓)")
    ax[4].errorbar(tbl.k, tbl.stability_ari, yerr=tbl.stability_sd, fmt="o-", c="teal", capsize=3)
    ax[4].set_title("Bootstrap stability ARI (↑)")
    ax[5].errorbar(tbl.k, tbl.gap, yerr=tbl.gap_sd, fmt="o-", c="brown", capsize=3)
    ax[5].axvline(k_gap, ls=":", c="grey", label=f"gap rule={k_gap}")
    ax[5].set_title("Gap statistic (↑)")
    for a in ax:
        a.axvline(k_star, c="red", alpha=0.5, lw=2, label=f"K*={k_star}")
        a.set_xlabel("k")
        a.legend(fontsize=8)
    fig.suptitle("K-Means – optimal number of clusters", fontweight="bold")
    savefig(fig, "08_kmeans_k_selection")
    return tbl, k_star, votes


# =============================================================================
# 8. HIERARCHICAL CLUSTERING
# =============================================================================
def propagate_labels(X_s, lab_s, X_full, idx, n_neighbors=15) -> tuple[np.ndarray, float]:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_s, lab_s)
    full = knn.predict(X_full)
    full[idx] = lab_s
    acc = cross_val_score(KNeighborsClassifier(n_neighbors=n_neighbors), X_s, lab_s, cv=5).mean()
    return full, float(acc)


def hierarchical_analysis(X: np.ndarray, cfg: Config, k_star: int) -> dict:
    log.info("STEP 8 | Hierarchical clustering")
    rng = np.random.default_rng(cfg.random_state + 7)
    idx = rng.choice(len(X), min(cfg.hc_sample, len(X)), replace=False)
    Xs = X[idx]
    D = pdist(Xs)
    Z, coph = {}, {}
    for m in ["ward", "average", "complete"]:
        Z[m] = linkage(D, method=m)
        coph[m] = float(cophenet(Z[m], D)[0])
    log.info(f"   cophenetic correlation: {({k: round(v, 3) for k, v in coph.items()})}")
    Zw = Z["ward"]
    rows = []
    for k in range(cfg.k_min, cfg.k_max + 1):
        lab = fcluster(Zw, k, "maxclust") - 1
        rows.append({"k": k, "silhouette": silhouette_score(Xs, lab),
                     "calinski_harabasz": calinski_harabasz_score(Xs, lab),
                     "davies_bouldin": davies_bouldin_score(Xs, lab),
                     "merge_height_gap": Zw[-(k - 1), 2] - Zw[-k, 2], "min_share": min_share(lab)})
    tbl = pd.DataFrame(rows)
    tbl["valid"] = tbl.min_share >= cfg.min_cluster_share
    tbl["composite_rank"] = composite_rank(tbl, ["silhouette", "calinski_harabasz", "merge_height_gap"],
                                           ["davies_bouldin"], "valid")
    if np.isinf(tbl.composite_rank).all():
        tbl["composite_rank"] = composite_rank(tbl, ["silhouette", "calinski_harabasz", "merge_height_gap"],
                                               ["davies_bouldin"])
    k_hc = int(tbl.loc[tbl.composite_rank.idxmin(), "k"])
    tbl.to_csv(REP_DIR / "hierarchical_k_selection.csv", index=False)
    lab_s = fcluster(Zw, k_star, "maxclust") - 1
    full, acc = propagate_labels(Xs, lab_s, X, idx)
    log.info(f"   hierarchical own best k = {k_hc}; labels at K*={k_star} propagated (kNN CV acc={acc:.3f})")

    fig, ax = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={"width_ratios": [2, 1]})
    dendrogram(Zw, truncate_mode="lastp", p=40, ax=ax[0], leaf_rotation=90, leaf_font_size=7,
               show_contracted=True, color_threshold=(Zw[-k_star, 2] + Zw[-(k_star - 1), 2]) / 2)
    ax[0].axhline((Zw[-k_star, 2] + Zw[-(k_star - 1), 2]) / 2, c="red", ls="--", label=f"cut K*={k_star}")
    ax[0].set_title(f"Ward dendrogram (n={len(idx)} sample, truncated)")
    ax[0].legend()
    pd.Series(coph).plot.bar(ax=ax[1], color=["#3182bd", "#9ecae1", "#c6dbef"])
    ax[1].set_ylim(0, 1)
    ax[1].set_title("Cophenetic correlation by linkage")
    savefig(fig, "09_hierarchical_dendrogram")

    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for a, m in zip(ax, ["silhouette", "calinski_harabasz", "davies_bouldin", "merge_height_gap"]):
        a.plot(tbl.k, tbl[m], "o-")
        a.axvline(k_hc, c="grey", ls=":", label=f"HC best={k_hc}")
        a.axvline(k_star, c="red", alpha=0.5, label=f"K*={k_star}")
        a.set_title(m)
        a.legend(fontsize=8)
    savefig(fig, "10_hierarchical_k_selection")
    return {"table": tbl, "k_best": k_hc, "labels": full, "cophenetic": coph, "propagation_cv_acc": acc}


# =============================================================================
# 9. ADVANCED COMPOSITE METHOD: FAMD + GMM
# =============================================================================
def famd_embedding(df: pd.DataFrame, var_target: float, rs: int) -> tuple[np.ndarray, PCA, int]:
    """FAMD: continuous -> z-scores; categorical indicators -> (I - p)/sqrt(p); then PCA.
    Gives each continuous variable inertia 1 and each categorical variable inertia (levels-1)."""
    blocks = [StandardScaler().fit_transform(np.column_stack([df["LOG_INCOME"], df["LOG_TRB"]]))]
    cats = {"AGE": df["AGE"].values, **{c: df[c].values for c in BIN_COLS}}
    for _, v in cats.items():
        for level in pd.unique(v):
            ind = (v == level).astype(float)
            p = ind.mean()
            if 0 < p < 1:
                blocks.append(((ind - p) / np.sqrt(p))[:, None])
    F = np.hstack(blocks)
    pca = PCA(random_state=rs).fit(F)
    ncomp = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), var_target) + 1)
    return pca.transform(F)[:, :ncomp], pca, ncomp


def famd_gmm_analysis(df: pd.DataFrame, cfg: Config, k_star: int, sil_idx: np.ndarray) -> dict:
    log.info("STEP 9 | Advanced composite method: FAMD embedding + Gaussian Mixture Model")
    E, pca, ncomp = famd_embedding(df, cfg.famd_var_target, cfg.random_state)
    log.info(f"   FAMD components retained: {ncomp} ({cfg.famd_var_target:.0%} inertia)")
    rows = []
    for k in range(cfg.k_min, cfg.k_max + 1):
        g = GaussianMixture(n_components=k, covariance_type="full", n_init=3, reg_covar=1e-4,
                            max_iter=500, random_state=cfg.random_state).fit(E)
        post = g.predict_proba(E)
        lab = post.argmax(1)
        ent = -(post * np.log(post + 1e-12)).sum(1).mean() / np.log(k)
        rows.append({"k": k, "bic": g.bic(E), "aic": g.aic(E),
                     "silhouette_famd": silhouette_score(E[sil_idx], lab[sil_idx]),
                     "mean_max_posterior": post.max(1).mean(), "normalised_entropy": ent,
                     "min_share": min_share(lab), "converged": g.converged_})
        log.info(f"   k={k:>2} BIC={rows[-1]['bic']:.0f} sil={rows[-1]['silhouette_famd']:.3f} "
                 f"maxP={rows[-1]['mean_max_posterior']:.3f}")
    tbl = pd.DataFrame(rows)
    valid = tbl[tbl.min_share >= cfg.min_cluster_share]
    valid = valid if len(valid) else tbl
    k_bic = int(valid.loc[valid.bic.idxmin(), "k"])
    rule = "min BIC"
    if k_bic == cfg.k_max:
        k_bic = knee_point(tbl.k, tbl.bic)
        rule = "BIC elbow (BIC still decreasing at K_MAX)"
    tbl.to_csv(REP_DIR / "famd_gmm_k_selection.csv", index=False)
    g = GaussianMixture(n_components=k_star, covariance_type="full", n_init=5, reg_covar=1e-4,
                        max_iter=500, random_state=cfg.random_state).fit(E)
    post = g.predict_proba(E)
    log.info(f"   GMM own best k = {k_bic} ({rule}); fitted at K*={k_star}")

    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    cum = np.cumsum(pca.explained_variance_ratio_)
    ax[0].bar(range(1, len(cum) + 1), pca.explained_variance_ratio_, color="#9ecae1")
    ax[0].plot(range(1, len(cum) + 1), cum, "o-", c="#08519c")
    ax[0].axvline(ncomp, c="red", ls="--", label=f"{ncomp} comps")
    ax[0].set_title("FAMD inertia explained")
    ax[0].legend()
    ax[1].plot(tbl.k, tbl.bic, "o-", label="BIC")
    ax[1].plot(tbl.k, tbl.aic, "s-", label="AIC")
    ax[1].axvline(k_bic, c="grey", ls=":", label=f"GMM best={k_bic}")
    ax[1].axvline(k_star, c="red", alpha=0.5, label=f"K*={k_star}")
    ax[1].set_title("GMM information criteria (↓)")
    ax[1].legend()
    ax[2].plot(tbl.k, tbl.mean_max_posterior, "o-", c="green", label="mean max posterior")
    ax[2].plot(tbl.k, 1 - tbl.normalised_entropy, "s-", c="purple", label="1 - norm. entropy")
    ax[2].set_title("Assignment certainty (↑)")
    ax[2].legend()
    savefig(fig, "11_famd_gmm_selection")
    return {"table": tbl, "k_best": k_bic, "rule": rule, "labels": post.argmax(1),
            "max_posterior": post.max(1), "embedding": E, "ncomp": ncomp}


# =============================================================================
# 10. ENSEMBLE (CONSENSUS) CLUSTERING
# =============================================================================
def item_consensus(cons: np.ndarray, L: np.ndarray) -> np.ndarray:
    """For each customer: average over base partitions of how typical its base label is in its consensus segment."""
    score = np.zeros(len(cons))
    for j in range(L.shape[1]):
        d = pd.DataFrame({"c": cons, "b": L[:, j]})
        score += (d.groupby(["c", "b"])["c"].transform("size") / d.groupby("c")["c"].transform("size")).values
    return score / L.shape[1]


def consensus_ensemble(base: dict, k: int, cfg: Config) -> dict:
    log.info(f"STEP 10 | Ensemble consensus clustering over {len(base)} base partitions")
    names = list(base)
    L = np.column_stack([base[nm] for nm in names]).astype(int)
    n, P = L.shape
    rng = np.random.default_rng(cfg.random_state + 11)
    m = min(cfg.consensus_sample, n)
    idx = rng.choice(n, m, replace=False)
    Ls = L[idx]
    co = np.zeros((m, m), dtype=np.float32)
    for j in range(P):
        co += (Ls[:, j][:, None] == Ls[:, j][None, :])
    co /= P
    D = 1.0 - co
    np.fill_diagonal(D, 0.0)
    Dc = squareform(D.astype(np.float64), checks=False)

    cands = {}
    for method in ["average", "ward"]:
        lab_s = fcluster(linkage(Dc, method=method), k, "maxclust") - 1
        # Hamming distance on label vectors == 1 - co-association -> exact propagation rule
        knn = KNeighborsClassifier(n_neighbors=25, metric="hamming", algorithm="brute").fit(Ls, lab_s)
        full = knn.predict(L)
        full[idx] = lab_s
        cands[f"EAC_{method}"] = full
    H = np.hstack([(L[:, [j]] == np.unique(L[:, j])[None, :]).astype(np.float32) for j in range(P)])
    cands["MetaKMeans_onehot"] = KMeans(n_clusters=k, n_init=20, random_state=cfg.random_state).fit_predict(H)

    rows = []
    for nm, lab in cands.items():
        rows.append({"candidate": nm, "ANMI": np.mean([normalized_mutual_info_score(lab, L[:, j]) for j in range(P)]),
                     "n_clusters": len(np.unique(lab)), "min_share": min_share(lab)})
    ct = pd.DataFrame(rows)
    ct["valid"] = (ct.n_clusters == k) & (ct.min_share >= cfg.min_cluster_share)
    pool = ct[ct.valid] if ct.valid.any() else ct
    chosen = pool.loc[pool.ANMI.idxmax(), "candidate"]
    final = relabel_by_size(cands[chosen])
    ct.to_csv(REP_DIR / "ensemble_candidates.csv", index=False)
    log.info(f"\n{ct.round(4)}\n   -> consensus selected: {chosen}")

    # co-association heat-map (ordered by consensus label)
    sub = np.arange(min(1500, m))
    order = sub[np.argsort(final[idx][sub], kind="stable")]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(co[np.ix_(order, order)], cmap="mako", ax=ax, xticklabels=False, yticklabels=False,
                cbar_kws={"label": "co-association"})
    ax.set_title("Co-association matrix ordered by consensus segment")
    savefig(fig, "12_coassociation_matrix")

    within = []
    lab_s = final[idx]
    for c in np.unique(lab_s):
        mm = lab_s == c
        within.append(float(co[np.ix_(mm, mm)].mean()))
    return {"labels": final, "chosen": chosen, "candidates": ct, "base_names": names,
            "item_consensus": item_consensus(final, L), "cluster_consensus": within}


# =============================================================================
# 11. VALIDATION
# =============================================================================
def internal_validation(X, labels_dict, sil_idx) -> pd.DataFrame:
    rows = []
    for nm, lab in labels_dict.items():
        rows.append({"method": nm, "n_clusters": len(np.unique(lab)),
                     "silhouette": silhouette_score(X[sil_idx], lab[sil_idx]),
                     "calinski_harabasz": calinski_harabasz_score(X, lab),
                     "davies_bouldin": davies_bouldin_score(X, lab), "min_share": min_share(lab)})
    return pd.DataFrame(rows)


def agreement_matrices(labels_dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    nm = list(labels_dict)
    ari = pd.DataFrame(index=nm, columns=nm, dtype=float)
    nmi = ari.copy()
    for a in nm:
        for b in nm:
            ari.loc[a, b] = adjusted_rand_score(labels_dict[a], labels_dict[b])
            nmi.loc[a, b] = normalized_mutual_info_score(labels_dict[a], labels_dict[b])
    return ari, nmi


def permutation_null_test(X, labels, k, cfg, sil_idx) -> dict:
    """H0: no multivariate structure (columns independently permuted, marginals kept)."""
    rng = np.random.default_rng(cfg.random_state + 99)
    Xs = X[sil_idx]
    obs = silhouette_score(Xs, labels[sil_idx])
    null = []
    for b in range(cfg.n_null):
        Xp = np.column_stack([rng.permutation(Xs[:, j]) for j in range(Xs.shape[1])])
        lab = KMeans(n_clusters=k, n_init=5, random_state=cfg.random_state + b).fit_predict(Xp)
        null.append(silhouette_score(Xp, lab))
    null = np.array(null)
    res = {"observed_silhouette": obs, "null_mean": null.mean(), "null_sd": null.std(),
           "z_score": (obs - null.mean()) / (null.std() + 1e-12),
           "empirical_p_value": (1 + (null >= obs).sum()) / (len(null) + 1), "null_values": null}
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(null, bins=15, ax=ax, color="grey", label="null (permuted data, K-Means optimised)")
    ax.axvline(obs, c="red", lw=2, label=f"observed ensemble = {obs:.3f}")
    ax.set_title(f"Permutation null test – z={res['z_score']:.1f}, p={res['empirical_p_value']:.3f}")
    ax.legend()
    savefig(fig, "17_permutation_null_test")
    return res


def rf_separability(df, labels, cfg) -> dict:
    rng = np.random.default_rng(cfg.random_state)
    idx = rng.choice(len(df), min(15000, len(df)), replace=False)
    Xr, y = df.iloc[idx][MODEL_FEATURES].values, labels[idx]
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=cfg.random_state, n_jobs=cfg.n_jobs)
    cv = cross_validate(rf, Xr, y, cv=StratifiedKFold(5, shuffle=True, random_state=cfg.random_state),
                        scoring=["accuracy", "f1_macro"], n_jobs=1)
    Xtr, Xte, ytr, yte = train_test_split(Xr, y, test_size=0.3, stratify=y, random_state=cfg.random_state)
    rf.fit(Xtr, ytr)
    pi = permutation_importance(rf, Xte, yte, n_repeats=10, random_state=cfg.random_state, n_jobs=cfg.n_jobs)
    imp = pd.DataFrame({"feature": MODEL_FEATURES, "importance_mean": pi.importances_mean,
                        "importance_sd": pi.importances_std}).sort_values("importance_mean", ascending=False)
    imp.to_csv(REP_DIR / "segment_driver_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(imp.feature[::-1], imp.importance_mean[::-1], xerr=imp.importance_sd[::-1], color="#4292c6")
    ax.set_title("Segment drivers – permutation importance (Random Forest)")
    ax.set_xlabel("mean decrease in accuracy")
    savefig(fig, "18_segment_drivers_importance")
    return {"cv_accuracy_mean": cv["test_accuracy"].mean(), "cv_accuracy_sd": cv["test_accuracy"].std(),
            "cv_f1_macro_mean": cv["test_f1_macro"].mean(), "cv_f1_macro_sd": cv["test_f1_macro"].std(),
            "chance_level": float(pd.Series(y).value_counts(normalize=True).max())}


def statistical_tests(df, labels) -> pd.DataFrame:
    n = len(df)
    rows = []
    for c in ["INCOME", "TRB", "AGE_ORD", "PRODUCT_COUNT", "DIGITAL_SCORE"]:
        groups = [df.loc[labels == g, c].values for g in np.unique(labels)]
        H, p = stats.kruskal(*groups)
        e2 = H / (n - 1)
        mag = "negligible" if e2 < 0.01 else "small" if e2 < 0.08 else "medium" if e2 < 0.26 else "large"
        rows.append({"feature": c, "test": "Kruskal-Wallis", "statistic": H, "p_value": p,
                     "effect_size": e2, "effect_metric": "epsilon²", "magnitude": mag})
    for c in BIN_COLS + ["AGE"]:
        ct = pd.crosstab(labels, df[c])
        if ct.shape[1] < 2:
            continue
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        V = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
        mag = "negligible" if V < 0.1 else "small" if V < 0.3 else "medium" if V < 0.5 else "large"
        rows.append({"feature": c, "test": "Chi-square", "statistic": chi2, "p_value": p,
                     "effect_size": V, "effect_metric": "Cramér's V", "magnitude": mag})
    t = pd.DataFrame(rows)
    t["p_adj_BH"] = bh_adjust(t.p_value)
    t["significant_FDR_0.05"] = t.p_adj_BH < 0.05
    return t.sort_values("effect_size", ascending=False)


# =============================================================================
# 12. PROFILING & PERSONAS
# =============================================================================
def _profile(d: pd.DataFrame, key: str) -> pd.DataFrame:
    g = d.groupby(key)
    P = pd.DataFrame({"n": g.size()})
    P["share"] = P["n"] / len(d)
    P["income_median"] = g["INCOME"].median()
    P["income_mean"] = g["INCOME"].mean()
    P["trb_median"] = g["TRB"].median()
    P["trb_mean"] = g["TRB"].mean()
    P["age_mode"] = g["AGE"].agg(lambda s: s.mode().iloc[0])
    for b in AGE_BANDS:
        P[f"age_{b}"] = g["AGE"].agg(lambda s, b=b: (s == b).mean())
    for c in BIN_COLS + ["INVESTOR", "CREDIT_HOLDER"]:
        P[f"{c}_rate"] = g[c].mean()
    P["product_count_mean"] = g["PRODUCT_COUNT"].mean()
    P["digital_score_mean"] = g["DIGITAL_SCORE"].mean()
    return P


def build_profiles(df, labels):
    prof = _profile(df.assign(SEGMENT=labels), "SEGMENT")
    overall = _profile(df.assign(_ALL=0), "_ALL").iloc[0]
    lift = pd.DataFrame(index=prof.index)
    lift["INCOME"] = prof.income_median / max(overall.income_median, 1e-9)
    lift["TRB"] = prof.trb_median / max(overall.trb_median, 1e-9)
    for c in BIN_COLS + ["INVESTOR", "CREDIT_HOLDER"]:
        lift[c] = prof[f"{c}_rate"] / max(overall[f"{c}_rate"], 1e-9)
    lift["PRODUCT_COUNT"] = prof.product_count_mean / max(overall.product_count_mean, 1e-9)
    return prof, overall, lift


TRAIT_RULES = [("INVESTOR", 1.5, "Investor"), ("MT", 1.8, "Home-Owner"), ("LN", 1.5, "Borrower"),
               ("FX_TRANS", 1.5, "Globetrotter"), ("TD", 1.5, "Saver"), ("MPF", 1.5, "Retirement-Planner"),
               ("CC", 1.3, "Card-Spender"), ("PAYME", 1.3, "PayMe-Social-Payer")]
NEXT_BEST = {
    "DIG_ACTIVE": "Digital-banking activation journey (app onboarding, e-statements)",
    "PAYME": "PayMe activation with P2P / merchant incentives",
    "FX_TRANS": "Multi-currency account & travel FX offers",
    "CC": "Starter / cash-back credit card linked to PayMe",
    "LN": "Pre-approved personal loan (affordability-checked)",
    "MT": "First-home mortgage advisory",
    "TD": "Goal-based term deposits / time-deposit promotions",
    "SC": "Low-cost securities trading & monthly stock-investment plan",
    "SP": "Entry structured products (subject to suitability assessment)",
    "BD": "Bond / fixed-income funds for capital preservation",
    "MPF": "MPF consolidation & tax-deductible voluntary contributions (TVC)",
}
AGE_LABEL = {"18-24": "Gen-Z Starters (18-24)", "25-29": "Young Professionals (25-29)",
             "30-34": "Establishing Adults (30-34)"}


def generate_personas(prof, overall, lift) -> dict:
    personas, used = {}, set()
    for seg, r in prof.iterrows():
        L = lift.loc[seg]
        if L.TRB >= 3 or L.INCOME >= 2:
            wealth = "Affluent"
        elif L.TRB >= 1.5 or L.INCOME >= 1.3:
            wealth = "Emerging-Affluent"
        elif L.TRB <= 0.3 and L.INCOME <= 0.9:
            wealth = "Entry-Level"
        else:
            wealth = "Mass-Market"
        traits = sorted([(L[c], lab) for c, thr, lab in TRAIT_RULES
                         if L[c] >= thr and r[f"{c}_rate"] >= 0.05], reverse=True)
        trait_str = " & ".join(t[1] for t in traits[:2])
        if not trait_str:
            trait_str = "Low-Engagement" if L.PRODUCT_COUNT < 0.7 else "Mainstream"
        digital_light = r.DIG_ACTIVE_rate < 0.8 * overall.DIG_ACTIVE_rate
        name = f"{wealth} {trait_str}" + (" (Digital-Light)" if digital_light else "")
        if name in used:
            name = f"{name} – {r.age_mode}"
        used.add(name)

        over = L.drop(["INCOME", "TRB", "PRODUCT_COUNT"]).sort_values(ascending=False)
        top_over = [f"{FEATURE_LABELS.get(c, c)} ({r[f'{c}_rate']:.0%}, x{v:.1f})" for c, v in over.head(3).items() if v > 1.1]
        top_under = [f"{FEATURE_LABELS.get(c, c)} ({r[f'{c}_rate']:.0%}, x{v:.1f})" for c, v in over.tail(3).items() if v < 0.9]

        gaps = []
        for c in PRODUCT_COLS + CHANNEL_COLS:
            if r[f"{c}_rate"] >= overall[f"{c}_rate"] * 0.8:
                continue
            if c == "MT" and (L.INCOME < 1.0 or r.age_mode == "18-24"):
                continue
            if c in ("SP", "BD") and L.TRB < 1.0:
                continue
            if c == "LN" and L.INCOME < 0.8:
                continue
            gaps.append((overall[f"{c}_rate"] - r[f"{c}_rate"], c))
        actions = [NEXT_BEST[c] for _, c in sorted(gaps, reverse=True)[:3]]
        if digital_light:
            actions.insert(0, "Channel: digital migration nudges (SMS/branch QR to app activation)")
        personas[int(seg)] = {
            "name": name, "life_stage": AGE_LABEL.get(r.age_mode, r.age_mode),
            "size": int(r.n), "share": float(r.share),
            "median_income_hkd": float(r.income_median), "median_trb_hkd": float(r.trb_median),
            "avg_products": float(r.product_count_mean), "digital_active_rate": float(r.DIG_ACTIVE_rate),
            "over_indexed": top_over, "under_indexed": top_under,
            "recommended_actions": actions or ["Retention & deepening: loyalty / relationship upgrade"],
        }
    return personas


def write_personas_md(personas, overall, chosen, k) -> None:
    lines = ["# HSBC <35 Customer Personas", "",
             f"Segments: **{k}** | consensus method: **{chosen}** | "
             f"overall median income HKD {overall.income_median:,.0f} | overall median TRB HKD {overall.trb_median:,.0f}", ""]
    for seg, p in personas.items():
        lines += [f"## Segment {seg}: {p['name']}",
                  f"*{p['life_stage']}*  —  {p['size']:,} customers ({p['share']:.1%})", "",
                  f"- Median monthly income: **HKD {p['median_income_hkd']:,.0f}**",
                  f"- Median TRB: **HKD {p['median_trb_hkd']:,.0f}**",
                  f"- Avg. products held: **{p['avg_products']:.2f}** | Digital active: **{p['digital_active_rate']:.0%}**",
                  f"- Over-indexed on: {', '.join(p['over_indexed']) or '—'}",
                  f"- Under-indexed on: {', '.join(p['under_indexed']) or '—'}",
                  "- **Recommended actions:**"] + [f"  - {a}" for a in p["recommended_actions"]] + [""]
    (REP_DIR / "personas.md").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 13. FINAL FIGURES
# =============================================================================
def plot_final(X, df, labels_dict, final, personas, prof, lift, sil_idx, jacc, cfg) -> None:
    rng = np.random.default_rng(cfg.random_state)
    p2 = PCA(n_components=2, random_state=cfg.random_state).fit(X)
    pidx = rng.choice(len(X), min(6000, len(X)), replace=False)
    P = p2.transform(X[pidx])
    nm = list(labels_dict)
    ncol = 3
    nrow = int(np.ceil(len(nm) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(6 * ncol, 5 * nrow))
    ax = np.atleast_1d(ax).ravel()
    for a, m in zip(ax, nm):
        a.scatter(P[:, 0], P[:, 1], c=labels_dict[m][pidx], cmap="tab10", s=4, alpha=0.6)
        a.set_title(m)
        a.set_xlabel(f"PC1 ({p2.explained_variance_ratio_[0]:.0%})")
        a.set_ylabel(f"PC2 ({p2.explained_variance_ratio_[1]:.0%})")
    for a in ax[len(nm):]:
        a.axis("off")
    fig.suptitle("PCA projection coloured by each method's segments", fontweight="bold")
    savefig(fig, "14_pca_projection_methods")

    s = silhouette_samples(X[sil_idx], final[sil_idx])
    fig, ax = plt.subplots(figsize=(9, 6))
    y0 = 0
    for c in np.unique(final):
        v = np.sort(s[final[sil_idx] == c])
        ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, alpha=0.75, color=plt.cm.tab10(c % 10))
        ax.text(-0.05, y0 + len(v) / 2, f"S{c}")
        y0 += len(v) + 40
    ax.axvline(s.mean(), c="red", ls="--", label=f"mean={s.mean():.3f}")
    ax.set_title("Silhouette plot – final ensemble segments")
    ax.set_xlabel("silhouette coefficient")
    ax.set_yticks([])
    ax.legend()
    savefig(fig, "15_silhouette_final")

    fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
    names = [f"S{i}: {personas[i]['name']}" for i in prof.index]
    ax[0].barh(names[::-1], prof["share"][::-1] * 100, color=[plt.cm.tab10(i % 10) for i in prof.index][::-1])
    ax[0].set_xlabel("% customers")
    ax[0].set_title("Segment sizes")
    ax[1].bar([f"S{i}" for i in range(len(jacc))], jacc, color="#6baed6")
    ax[1].axhline(0.75, c="green", ls="--", label="0.75 stable")
    ax[1].axhline(0.6, c="orange", ls="--", label="0.60 pattern")
    ax[1].axhline(0.5, c="red", ls="--", label="0.50 dissolved")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("Cluster-wise bootstrap Jaccard (Hennig)")
    ax[1].legend(fontsize=8)
    savefig(fig, "16_segment_sizes_and_stability")

    rate_cols = ["INCOME", "TRB"] + BIN_COLS + ["INVESTOR", "PRODUCT_COUNT"]
    annot = pd.DataFrame(index=prof.index)
    annot["INCOME"] = prof.income_median.map(lambda v: f"{v / 1000:.0f}k")
    annot["TRB"] = prof.trb_median.map(lambda v: f"{v / 1000:.0f}k")
    for c in BIN_COLS + ["INVESTOR"]:
        annot[c] = prof[f"{c}_rate"].map(lambda v: f"{v:.0%}")
    annot["PRODUCT_COUNT"] = prof.product_count_mean.map(lambda v: f"{v:.2f}")
    fig, ax = plt.subplots(figsize=(15, 0.8 * len(prof) + 2))
    sns.heatmap(np.log2(lift[rate_cols].clip(lower=1 / 8, upper=8)), annot=annot[rate_cols].values, fmt="",
                cmap="RdBu_r", center=0, ax=ax, yticklabels=names, cbar_kws={"label": "log2 lift vs population"})
    ax.set_title("Segment profiles (cell text = median / rate; colour = lift vs total population)")
    savefig(fig, "19_profile_heatmap_lift")

    cats = ["INCOME", "TRB", "DIG_ACTIVE", "PAYME", "FX_TRANS", "CC", "LN", "MT", "TD", "INVESTOR", "MPF"]
    ang = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    ang += ang[:1]
    k = len(prof)
    nc = min(3, k)
    nr = int(np.ceil(k / nc))
    fig = plt.figure(figsize=(5.5 * nc, 5.5 * nr))
    for i, seg in enumerate(prof.index):
        a = fig.add_subplot(nr, nc, i + 1, polar=True)
        v = np.clip(lift.loc[seg, cats].values, 0, 3).tolist()
        v += v[:1]
        col = plt.cm.tab10(seg % 10)
        a.plot(ang, v, c=col, lw=2)
        a.fill(ang, v, c=col, alpha=0.25)
        a.plot(ang, [1] * len(ang), "--", c="grey", lw=1)
        a.set_xticks(ang[:-1])
        a.set_xticklabels(cats, fontsize=8)
        a.set_ylim(0, 3)
        a.set_title(f"S{seg}: {personas[seg]['name']}\n({personas[seg]['share']:.1%})", fontsize=9)
    fig.suptitle("Persona radar – lift vs population (dashed = 1.0, capped at 3)", fontweight="bold")
    savefig(fig, "20_persona_radar")

    d = df.assign(SEGMENT=final).sample(min(15000, len(df)), random_state=cfg.random_state)
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    sns.boxplot(data=d, x="SEGMENT", y="LOG_INCOME", ax=ax[0], palette="tab10")
    ax[0].set_title("log1p(INCOME) by segment")
    sns.boxplot(data=d, x="SEGMENT", y="LOG_TRB", ax=ax[1], palette="tab10")
    ax[1].set_title("log1p(TRB) by segment")
    pd.crosstab(d.SEGMENT, d.AGE, normalize="index").reindex(columns=AGE_BANDS).plot.bar(
        stacked=True, ax=ax[2], colormap="Blues")
    ax[2].set_title("Age mix by segment")
    ax[2].legend(fontsize=8)
    savefig(fig, "21_segment_distributions")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    t0 = time.time()
    cfg = Config.from_env()
    setup(cfg)
    np.random.seed(cfg.random_state)
    log.info("=" * 80)
    log.info("HSBC SYNTHETIC DATA – SEGMENTATION & PERSONA PIPELINE")
    log.info("=" * 80)

    df_raw = load_data(cfg)
    df, dq = audit_and_clean(df_raw)
    plot_missing(df_raw, df)

    imp_summary, best_imp = imputation_benchmark(df, cfg)
    df, imp_info = apply_imputation(df, best_imp, cfg)
    dq["imputation"] = imp_info
    df = add_derived(df)

    df, out_rep = outlier_analysis(df, cfg)
    eda_plots(df)
    save_json({"data_quality": dq, "outliers": out_rep}, "data_quality_report.json")
    df.to_csv(DATA_DIR / "clean_data.csv", index=False)

    prep_tbl, best_strat, second_strat = preprocessing_benchmark(df, cfg)
    X, feat_names = build_space(df, best_strat, cfg)
    rng = np.random.default_rng(cfg.random_state)
    sil_idx = rng.choice(len(X), min(cfg.sample_size, len(X)), replace=False)

    H = hopkins_statistic(X, 1000, cfg.random_state)
    log.info(f"STEP 6 | Hopkins statistic = {H:.3f} (0.5 = random, >0.75 = strong tendency)")

    km_tbl, k_star, votes = kmeans_selection(X, cfg, sil_idx)
    hc = hierarchical_analysis(X, cfg, k_star)
    gm = famd_gmm_analysis(df, cfg, k_star, sil_idx)

    # --------------------- base partitions -----------------------------------
    km_final = KMeans(n_clusters=k_star, n_init=20, random_state=cfg.random_state).fit(X)
    X2, _ = build_space(df, second_strat, cfg)
    base = {
        f"KMeans_k{k_star}": km_final.labels_,
        f"Ward_HC_k{k_star}": hc["labels"],
        f"FAMD_GMM_k{k_star}": gm["labels"],
        f"FAMD_KMeans_k{k_star}": KMeans(n_clusters=k_star, n_init=10, random_state=cfg.random_state).fit_predict(gm["embedding"]),
        f"KMeans_{second_strat}_k{k_star}": KMeans(n_clusters=k_star, n_init=10, random_state=cfg.random_state).fit_predict(X2),
        f"KMeans_k{k_star + 1}": KMeans(n_clusters=k_star + 1, n_init=10, random_state=cfg.random_state).fit_predict(X),
    }
    if k_star - 1 >= 2:
        base[f"KMeans_k{k_star - 1}"] = KMeans(n_clusters=k_star - 1, n_init=10, random_state=cfg.random_state).fit_predict(X)

    ens = consensus_ensemble(base, k_star, cfg)
    final = ens["labels"]

    # --------------------- validation ----------------------------------------
    log.info("STEP 11 | Statistical validation")
    compare = {f"KMeans_k{k_star}": km_final.labels_, f"Ward_HC_k{k_star}": hc["labels"],
               f"FAMD_GMM_k{k_star}": gm["labels"], f"FAMD_KMeans_k{k_star}": base[f"FAMD_KMeans_k{k_star}"],
               "ENSEMBLE": final}
    iv = internal_validation(X, compare, sil_idx)
    iv.to_csv(REP_DIR / "method_comparison_internal_indices.csv", index=False)
    log.info(f"\n{iv.round(4)}")
    ari, nmi = agreement_matrices(compare)
    ari.to_csv(REP_DIR / "method_agreement_ARI.csv")
    nmi.to_csv(REP_DIR / "method_agreement_NMI.csv")
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(ari.astype(float), annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax[0])
    ax[0].set_title("Adjusted Rand Index between methods")
    sns.heatmap(nmi.astype(float), annot=True, fmt=".2f", cmap="Greens", vmin=0, vmax=1, ax=ax[1])
    ax[1].set_title("Normalised Mutual Information between methods")
    savefig(fig, "13_method_agreement")

    null = permutation_null_test(X, final, k_star, cfg, sil_idx)
    st_idx = rng.choice(len(X), min(cfg.stability_sample, len(X)), replace=False)
    stab_m, stab_sd, jacc = bootstrap_stability(X[st_idx], k_star, cfg.n_bootstrap, cfg.random_state,
                                                ref_labels=final[st_idx])
    rf = rf_separability(df, final, cfg)
    tests = statistical_tests(df, final)
    tests.to_csv(REP_DIR / "segment_statistical_tests.csv", index=False)
    log.info(f"   null test: z={null['z_score']:.2f}, p={null['empirical_p_value']:.4f}")
    log.info(f"   ensemble bootstrap ARI={stab_m:.3f}±{stab_sd:.3f}; Jaccard per segment={np.round(jacc, 3)}")
    log.info(f"   RF separability acc={rf['cv_accuracy_mean']:.3f}±{rf['cv_accuracy_sd']:.3f} "
             f"(chance={rf['chance_level']:.3f})")

    # --------------------- personas ------------------------------------------
    log.info("STEP 12 | Profiling & personas")
    prof, overall, lift = build_profiles(df, final)
    personas = generate_personas(prof, overall, lift)
    prof.assign(persona=[personas[i]["name"] for i in prof.index]).to_csv(REP_DIR / "segment_profiles.csv")
    lift.to_csv(REP_DIR / "segment_lift_vs_population.csv")
    write_personas_md(personas, overall, ens["chosen"], k_star)
    save_json(personas, "personas.json")
    for i, p in personas.items():
        log.info(f"   S{i} [{p['share']:.1%}] {p['name']} – {p['life_stage']}")

    plot_final(X, df, compare, final, personas, prof, lift, sil_idx, jacc, cfg)

    out = df.copy()
    for nm_, lab in base.items():
        out[f"LBL_{nm_}"] = lab
    out["GMM_MAX_POSTERIOR"] = gm["max_posterior"]
    out["SEGMENT"] = final
    out["PERSONA"] = out["SEGMENT"].map({i: p["name"] for i, p in personas.items()})
    out["ENSEMBLE_ITEM_CONSENSUS"] = ens["item_consensus"]
    out.to_csv(DATA_DIR / "segmented_customers.csv", index=False)

    save_json({
        "n_customers": len(df), "selected_imputer": best_imp, "selected_preprocessing": best_strat,
        "runner_up_preprocessing": second_strat, "feature_space": feat_names, "hopkins": H,
        "K_star": k_star, "k_votes_kmeans": votes, "hierarchical_best_k": hc["k_best"],
        "hierarchical_cophenetic": hc["cophenetic"], "hierarchical_propagation_cv_acc": hc["propagation_cv_acc"],
        "gmm_best_k": gm["k_best"], "gmm_rule": gm["rule"], "famd_components": gm["ncomp"],
        "gmm_mean_max_posterior_at_Kstar": float(gm["max_posterior"].mean()),
        "ensemble_selected": ens["chosen"], "ensemble_candidates": ens["candidates"],
        "ensemble_base_partitions": ens["base_names"],
        "ensemble_mean_item_consensus": float(ens["item_consensus"].mean()),
        "ensemble_cluster_consensus": ens["cluster_consensus"],
        "internal_indices": iv, "permutation_null_test": {k: v for k, v in null.items() if k != "null_values"},
        "ensemble_bootstrap_ari_mean": stab_m, "ensemble_bootstrap_ari_sd": stab_sd,
        "ensemble_clusterwise_jaccard": jacc, "rf_separability": rf,
        "runtime_minutes": (time.time() - t0) / 60,
    }, "validation_report.json")

    log.info("=" * 80)
    log.info(f"DONE in {(time.time() - t0) / 60:.1f} min | outputs -> {cfg.output_dir.resolve()}")
    log.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.getLogger("segmentation").exception(f"Pipeline failed: {e}")
        sys.exit(1)