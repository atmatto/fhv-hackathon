"""
data_loading.py
===============
Load NHANES predictor files + the NCHS Public-Use Linked Mortality File (LMF),
merge them on SEQN, and construct a cause-specific cardiovascular-death
survival outcome (event indicator + follow-up time in years).

The PREDICTOR columns and their defaults are no longer chosen here by hand.
They are the 27 features the column model selected (see `column_model_spec.py`,
transcribed from `../column_model/cvd_model_report.md`). This module's job is
to produce those 27 columns from the raw NHANES 2021-2023 (`_L`) files, mapped
according to `../column_model/cvd_candidate_features.md`.

The real input files are downloaded directly from the CDC:
  * NHANES components (.XPT, SAS XPORT):  https://wwwn.cdc.gov/nchs/nhanes/
  * Linked Mortality Files (.dat, fixed-width ASCII):
        https://www.cdc.gov/nchs/data-linkage/mortality-public.htm

This module does not download anything; it reads files already on disk.

Mortality file layout (public-use LMF, verified against the NCHS read-in
programs). Column positions below are 0-indexed half-open [start, end) for
pandas.read_fwf, converted from the documented 1-indexed ranges:

    SEQN          1-6    -> (0, 6)
    ELIGSTAT      15     -> (14, 15)
    MORTSTAT      16     -> (15, 16)
    UCOD_LEADING  17-19  -> (16, 19)
    DIABETES      20     -> (19, 20)
    HYPERTEN      21     -> (20, 21)
    PERMTH_INT    43-45  -> (42, 45)
    PERMTH_EXM    46-48  -> (45, 48)

UCOD_LEADING == 1  means "Diseases of heart" (recode of UCOD_113 codes 54-64).
MORTSTAT: 0 = assumed alive, 1 = assumed deceased.
PERMTH_EXM: person-months of follow-up from the MEC exam date.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

from column_model_spec import FEATURES, MEDIANS

# The survival model's predictors ARE the column model's selected features.
FEATURE_COLS = FEATURES

# --- Mortality file -------------------------------------------------------

_LMF_COLSPECS = [(0, 6), (14, 15), (15, 16), (16, 19),
                 (19, 20), (20, 21), (42, 45), (45, 48)]
_LMF_NAMES = ["SEQN", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING",
              "DIABETES", "HYPERTEN", "PERMTH_INT", "PERMTH_EXM"]

# UCOD_LEADING recode value for "Diseases of heart".
HEART_DISEASE_CODE = 1


def load_mortality(path: str) -> pd.DataFrame:
    """Read a public-use LMF .dat (fixed-width ASCII) into a tidy frame."""
    mort = pd.read_fwf(
        path,
        colspecs=_LMF_COLSPECS,
        names=_LMF_NAMES,
        na_values=["", "."],
    )
    mort["SEQN"] = mort["SEQN"].astype("Int64")
    return mort


# --- NHANES predictor components -----------------------------------------

# Map: NHANES 2021-2023 (_L) component stem -> raw variables to pull from it.
# Variable names follow cvd_candidate_features.md; confirm against the official
# _L codebook before a real run, as a few names shift between cycles.
_DEFAULT_VARS = {
    "DEMO":   ["RIDAGEYR", "RIDRETH3", "DMDEDUC2", "INDFMPIR"],
    "BPXO":   ["BPXOSY1", "BPXOSY2", "BPXOSY3",
               "BPXODI1", "BPXODI2", "BPXODI3"],
    "BMX":    ["BMXBMI", "BMXWAIST"],
    "TCHOL":  ["LBXTC"],
    "HDL":    ["LBDHDD"],
    "TRIGLY": ["LBDLDL"],
    "GHB":    ["LBXGH"],
    "GLU":    ["LBXGLU"],
    "INS":    ["LBXIN"],
    "DIQ":    ["DIQ010"],
    "HSCRP":  ["LBXHSCRP"],
    "BIOPRO": ["LBXSCR", "LBXSBU", "LBXSUA"],
    "ALB_CR": ["URDACT"],
    "CBC":    ["LBXWBCSI"],
    "SMQ":    ["SMQ020", "SMQ040"],
    "PAQ":    ["PAD680"],
    "SLQ":    ["SLD012"],
    "BPQ":    ["BPQ020", "BPQ080"],
}


def _read_xpt(path: str, columns: list[str]) -> pd.DataFrame:
    df = pd.read_sas(path, format="xport")
    keep = ["SEQN"] + [c for c in columns if c in df.columns]
    df = df[keep].copy()
    df["SEQN"] = df["SEQN"].astype("Int64")
    return df


def _yes_no(series: pd.Series) -> pd.Series:
    """NHANES 1=yes / 2=no (7,9 = refused/don't know) -> 1.0 / 0.0 / NaN."""
    return series.map({1: 1.0, 2: 0.0})


def _smoking_status(smq020: pd.Series, smq040: pd.Series) -> pd.Series:
    """Combine SMQ020 (ever 100 cigs) + SMQ040 (now) into 0/1/2.

    0 never, 1 former, 2 current  (matches the report's `smoking` encoding).
    """
    status = pd.Series(np.nan, index=smq020.index, dtype="float")
    never = smq020 == 2
    ever = smq020 == 1
    status[never] = 0.0
    status[ever & smq040.isin([1, 2])] = 2.0   # every/some days -> current
    status[ever & (smq040 == 3)] = 1.0          # not at all -> former
    return status


def load_predictors(component_files: dict[str, str]) -> pd.DataFrame:
    """
    component_files: {"DEMO": "DEMO_L.XPT", "BPXO": "BPXO_L.XPT", ...}

    Returns a feature frame keyed by SEQN with the 27 model-ready columns named
    in `column_model_spec.FEATURES`. Any component not provided leaves its
    columns missing; build_survival_frame() fills/handles them downstream.
    """
    frames = []
    for stem, path in component_files.items():
        wanted = _DEFAULT_VARS.get(stem, [])
        frames.append(_read_xpt(path, wanted))

    if not frames:
        raise ValueError("No NHANES component files were provided.")

    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on="SEQN", how="outer")

    def col(name):
        return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

    out = pd.DataFrame({"SEQN": df["SEQN"]})

    # Demographics
    out["age"] = col("RIDAGEYR")
    out["race"] = col("RIDRETH3")
    out["education"] = col("DMDEDUC2")
    out["income"] = col("INDFMPIR")

    # Blood pressure (mean of the available oscillometric readings)
    sbp_cols = [c for c in ["BPXOSY1", "BPXOSY2", "BPXOSY3"] if c in df.columns]
    dbp_cols = [c for c in ["BPXODI1", "BPXODI2", "BPXODI3"] if c in df.columns]
    out["sbp"] = df[sbp_cols].mean(axis=1) if sbp_cols else np.nan
    out["dbp"] = df[dbp_cols].mean(axis=1) if dbp_cols else np.nan

    # Body
    out["bmi"] = col("BMXBMI")
    out["waist"] = col("BMXWAIST")

    # Lipids (non-HDL is the documented derived feature: total - HDL)
    out["total_chol"] = col("LBXTC")
    out["hdl"] = col("LBDHDD")
    out["ldl"] = col("LBDLDL")
    out["non_hdl"] = out["total_chol"] - out["hdl"]

    # Glucose / metabolic
    out["hba1c"] = col("LBXGH")
    out["glucose"] = col("LBXGLU")
    out["insulin"] = col("LBXIN")
    out["diabetes_dx"] = _yes_no(col("DIQ010"))

    # Inflammation / kidney / other labs
    out["crp"] = col("LBXHSCRP")
    out["creatinine"] = col("LBXSCR")
    out["bun"] = col("LBXSBU")
    out["uric_acid"] = col("LBXSUA")
    out["uacr"] = col("URDACT")
    out["wbc"] = col("LBXWBCSI")

    # Behaviors
    out["smoking"] = _smoking_status(col("SMQ020"), col("SMQ040"))
    out["sedentary_min"] = col("PAD680")
    out["sleep_hours"] = col("SLD012")

    # Conditions / history (risk factors, not CVD itself)
    out["htn_dx"] = _yes_no(col("BPQ020"))
    out["highchol_dx"] = _yes_no(col("BPQ080"))

    return out


# --- Build the survival dataset ------------------------------------------

def build_survival_frame(
    predictors: pd.DataFrame,
    mortality: pd.DataFrame,
    min_age: int = 40,
    time_var: str = "PERMTH_EXM",
    require_complete: bool = False,
) -> pd.DataFrame:
    """
    Merge predictors with mortality and build a CAUSE-SPECIFIC CVD-death
    survival outcome:

        event = 1  if MORTSTAT == 1 AND UCOD_LEADING == heart disease
        event = 0  otherwise  (alive at end of follow-up, OR died of another
                               cause -> censored; see note on competing risks)

        time  = follow-up time in YEARS = PERMTH_EXM / 12

    Non-CVD deaths are treated as censored (standard cause-specific approach).
    A methodologically stricter alternative is a competing-risks model
    (Fine-Gray subdistribution hazard); see README.

    min_age defaults to 40 to match the column model (the heart questions
    target adults 40+). Missing predictors are filled with the column model's
    training medians (column_model_spec.MEDIANS) so a row is never dropped just
    for a missing lab; pass require_complete=True to drop incomplete rows
    instead.
    """
    df = predictors.merge(mortality, on="SEQN", how="inner")

    # Eligible adults with non-missing follow-up only.
    df = df[df["ELIGSTAT"] == 1]
    df = df[df["age"] >= min_age]
    df = df[df[time_var].notna() & (df[time_var] > 0)]

    deceased = df["MORTSTAT"] == 1
    df["event"] = (deceased &
                   (df["UCOD_LEADING"] == HEART_DISEASE_CODE)).astype(int)
    # Competing event: died of any non-CVD cause (incl. deceased w/ unknown
    # cause). Used by the competing-risks model; ignored by the others.
    df["event_other"] = (deceased &
                         (df["UCOD_LEADING"] != HEART_DISEASE_CODE)).astype(int)
    df["time_years"] = df[time_var] / 12.0

    if require_complete:
        df = df.dropna(subset=FEATURE_COLS)
    else:
        df = df.fillna({f: MEDIANS[f] for f in FEATURE_COLS})

    cols = ["SEQN"] + FEATURE_COLS + ["event", "event_other", "time_years"]
    df = df[cols].dropna(subset=["age"])  # age must be real
    return df.reset_index(drop=True)
