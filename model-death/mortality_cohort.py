"""
mortality_cohort.py
===================
Build the REAL training set for the death-learned model: NHANES cycles that
have actual mortality follow-up, linked to the NCHS Public-Use Linked Mortality
Files (deaths observed through 31 Dec 2019).

Why these cycles (not the column model's 2021-2023):
  The 2021-2023 (_L) cycle has NO mortality follow-up yet, so it cannot teach a
  death model anything. We pool the six cycles 2007-2018, which have up to ~12
  years of follow-up and enough CVD deaths to LEARN per-feature weights instead
  of borrowing them. Variable names drift across cycles, so each feature is
  resolved from a list of aliases.

This module: downloads what it needs, loads each cycle into the 27-feature
schema (column_model_spec.FEATURES), links mortality, and returns one pooled
survival frame ready for model.py.
"""

from __future__ import annotations
import os
import ssl
import time
import urllib.request

import numpy as np
import pandas as pd

from column_model_spec import FEATURES, MEDIANS
from data_loading import (load_mortality, build_survival_frame,
                          _yes_no, _smoking_status)

# --- Cohort definition ----------------------------------------------------

# cycle first-year -> NHANES file suffix.
CYCLES = {2007: "E", 2009: "F", 2011: "G",
          2013: "H", 2015: "I", 2017: "J"}

# NHANES component stems we pull (a few are absent in early cycles; skipped).
COMPONENTS = ["DEMO", "BPX", "BMX", "TCHOL", "HDL", "TRIGLY", "GHB", "GLU",
              "INS", "DIQ", "HSCRP", "BIOPRO", "ALB_CR", "CBC", "SMQ",
              "PAQ", "SLQ", "BPQ"]

DATA_DIR = "./nhanes_mortality"
NHANES_URL = ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/"
              "{year}/DataFiles/{stem}_{suf}.xpt")
MORT_URL = ("https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/"
            "linked_mortality/NHANES_{y1}_{y2}_MORT_2019_PUBLIC.dat")


def _ssl_ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl._create_unverified_context()


def _fetch(url: str, dest: str, ctx) -> str:
    """Download url -> dest. Returns 'have' | 'ok' | 'skip(<code>)'."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "have"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60, context=ctx) as r, \
                open(dest, "wb") as fh:
            fh.write(r.read())
        return "ok"
    except Exception as e:
        if os.path.exists(dest):
            os.remove(dest)
        return f"skip({getattr(e, 'code', type(e).__name__)})"


def download(data_dir: str = DATA_DIR) -> None:
    """Fetch every component + mortality file we need. Idempotent; 404s skip."""
    os.makedirs(data_dir, exist_ok=True)
    ctx = _ssl_ctx()
    for year, suf in CYCLES.items():
        y2 = year + 1
        for stem in COMPONENTS:
            dest = os.path.join(data_dir, f"{stem}_{suf}.XPT")
            status = _fetch(NHANES_URL.format(year=year, stem=stem, suf=suf),
                            dest, ctx)
            if status != "have":
                print(f"  {stem}_{suf}: {status}")
            time.sleep(0.05)
        mdest = os.path.join(data_dir, f"MORT_{year}_{y2}.dat")
        ms = _fetch(MORT_URL.format(y1=year, y2=y2), mdest, ctx)
        print(f"cycle {year}-{y2} mortality: {ms}")


# --- Cross-cycle feature engineering -------------------------------------

# feature -> ordered list of NHANES variable aliases (first present wins).
_ALIASES = {
    "age":        ["RIDAGEYR"],
    "race":       ["RIDRETH3", "RIDRETH1"],
    "education":  ["DMDEDUC2"],
    "income":     ["INDFMPIR"],
    "bmi":        ["BMXBMI"],
    "waist":      ["BMXWAIST"],
    "total_chol": ["LBXTC"],
    "hdl":        ["LBDHDD", "LBXHDD"],
    "ldl":        ["LBDLDL"],
    "hba1c":      ["LBXGH"],
    "glucose":    ["LBXGLU"],
    "insulin":    ["LBXIN"],
    "crp":        ["LBXHSCRP"],          # hsCRP only; skip old LBXCRP (units)
    "creatinine": ["LBXSCR"],
    "bun":        ["LBXSBU"],
    "uric_acid":  ["LBXSUA"],
    "uacr":       ["URDACT"],
    "wbc":        ["LBXWBCSI"],
    "sedentary_min": ["PAD680"],
    "sleep_hours":   ["SLD012", "SLD010H"],
}
_SBP_VARS = ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"]   # manual readings
_DBP_VARS = ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"]


def _read_all_components(data_dir: str, suf: str) -> pd.DataFrame:
    """Outer-merge every present {stem}_{suf}.XPT on SEQN into one wide frame."""
    wide = None
    for stem in COMPONENTS:
        path = os.path.join(data_dir, f"{stem}_{suf}.XPT")
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            continue
        df = pd.read_sas(path, format="xport")
        df = df.loc[:, ~df.columns.duplicated()].copy()
        df["SEQN"] = df["SEQN"].astype("Int64")
        wide = df if wide is None else wide.merge(
            df, on="SEQN", how="outer", suffixes=("", f"_{stem}"))
    if wide is None:
        raise FileNotFoundError(f"No NHANES components for suffix _{suf}")
    return wide


def _first_alias(wide: pd.DataFrame, names: list[str]) -> pd.Series:
    for n in names:
        if n in wide.columns:
            return wide[n]
    return pd.Series(np.nan, index=wide.index)


def engineer_cycle(data_dir: str, suf: str) -> pd.DataFrame:
    """One cycle's wide frame -> the 27-feature model schema keyed by SEQN."""
    w = _read_all_components(data_dir, suf)
    out = pd.DataFrame({"SEQN": w["SEQN"]})

    for feat, aliases in _ALIASES.items():
        out[feat] = pd.to_numeric(_first_alias(w, aliases), errors="coerce")

    sbp_cols = [c for c in _SBP_VARS if c in w.columns]
    dbp_cols = [c for c in _DBP_VARS if c in w.columns]
    out["sbp"] = w[sbp_cols].replace(0, np.nan).mean(axis=1) if sbp_cols else np.nan
    out["dbp"] = w[dbp_cols].replace(0, np.nan).mean(axis=1) if dbp_cols else np.nan

    out["non_hdl"] = out["total_chol"] - out["hdl"]
    out["diabetes_dx"] = _yes_no(_first_alias(w, ["DIQ010"]))
    out["htn_dx"] = _yes_no(_first_alias(w, ["BPQ020"]))
    out["highchol_dx"] = _yes_no(_first_alias(w, ["BPQ080"]))
    out["smoking"] = _smoking_status(_first_alias(w, ["SMQ020"]),
                                     _first_alias(w, ["SMQ040"]))
    return out


def build_pooled_frame(data_dir: str = DATA_DIR,
                       min_age: int = 40) -> pd.DataFrame:
    """Pool all cycles into one survival frame (features + event + time_years).

    Each cycle is linked to its own mortality file, missing labs are filled
    with the column model's medians, then cycles are concatenated.
    """
    frames = []
    for year, suf in CYCLES.items():
        y2 = year + 1
        mpath = os.path.join(data_dir, f"MORT_{year}_{y2}.dat")
        if not os.path.exists(mpath):
            print(f"  (no mortality file for {year}-{y2}, skipping)")
            continue
        preds = engineer_cycle(data_dir, suf)
        mort = load_mortality(mpath)
        sf = build_survival_frame(preds, mort, min_age=min_age)
        sf["cycle"] = year
        frames.append(sf)
        print(f"  {year}-{y2}: n={len(sf)}, "
              f"CVD deaths={int(sf['event'].sum())} "
              f"({sf['event'].mean():.1%})")

    if not frames:
        raise SystemExit("No cycles built. Run download() first.")
    pooled = pd.concat(frames, ignore_index=True)
    return pooled


if __name__ == "__main__":
    print("Downloading NHANES cohort + mortality files...")
    download()
    print("\nBuilding pooled survival frame...")
    df = build_pooled_frame()
    print(f"\nPOOLED: n={len(df)}, CVD deaths={int(df['event'].sum())} "
          f"({df['event'].mean():.1%}), "
          f"median follow-up={df['time_years'].median():.1f} y")
