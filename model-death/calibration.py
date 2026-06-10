"""
calibration.py
==============
Measure and fix the ABSOLUTE accuracy of horizon risks (not just ranking).

A model can rank who-dies-sooner well (high C-index) yet still print wrong
percentages. Two pieces here:

  * km_calibration()  - honest measurement. Bins people by predicted horizon
                        risk and compares the bin's mean prediction to the
                        Kaplan-Meier observed risk, which is the correct
                        estimate under censoring (people lost to follow-up
                        before the horizon are handled, not dropped).

  * HorizonRecalibrator - the fix. Learns a monotone map from predicted risk
                        to true risk at each horizon using IPCW-weighted
                        isotonic regression (inverse-probability-of-censoring
                        weighting, the standard way to build an unbiased
                        binary "did they die by year H" target from censored
                        data). Fit on a held-out calibration split, never test.

Both are model-agnostic: they only need a `.risk_by_horizon(person, horizons)`.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, AalenJohansenFitter
from sklearn.isotonic import IsotonicRegression

from data_loading import FEATURE_COLS


def predicted_horizon_risk(model, df: pd.DataFrame, horizon: int) -> np.ndarray:
    """Vector of predicted P(CVD death by `horizon`) for every row in df."""
    return np.array([model.risk_by_horizon(r, horizons=(horizon,))[horizon]
                     for r in df[FEATURE_COLS].to_dict("records")])


def _km_risk(times: np.ndarray, events: np.ndarray, horizon: float) -> float:
    """Kaplan-Meier estimate of P(event by horizon) = 1 - S(horizon)."""
    if len(times) == 0:
        return np.nan
    kmf = KaplanMeierFitter().fit(times, events)
    return float(1.0 - kmf.survival_function_at_times(horizon).iloc[0])


def km_calibration(model, df: pd.DataFrame, horizon: int, bins: int = 10):
    """Return [(mean_predicted, km_observed, n), ...] over predicted-risk bins."""
    p = predicted_horizon_risk(model, df, horizon)
    t, e = df["time_years"].values, df["event"].values
    order = np.argsort(p)
    rows = []
    for g in np.array_split(order, bins):
        rows.append((float(p[g].mean()), _km_risk(t[g], e[g], horizon), len(g)))
    return rows


def _aj_cif(times: np.ndarray, event_code: np.ndarray, horizon: float) -> float:
    """Aalen-Johansen observed CIF for CVD death (code 1) at the horizon.

    event_code: 0 = censored, 1 = CVD death, 2 = non-CVD death. This is the
    correct 'observed' risk for a competing-risks model (KM would overstate it).
    """
    if len(times) == 0 or (event_code == 1).sum() == 0:
        return 0.0 if len(times) else np.nan
    # AJ requires no tied event times; jitter microscopically to break ties.
    rng = np.random.default_rng(0)
    t = times + rng.uniform(0, 1e-6, size=len(times))
    ajf = AalenJohansenFitter(calculate_variance=False)
    ajf.fit(t, event_code, event_of_interest=1)
    cif = ajf.cumulative_density_.iloc[:, 0]
    at = cif.reindex(sorted(set(cif.index) | {horizon})).ffill()
    return float(at.loc[horizon])


def cif_calibration(model, df: pd.DataFrame, horizon: int, bins: int = 10):
    """Calibration for a competing-risks model: predicted CIF vs AJ-observed CIF.

    Needs df to carry 'event_other' so the competing event (code 2) is known.
    """
    p = predicted_horizon_risk(model, df, horizon)
    t = df["time_years"].values
    code = np.where(df["event"].values == 1, 1,
                    np.where(df["event_other"].values == 1, 2, 0))
    order = np.argsort(p)
    rows = []
    for g in np.array_split(order, bins):
        rows.append((float(p[g].mean()), _aj_cif(t[g], code[g], horizon), len(g)))
    return rows


def ici(rows) -> float:
    """Integrated Calibration Index: mean |predicted - observed| across bins."""
    return float(np.mean([abs(p - o) for p, o, _ in rows if not np.isnan(o)]))


def _ipcw_binary_target(df: pd.DataFrame, preds: np.ndarray, horizon: float,
                        g_floor: float = 0.05):
    """Build an unbiased binary 'died by horizon' target with IPCW weights.

    Events on/before horizon -> label 1, weight 1/G(event time).
    Survivors past horizon    -> label 0, weight 1/G(horizon).
    Censored before horizon   -> dropped (their status is unknown).
    G is the Kaplan-Meier survival of the CENSORING distribution.
    """
    t, e = df["time_years"].values, df["event"].values
    gkm = KaplanMeierFitter().fit(t, 1 - e)   # censoring = event flipped

    def G(x):
        return max(float(gkm.survival_function_at_times(x).iloc[0]), g_floor)

    P, Y, W = [], [], []
    for i in range(len(t)):
        if e[i] == 1 and t[i] <= horizon:
            P.append(preds[i]); Y.append(1.0); W.append(1.0 / G(t[i]))
        elif t[i] >= horizon:
            P.append(preds[i]); Y.append(0.0); W.append(1.0 / G(horizon))
        # else: censored before horizon -> excluded
    return np.array(P), np.array(Y), np.array(W)


class HorizonRecalibrator:
    """Wrap a fitted survival model; remap its horizon risks to be calibrated.

    Ranking (C-index) is unchanged — isotonic is monotone — but the absolute
    percentages are corrected. `survival_function` / `median_survival_years`
    delegate to the base model (the curve itself is not recalibrated; the app
    leads with the recalibrated horizon risks).
    """

    def __init__(self, base, horizons=(1, 5, 10)):
        self.base = base
        self.horizons = tuple(horizons)
        self.maps: dict[int, IsotonicRegression] = {}

    def fit(self, cal_df: pd.DataFrame) -> "HorizonRecalibrator":
        for h in self.horizons:
            p = predicted_horizon_risk(self.base, cal_df, h)
            P, Y, W = _ipcw_binary_target(cal_df, p, h)
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(P, Y, sample_weight=W)
            self.maps[h] = iso
        return self

    def risk_by_horizon(self, person: dict, horizons=None) -> dict:
        horizons = horizons or self.horizons
        out = {}
        for h in horizons:
            p = self.base.risk_by_horizon(person, horizons=(h,))[h]
            out[h] = (float(self.maps[h].predict([p])[0])
                      if h in self.maps else float(p))
        return out

    def survival_function(self, person: dict, times=None):
        return self.base.survival_function(person, times=times)

    def median_survival_years(self, person: dict):
        return self.base.median_survival_years(person)
