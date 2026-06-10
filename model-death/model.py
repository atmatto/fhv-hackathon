"""
model.py
========
Train and evaluate survival models for time-to-cardiovascular-death, and turn
a fitted model into the predictions the application needs:

  * a full predicted survival curve S(t) for an individual,
  * the probability of CVD death by a given horizon (e.g. 5 / 10 years),
  * a median survival time WHERE it is defined.

How the column model feeds in
-----------------------------
The predictors are the 27 columns the column model selected, and by default the
Cox model does NOT re-learn a weight for each of them. Instead it regresses
CVD-death timing on a single covariate: the importance-weighted CVD risk index
from `column_model_spec.cvd_risk_index`. That index already encodes the column
model's gains as the per-feature weights, so the survival fit only has to learn
two things from the mortality data: the baseline hazard and one overall scaling
coefficient. This is the "use the report as the basis to calculate mortality"
path. Set `use_importance_index=False` to instead let Cox fit its own
coefficient on every feature (the column model then only chose the columns).

Two models are provided so the report can compare a classical statistical
model against a machine-learning one:

  * Cox proportional hazards  (lifelines.CoxPHFitter)
  * Random Survival Forest    (sksurv.ensemble.RandomSurvivalForest)

Honest note on "how far in the future will they die":
For a CAUSE-SPECIFIC outcome like CVD death, most people never reach a 50%
cumulative risk within the observed follow-up, so a literal median survival
time is undefined for them. That is a property of the data, not a bug. The
app therefore reports horizon risks (e.g. "12% chance of CVD death within 10
years") as the primary output, and a median time only when the curve actually
crosses 0.5.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

# scikit-survival is imported lazily inside RSFModel only. That keeps the
# prediction path (CompetingRisksModel + lifelines) free of the heavy
# scikit-survival dependency, so deploying the app does not require it.
from column_model_spec import (FEATURES as FEATURE_COLS,
                               cvd_risk_index, fill_defaults)

# Name of the single derived covariate built from the column model report.
INDEX_COL = "cvd_index"


def add_risk_index(df: pd.DataFrame) -> pd.DataFrame:
    """Append the importance-weighted CVD risk index column to a frame."""
    out = df.copy()
    out[INDEX_COL] = [cvd_risk_index(row) for row in
                      df[FEATURE_COLS].to_dict("records")]
    return out


# --- Cox proportional hazards --------------------------------------------

class CoxModel:
    def __init__(self, penalizer: float = 0.01,
                 use_importance_index: bool = True,
                 features: list[str] | None = None):
        """
        use_importance_index : fit on the single column-model risk index.
        features : when NOT using the index, the explicit feature subset to fit
                   (defaults to all 27). Lets callers drop collinear or
                   near-constant columns before a full death-learned fit.
        """
        self.cph = CoxPHFitter(penalizer=penalizer)
        self.use_importance_index = use_importance_index
        if use_importance_index:
            self.features = [INDEX_COL]
        else:
            self.features = list(features) if features else FEATURE_COLS

    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        return add_risk_index(df) if self.use_importance_index else df

    def fit(self, train: pd.DataFrame) -> "CoxModel":
        train = self._design(train)
        cols = self.features + ["time_years", "event"]
        self.cph.fit(train[cols], duration_col="time_years",
                     event_col="event")
        return self

    def concordance(self, test: pd.DataFrame) -> float:
        # Higher partial hazard -> shorter survival, so negate for concordance.
        test = self._design(test)
        risk = self.cph.predict_partial_hazard(test[self.features])
        return concordance_index(test["time_years"], -risk, test["event"])

    def _person_row(self, person: dict) -> pd.DataFrame:
        """One-row design matrix for a person dict (missing fields -> median)."""
        if self.use_importance_index:
            return pd.DataFrame([{INDEX_COL: cvd_risk_index(person)}])
        from column_model_spec import fill_defaults
        return pd.DataFrame([fill_defaults(person)])[self.features]

    def survival_function(self, person: dict, times=None) -> pd.Series:
        x = self._person_row(person)
        sf = self.cph.predict_survival_function(x, times=times)
        return sf.iloc[:, 0]

    def risk_by_horizon(self, person: dict, horizons=(1, 5, 10)) -> dict:
        sf = self.survival_function(person, times=list(horizons))
        return {h: float(1.0 - sf.loc[h]) for h in horizons}

    def median_survival_years(self, person: dict):
        x = self._person_row(person)
        med = self.cph.predict_median(x)
        med = med.iloc[0] if hasattr(med, "iloc") else float(med)
        return None if np.isinf(med) else float(med)

    def hazard_ratios(self) -> pd.Series:
        return np.exp(self.cph.params_)


# --- Competing-risks model (the accurate one) ----------------------------

class CompetingRisksModel:
    """Cause-specific Cox for CVD death AND for non-CVD death, combined into a
    proper cumulative incidence function (CIF) for CVD death.

    Why this is more accurate: the plain Cox/RSF treat non-CVD deaths as
    censored, so 1 - S(t) answers "risk of CVD death IF nothing else could kill
    you first" -- which overstates real risk, especially for the old (who have
    high competing mortality). The CIF answers the real-world question: "chance
    of dying OF CVD by year t, given you might die of something else first."

        CIF_cvd(t) = sum_{u<=t}  S(u-) * dH1(u|x)

    where H1 is the CVD cause-specific cumulative hazard, H2 the non-CVD one,
    and S = exp(-(H1+H2)) is overall survival from both causes.

    Interface matches the other models (risk_by_horizon / survival_function /
    median_survival_years), with survival_function returning 1 - CIF so the
    app's existing curve still means "probability of not having died of CVD".
    """

    def __init__(self, penalizer: float = 0.01, features: list[str] | None = None):
        self.cph_cvd = CoxPHFitter(penalizer=penalizer)
        self.cph_other = CoxPHFitter(penalizer=penalizer)
        self.features = list(features) if features else FEATURE_COLS

    def fit(self, train: pd.DataFrame) -> "CompetingRisksModel":
        cols = self.features + ["time_years"]
        cvd = train[cols].copy(); cvd["e"] = train["event"].values
        oth = train[cols].copy(); oth["e"] = train["event_other"].values
        self.cph_cvd.fit(cvd, "time_years", "e")
        self.cph_other.fit(oth, "time_years", "e")
        # Common time grid + baseline cumulative hazard increments.
        h1 = self.cph_cvd.baseline_cumulative_hazard_.iloc[:, 0]
        h2 = self.cph_other.baseline_cumulative_hazard_.iloc[:, 0]
        self._grid = np.array(sorted(set(h1.index) | set(h2.index)))
        self._H1_0 = h1.reindex(self._grid, method="ffill").fillna(0.0).values
        self._H2_0 = h2.reindex(self._grid, method="ffill").fillna(0.0).values
        self._dH1_0 = np.diff(self._H1_0, prepend=0.0)
        return self

    def _cif_curve(self, person: dict) -> np.ndarray:
        x = pd.DataFrame([fill_defaults(person)])[self.features]
        ph1 = float(self.cph_cvd.predict_partial_hazard(x).iloc[0])
        ph2 = float(self.cph_other.predict_partial_hazard(x).iloc[0])
        S = np.exp(-(self._H1_0 * ph1 + self._H2_0 * ph2))
        S_prev = np.concatenate([[1.0], S[:-1]])
        return np.cumsum(S_prev * self._dH1_0 * ph1)   # CIF at each grid time

    def _at(self, cif: np.ndarray, times) -> pd.Series:
        idx = np.searchsorted(self._grid, np.asarray(times, float), side="right") - 1
        vals = np.where(idx >= 0, cif[idx.clip(min=0)], 0.0)
        return pd.Series(vals, index=list(times))

    def cif_by_horizon(self, person: dict, horizons=(1, 5, 10)) -> dict:
        cif = self._cif_curve(person)
        s = self._at(cif, list(horizons))
        return {h: float(s.loc[h]) for h in horizons}

    # --- shared interface (CIF-based) ---
    def risk_by_horizon(self, person: dict, horizons=(1, 5, 10)) -> dict:
        return self.cif_by_horizon(person, horizons)

    def survival_function(self, person: dict, times=None) -> pd.Series:
        cif = self._cif_curve(person)
        if times is None:
            times = self._grid
        return 1.0 - self._at(cif, list(times))

    def median_survival_years(self, person: dict):
        cif = self._cif_curve(person)
        crossed = np.where(cif >= 0.5)[0]
        return None if len(crossed) == 0 else float(self._grid[crossed[0]])

    def concordance(self, test: pd.DataFrame) -> float:
        risk = np.array([self.cif_by_horizon(r, (10,))[10]
                         for r in test[self.features].to_dict("records")])
        return concordance_index(test["time_years"], -risk, test["event"])


# --- Random Survival Forest (ML comparison) ------------------------------

class RSFModel:
    def __init__(self, n_estimators: int = 200, min_samples_leaf: int = 30,
                 random_state: int = 0):
        from sksurv.ensemble import RandomSurvivalForest  # lazy: training only
        self.rsf = RandomSurvivalForest(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=random_state,
        )
        self.features = FEATURE_COLS
        self._train_max_time = None

    @staticmethod
    def _to_surv(df: pd.DataFrame):
        from sksurv.util import Surv
        return Surv.from_arrays(event=df["event"].astype(bool),
                                time=df["time_years"].values)

    def fit(self, train: pd.DataFrame) -> "RSFModel":
        self.rsf.fit(train[self.features].values, self._to_surv(train))
        self._train_max_time = train["time_years"].max()
        return self

    def concordance(self, test: pd.DataFrame) -> float:
        return self.rsf.score(test[self.features].values, self._to_surv(test))

    def time_dependent_auc(self, train: pd.DataFrame, test: pd.DataFrame,
                           times):
        from sksurv.metrics import cumulative_dynamic_auc
        risk = self.rsf.predict(test[self.features].values)
        auc, mean_auc = cumulative_dynamic_auc(
            self._to_surv(train), self._to_surv(test), risk, times)
        return auc, mean_auc

    def _person_row(self, person: dict):
        from column_model_spec import fill_defaults
        return pd.DataFrame([fill_defaults(person)])[self.features].values

    def survival_function(self, person: dict, times=None) -> pd.Series:
        x = self._person_row(person)
        fn = self.rsf.predict_survival_function(x, return_array=False)[0]
        if times is None:
            times = fn.x
        vals = fn(times)
        return pd.Series(vals, index=times)

    def risk_by_horizon(self, person: dict, horizons=(1, 5, 10)) -> dict:
        sf = self.survival_function(person, times=list(horizons))
        return {h: float(1.0 - sf.loc[h]) for h in horizons}
