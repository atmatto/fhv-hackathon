"""
column_model_spec.py
====================
The bridge between the two models.

The column model (`../column_model/`) is an XGBoost CLASSIFIER for *prevalent*
CVD. It already did the expensive work of deciding WHICH inputs matter and HOW
MUCH each one matters, and wrote the answer to `cvd_model_report.md`. This file
transcribes that report so the mortality (survival) model can reuse it instead
of guessing its own short feature list.

Three things come straight from the report, verbatim:

  * IMPORTANCE  - the 27 features RFECV kept, with their normalized XGBoost
                  gain (how much each column matters). Sums to ~1.
  * MEDIANS     - the imputer's learned median for each feature; the report
                  says to use these as defaults for any blank web-app input.
  * DESCRIPTION - the human-readable meaning of each column.

Two things the report does NOT contain, which a *risk score* needs, are added
here with clear provenance:

  * DIRECTION   - the SIGN of each feature's effect on risk. XGBoost gain is
                  unsigned (it says a column matters, not which way), so the
                  clinically-known direction is supplied: +1 = higher value
                  means higher risk, -1 = protective, 0 = nominal/no monotone
                  direction (e.g. race code) so it is left out of the index.
  * SCALE       - an approximate population standard deviation per feature,
                  used only to put columns on a comparable footing before the
                  importances weight them. The Cox fit re-calibrates the final
                  magnitude, so these only need to be roughly right.

`cvd_risk_index()` combines all of the above into ONE number: an
importance-weighted, direction-aware, median-centered linear score. That single
index is what the survival model regresses CVD-death timing on, so the column
model's importances drive the mortality estimate while the mortality data only
sets the baseline hazard and overall scale.
"""

from __future__ import annotations

# --- Straight from cvd_model_report.md -----------------------------------

# Feature -> normalized XGBoost gain (the "Feature importance" table).
IMPORTANCE: dict[str, float] = {
    "htn_dx":        0.3157,
    "diabetes_dx":   0.0689,
    "highchol_dx":   0.0565,
    "age":           0.0525,
    "smoking":       0.0514,
    "total_chol":    0.0387,
    "non_hdl":       0.0291,
    "education":     0.0270,
    "income":        0.0261,
    "uacr":          0.0211,
    "creatinine":    0.0211,
    "sleep_hours":   0.0207,
    "ldl":           0.0199,
    "race":          0.0195,
    "sbp":           0.0193,
    "hdl":           0.0192,
    "uric_acid":     0.0191,
    "sedentary_min": 0.0188,
    "dbp":           0.0187,
    "wbc":           0.0180,
    "hba1c":         0.0180,
    "waist":         0.0177,
    "bmi":           0.0177,
    "crp":           0.0176,
    "insulin":       0.0176,
    "bun":           0.0150,
    "glucose":       0.0150,
}

# Feature -> training median (the "Default values" table). Defaults for blanks.
MEDIANS: dict[str, float] = {
    "age":           63.000,
    "race":          3.000,
    "education":     4.000,
    "income":        2.890,
    "sbp":           123.667,
    "dbp":           74.667,
    "bmi":           28.700,
    "waist":         101.500,
    "total_chol":    188.000,
    "hdl":           53.000,
    "ldl":           106.000,
    "non_hdl":       131.000,
    "hba1c":         5.600,
    "glucose":       103.000,
    "insulin":       9.410,
    "diabetes_dx":   0.000,
    "crp":           1.870,
    "creatinine":    0.860,
    "bun":           15.000,
    "uric_acid":     5.100,
    "uacr":          8.870,
    "wbc":           6.600,
    "htn_dx":        0.000,
    "highchol_dx":   1.000,
    "smoking":       0.000,
    "sedentary_min": 300.000,
    "sleep_hours":   8.000,
}

# Feature -> description (the "Feature importance" table, Description column).
DESCRIPTION: dict[str, str] = {
    "htn_dx":        "Diagnosed high blood pressure (0/1)",
    "diabetes_dx":   "Diagnosed diabetes (0/1)",
    "highchol_dx":   "Diagnosed high cholesterol (0/1)",
    "age":           "Age in years",
    "smoking":       "Smoking status (0 never, 1 former, 2 current)",
    "total_chol":    "Total cholesterol (mg/dL)",
    "non_hdl":       "Non-HDL cholesterol (mg/dL)",
    "education":     "Education level",
    "income":        "Income-to-poverty ratio",
    "uacr":          "Urine albumin-to-creatinine ratio",
    "creatinine":    "Serum creatinine",
    "sleep_hours":   "Usual sleep hours",
    "ldl":           "LDL cholesterol (mg/dL)",
    "race":          "Race / ethnicity code",
    "sbp":           "Systolic blood pressure (mean of 3)",
    "hdl":           "HDL cholesterol (mg/dL)",
    "uric_acid":     "Uric acid",
    "sedentary_min": "Sedentary minutes/day",
    "dbp":           "Diastolic blood pressure (mean of 3)",
    "wbc":           "White blood cell count",
    "hba1c":         "HbA1c (%)",
    "waist":         "Waist circumference (cm)",
    "bmi":           "Body mass index",
    "crp":           "High-sensitivity CRP (mg/L)",
    "insulin":       "Insulin",
    "bun":           "Blood urea nitrogen",
    "glucose":       "Fasting glucose (mg/dL)",
}

# Features, ordered most- to least-important (the report's rank order).
FEATURES: list[str] = list(IMPORTANCE.keys())


# --- Added here (not in the report); see module docstring ----------------

# Sign of each feature's effect on cardiovascular risk.
#   +1 higher value -> higher risk,  -1 protective,  0 nominal -> excluded.
DIRECTION: dict[str, int] = {
    "htn_dx":        +1,
    "diabetes_dx":   +1,
    "highchol_dx":   +1,
    "age":           +1,
    "smoking":       +1,
    "total_chol":    +1,
    "non_hdl":       +1,
    "education":     -1,   # more education -> lower risk
    "income":        -1,   # higher income -> lower risk
    "uacr":          +1,
    "creatinine":    +1,
    "sleep_hours":   -1,   # weak; very short sleep is the risky tail
    "ldl":           +1,
    "race":           0,   # nominal code, no monotone direction -> not in index
    "sbp":           +1,
    "hdl":           -1,   # higher HDL is protective
    "uric_acid":     +1,
    "sedentary_min": +1,
    "dbp":           +1,
    "wbc":           +1,
    "hba1c":         +1,
    "waist":         +1,
    "bmi":           +1,
    "crp":           +1,
    "insulin":       +1,
    "bun":           +1,
    "glucose":       +1,
}

# Approximate population SD per feature, to make columns comparable before the
# importances weight them. Rough is fine: the Cox fit re-scales the index.
SCALE: dict[str, float] = {
    "htn_dx":        0.5,
    "diabetes_dx":   0.4,
    "highchol_dx":   0.5,
    "age":           15.0,
    "smoking":       0.8,
    "total_chol":    38.0,
    "non_hdl":       38.0,
    "education":     1.2,
    "income":        1.6,
    "uacr":          30.0,
    "creatinine":    0.3,
    "sleep_hours":   1.5,
    "ldl":           35.0,
    "race":          1.5,
    "sbp":           17.0,
    "hdl":           15.0,
    "uric_acid":     1.4,
    "sedentary_min": 180.0,
    "dbp":           11.0,
    "wbc":           2.0,
    "hba1c":         1.0,
    "waist":         14.0,
    "bmi":           6.0,
    "crp":           4.0,
    "insulin":       8.0,
    "bun":           6.0,
    "glucose":       30.0,
}


# --- Helpers --------------------------------------------------------------

def fill_defaults(person: dict) -> dict:
    """Return a copy of `person` with every missing feature set to its median.

    Lets the app accept a few inputs and still score all 27 features, exactly
    as the report intends ('sensible defaults for any input a user leaves
    blank'). Unknown keys are ignored.
    """
    out = {f: MEDIANS[f] for f in FEATURES}
    for k, v in person.items():
        if k in out and v is not None:
            out[k] = float(v)
    return out


def cvd_risk_index(person: dict) -> float:
    """Importance-weighted CVD risk index built from the column model report.

        index = sum_i  importance_i * direction_i * (x_i - median_i) / scale_i

    Higher = worse. Centered so a person sitting at every training median scores
    0. This single number is the column model's verdict on one person, and is
    what the survival model turns into a mortality estimate.
    """
    p = fill_defaults(person)
    total = 0.0
    for f in FEATURES:
        d = DIRECTION[f]
        if d == 0:
            continue
        total += IMPORTANCE[f] * d * (p[f] - MEDIANS[f]) / SCALE[f]
    return total
