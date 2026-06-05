import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from groq import Groq
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in environment or .env file")
    groq_client = Groq(api_key=GROQ_API_KEY)
except (ImportError, ValueError) as e:
    groq_client = None
    print(f"  Warning: Groq client could not be initialized: {e}")

BUNDLE_PATH = os.environ.get("CVD_MODEL_PATH", "../model/cvd_model_bundle.joblib")

try:
    bundle = joblib.load(BUNDLE_PATH)
    MODEL = bundle["model"]
    LR_MODEL = bundle.get("lr_model")
    FEATURES = bundle["features"]
    FEAT_META = bundle.get("feature_meta", {})
    SHAP_IMP = bundle.get("shap_importance", {})
    LABEL_DESC = bundle.get("label", "Prevalent CVD")
    DATA_SOURCE = bundle.get("data_source", "NHANES")
    MODEL_TYPE = bundle.get("model_type", "XGBoost")
    print(f"  Loaded model bundle from {BUNDLE_PATH}")
    print(f"  Model type : {MODEL_TYPE}")
    print(f"  Features   : {FEATURES}")
except FileNotFoundError:
    raise RuntimeError(f"Model bundle not found at {BUNDLE_PATH}.")

class CompetingRisksModel:
    def __init__(self, data: dict):
        self.cph_cvd = data["cph_cvd"]
        self.cph_other = data["cph_other"]
        self.features = data["features"]
        self._grid = data["_grid"]
        self._H1_0 = data["_H1_0"]
        self._H2_0 = data["_H2_0"]
        self._dH1_0 = data["_dH1_0"]

    def _cif_curve(self, person: dict) -> np.ndarray:
        x = pd.DataFrame([_mort_fill_defaults(person)])[self.features]
        ph1 = float(self.cph_cvd.predict_partial_hazard(x).iloc[0])
        ph2 = float(self.cph_other.predict_partial_hazard(x).iloc[0])
        S = np.exp(-(self._H1_0 * ph1 + self._H2_0 * ph2))
        S_prev = np.concatenate([[1.0], S[:-1]])
        return np.cumsum(S_prev * self._dH1_0 * ph1)   # CIF at each grid time

    def _at(self, cif: np.ndarray, times) -> pd.Series:
        idx = np.searchsorted(self._grid, np.asarray(times, float), side="right") - 1
        vals = np.where(idx >= 0, cif[idx.clip(min=0)], 0.0)
        return pd.Series(vals, index=list(times))

    def risk_by_horizon(self, person: dict, horizons=(1, 5, 10)) -> dict:
        cif = self._cif_curve(person)
        s = self._at(cif, list(horizons))
        return {h: float(s.loc[h]) for h in horizons}

    def survival_function(self, person: dict, times=None) -> pd.Series:
        cif = self._cif_curve(person)
        if times is None:
            times = self._grid
        return 1.0 - self._at(cif, list(times))

    def median_survival_years(self, person: dict):
        cif = self._cif_curve(person)
        crossed = np.where(cif >= 0.5)[0]
        return None if len(crossed) == 0 else float(self._grid[crossed[0]])

MORTALITY_MODEL_PATH = os.environ.get("CVD_MORTALITY_MODEL_PATH", "../model-death/cvd_death_model.joblib")

try:
    mort_data = joblib.load(MORTALITY_MODEL_PATH)
    MORTALITY_MODEL = CompetingRisksModel(mort_data)
    print(f"  Loaded mortality model from {MORTALITY_MODEL_PATH}")
except FileNotFoundError:
    raise RuntimeError(f"Mortality model not found at {MORTALITY_MODEL_PATH}.")


MORT_IMPORTANCE: dict[str, float] = {
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

MORT_MEDIANS: dict[str, float] = {
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

MORT_DESCRIPTION: dict[str, str] = {
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

MORT_FEATURES: list[str] = list(MORT_IMPORTANCE.keys())

# Sign of each feature's effect: +1 higher = higher risk, -1 protective, 0 nominal.
MORT_DIRECTION: dict[str, int] = {
    "htn_dx":        +1,
    "diabetes_dx":   +1,
    "highchol_dx":   +1,
    "age":           +1,
    "smoking":       +1,
    "total_chol":    +1,
    "non_hdl":       +1,
    "education":     -1,
    "income":        -1,
    "uacr":          +1,
    "creatinine":    +1,
    "sleep_hours":   -1,
    "ldl":           +1,
    "race":           0,
    "sbp":           +1,
    "hdl":           -1,
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

# Approximate population SD per feature (used only to normalise before weighting).
MORT_SCALE: dict[str, float] = {
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

def _mort_fill_defaults(person: dict) -> dict:
    out = {f: MORT_MEDIANS[f] for f in MORT_FEATURES}
    for k, v in person.items():
        if k in out and v is not None:
            out[k] = float(v)
    return out

def _cvd_risk_index(person: dict) -> float:
    """Importance-weighted CVD risk index: sum_i importance_i * direction_i * (x_i - median_i) / scale_i."""
    p = _mort_fill_defaults(person)
    total = 0.0
    for f in MORT_FEATURES:
        d = MORT_DIRECTION[f]
        if d == 0:
            continue
        total += MORT_IMPORTANCE[f] * d * (p[f] - MORT_MEDIANS[f]) / MORT_SCALE[f]
    return total

def _mort_risk_band(risk_10y: float) -> str:
    if risk_10y < 0.05:
        return "low"
    if risk_10y < 0.15:
        return "moderate"
    return "elevated"

def _predict_mortality(person: dict, horizons=(1, 5, 10), curve_to_years: int = 15) -> dict:
    filled = _mort_fill_defaults(person)
    risk = MORTALITY_MODEL.risk_by_horizon(filled, horizons=horizons)
    median = MORTALITY_MODEL.median_survival_years(filled)
    curve = MORTALITY_MODEL.survival_function(filled, times=list(range(curve_to_years + 1)))

    risk_10 = risk.get(10, max(risk.values()))
    return {
        "cvd_index":                 round(_cvd_risk_index(person), 4),
        "horizon_cvd_death_risk":    {f"{h}y": round(p, 4) for h, p in risk.items()},
        "median_years_to_cvd_death": round(median, 1) if median is not None else None,
        "survival_curve":            {int(t): round(float(s), 4) for t, s in curve.items()},
        "risk_band":                 _mort_risk_band(risk_10),
        "recommend_doctor_visit":    risk_10 >= 0.15,
    }

app = FastAPI(
    title="CVD Risk Prediction API",
    description=(
        "Predicts prevalent cardiovascular disease (heart attack or stroke ever"
        "diagnosed) from patient risk factors, trained on the NHANES dataset."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientData(BaseModel):
    # All fields are optional - missing values are imputed by the pipeline.
    age:            Optional[float] = Field(None, ge=18, le=120, description="Age in years")
    female:         Optional[float] = Field(None, ge=0, le=1,   description="Sex: 1=female, 0=male")
    race:           Optional[float] = Field(None,                description="Race/ethnicity code (1=Mexican American, 2=Other Hispanic, 3=NH White, 4=NH Black, 6=NH Asian, 7=Other)")
    education:      Optional[float] = Field(None, ge=1, le=5,   description="Education level 1-5 (1=<9th grade ... 5=college grad)")
    income:         Optional[float] = Field(None, ge=0,         description="Income-to-poverty ratio")
    sbp:            Optional[float] = Field(None, ge=50, le=300, description="Systolic blood pressure (mmHg)")
    dbp:            Optional[float] = Field(None, ge=20, le=200, description="Diastolic blood pressure (mmHg)")
    pulse:          Optional[float] = Field(None, ge=20, le=300, description="Resting pulse (bpm)")
    bmi:            Optional[float] = Field(None, ge=10, le=80,  description="Body mass index (kg/m2)")
    waist:          Optional[float] = Field(None, ge=40, le=250, description="Waist circumference (cm)")
    total_chol:     Optional[float] = Field(None, ge=50, le=600, description="Total cholesterol (mg/dL)")
    hdl:            Optional[float] = Field(None, ge=10, le=200, description="HDL cholesterol (mg/dL)")
    non_hdl:        Optional[float] = Field(None, ge=10, le=500, description="Non-HDL cholesterol (mg/dL). If omitted and total_chol+hdl are given, it is derived automatically.")
    hba1c:          Optional[float] = Field(None, ge=2,  le=20,  description="Glycohemoglobin HbA1c (%)")
    diabetes_dx:    Optional[float] = Field(None, ge=0, le=1,   description="Diabetes diagnosed: 1=yes, 0=no")
    family_history: Optional[float] = Field(None, ge=0, le=1,   description="Family history of CVD: 1=yes, 0=no")
    smoking:        Optional[float] = Field(None, ge=0, le=2,   description="Smoking status: 0=never, 1=former, 2=current")
    sedentary_min:  Optional[float] = Field(None, ge=0,         description="Sedentary time (minutes/day)")
    sleep_hours:    Optional[float] = Field(None, ge=0, le=24,  description="Usual sleep duration (hours/night)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 65,
                "female": 0,
                "sbp": 148,
                "dbp": 90,
                "bmi": 29.0,
                "total_chol": 218,
                "hdl": 42,
                "hba1c": 7.1,
                "diabetes_dx": 1,
                "smoking": 1,
                "family_history": 1
            }
        }
    }

class MortalityPatientData(BaseModel):
    # All 27 features are optional; missing values fall back to training medians.
    htn_dx:        Optional[float] = Field(None, ge=0, le=1,   description="Diagnosed high blood pressure (0/1)")
    diabetes_dx:   Optional[float] = Field(None, ge=0, le=1,   description="Diagnosed diabetes (0/1)")
    highchol_dx:   Optional[float] = Field(None, ge=0, le=1,   description="Diagnosed high cholesterol (0/1)")
    age:           Optional[float] = Field(None, ge=18, le=120, description="Age in years")
    smoking:       Optional[float] = Field(None, ge=0, le=2,   description="Smoking status: 0=never, 1=former, 2=current")
    total_chol:    Optional[float] = Field(None, ge=50, le=600, description="Total cholesterol (mg/dL)")
    non_hdl:       Optional[float] = Field(None, ge=10, le=500, description="Non-HDL cholesterol (mg/dL)")
    education:     Optional[float] = Field(None, ge=1, le=5,   description="Education level 1-5")
    income:        Optional[float] = Field(None, ge=0,          description="Income-to-poverty ratio")
    uacr:          Optional[float] = Field(None, ge=0,          description="Urine albumin-to-creatinine ratio")
    creatinine:    Optional[float] = Field(None, ge=0,          description="Serum creatinine (mg/dL)")
    sleep_hours:   Optional[float] = Field(None, ge=0, le=24,  description="Usual sleep hours")
    ldl:           Optional[float] = Field(None, ge=10, le=500, description="LDL cholesterol (mg/dL)")
    race:          Optional[float] = Field(None,                description="Race/ethnicity code (1=Mexican American, 2=Other Hispanic, 3=NH White, 4=NH Black, 6=NH Asian, 7=Other)")
    sbp:           Optional[float] = Field(None, ge=50, le=300, description="Systolic blood pressure (mmHg)")
    hdl:           Optional[float] = Field(None, ge=10, le=200, description="HDL cholesterol (mg/dL)")
    uric_acid:     Optional[float] = Field(None, ge=0,          description="Uric acid (mg/dL)")
    sedentary_min: Optional[float] = Field(None, ge=0,          description="Sedentary minutes/day")
    dbp:           Optional[float] = Field(None, ge=20, le=200, description="Diastolic blood pressure (mmHg)")
    wbc:           Optional[float] = Field(None, ge=0,          description="White blood cell count (10^3/uL)")
    hba1c:         Optional[float] = Field(None, ge=2, le=20,  description="HbA1c (%)")
    waist:         Optional[float] = Field(None, ge=40, le=250, description="Waist circumference (cm)")
    bmi:           Optional[float] = Field(None, ge=10, le=80,  description="Body mass index (kg/m2)")
    crp:           Optional[float] = Field(None, ge=0,          description="High-sensitivity CRP (mg/L)")
    insulin:       Optional[float] = Field(None, ge=0,          description="Insulin (uU/mL)")
    bun:           Optional[float] = Field(None, ge=0,          description="Blood urea nitrogen (mg/dL)")
    glucose:       Optional[float] = Field(None, ge=0,          description="Fasting glucose (mg/dL)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 80,
                "sbp": 170,
                "htn_dx": 1,
                "diabetes_dx": 1,
                "creatinine": 1.8
            }
        }
    }

def patient_to_row(patient: PatientData) -> pd.DataFrame:
    d = patient.model_dump()
    # Derive non_hdl if possible and not provided
    if d.get("non_hdl") is None:
        tc  = d.get("total_chol")
        hdl = d.get("hdl")
        if tc is not None and hdl is not None:
            d["non_hdl"] = tc - hdl
    # Replace None with NaN, keep only model features
    row = {}
    for f in FEATURES:
        val = d.get(f)
        row[f] = float(val) if val is not None else np.nan
    return pd.DataFrame([row])

def _is_missing(val) -> bool:
    """Return True if the value is NaN (missing)."""
    try:
        return np.isnan(float(val))
    except (TypeError, ValueError):
        return True

@app.get("/", summary="Health check & model metadata")
def root():
    return {
        "status": "ok",
        "model_type":   MODEL_TYPE,
        "label":        LABEL_DESC,
        "data_source":  DATA_SOURCE,
        "n_features":   len(FEATURES),
        "features":     FEATURES,
        "shap_importance_ranked": (
            sorted(SHAP_IMP.items(), key=lambda x: -x[1])
            if SHAP_IMP else None
        )
    }

@app.get("/features", summary="List accepted input features with metadata")
def list_features():
    out = {}
    for feat in FEATURES:
        meta = FEAT_META.get(feat, {})
        out[feat] = {
            "label":           meta.get("label", feat),
            "unit":            meta.get("unit", ""),
            "shap_importance": round(SHAP_IMP.get(feat, 0), 4) if SHAP_IMP else None,
            "required":        False,   # all features are optional; missing -> imputed
        }
    return out

@app.post("/predict", summary="Predict CVD risk probability")
def predict(patient: PatientData):
    X = patient_to_row(patient)
    try:
        proba = float(MODEL.predict_proba(X)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    provided = [f for f in FEATURES if not _is_missing(X[f].values[0])]
    imputed  = [f for f in FEATURES if _is_missing(X[f].values[0])]

    risk_label = (
        "high"   if proba >= 0.40 else
        "medium" if proba >= 0.20 else
        "low"
    )

    return {
        "cvd_probability": round(proba, 4),
        "risk_level":      risk_label,
        "label":           LABEL_DESC,
        "features_provided": provided,
        "features_imputed":  imputed
    }

@app.post("/predict/explain", summary="Predict CVD risk with SHAP feature attributions")
def predict_explain(patient: PatientData):
    X = patient_to_row(patient)
    try:
        proba = float(MODEL.predict_proba(X)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # SHAP attributions
    shap_values = None
    shap_error  = None
    try:
        import shap
        X_imp = MODEL.named_steps["impute"].transform(X)
        explainer = shap.TreeExplainer(MODEL.named_steps["clf"])
        raw = explainer.shap_values(X_imp)
        if isinstance(raw, list):
            raw = raw[1]
        sv = raw[0]  # shape: (n_features,)
        shap_values = {
            feat: round(float(sv[i]), 5)
            for i, feat in enumerate(FEATURES)
        }
        # Sort by absolute magnitude for readability
        shap_values = dict(
            sorted(shap_values.items(), key=lambda kv: -abs(kv[1]))
        )
    except Exception as e:
        shap_error = str(e)

    provided = [f for f in FEATURES if not _is_missing(X[f].values[0])]
    imputed  = [f for f in FEATURES if _is_missing(X[f].values[0])]

    risk_label = (
        "high"   if proba >= 0.40 else
        "medium" if proba >= 0.20 else
        "low"
    )

    response = {
        "cvd_probability":   round(proba, 4),
        "risk_level":        risk_label,
        "label":             LABEL_DESC,
        "features_provided": provided,
        "features_imputed":  imputed,
    }

    if shap_values is not None:
        top_risk    = [(k, v) for k, v in shap_values.items() if v > 0][:5]
        top_protect = [(k, v) for k, v in shap_values.items() if v < 0][:3]

        response["explanation"] = {
            "shap_values":         shap_values,
            "top_risk_drivers":    [
                {
                    "feature": k,
                    "label":   FEAT_META.get(k, {}).get("label", k),
                    "shap":    round(v, 5),
                    "direction": "increases risk",
                }
                for k, v in top_risk
            ],
            "top_protective_factors": [
                {
                    "feature": k,
                    "label":   FEAT_META.get(k, {}).get("label", k),
                    "shap":    round(v, 5),
                    "direction": "decreases risk",
                }
                for k, v in top_protect
            ],
            "note": (
                "SHAP values show each feature's additive contribution to the "
                "log-odds of cardiovascular disease. Positive = pushes prediction higher; "
                "negative = pushes it lower."
            ),
        }
    else:
        response["explanation"] = {"error": shap_error}

    return response

@app.get("/mortality/features", summary="List accepted input features for the CVD mortality model")
def list_mortality_features():
    return {
        feat: {
            "description":  MORT_DESCRIPTION[feat],
            "importance":   round(MORT_IMPORTANCE[feat], 4),
            "median":       MORT_MEDIANS[feat],
            "required":     False,  # all features are optional; missing -> population median
        }
        for feat in MORT_FEATURES
    }

@app.post("/mortality/predict", summary="Predict CVD death risk by horizon (1 / 5 / 10 years)")
def mortality_predict(patient: MortalityPatientData):
    person = {k: v for k, v in patient.model_dump().items() if v is not None}
    try:
        return _predict_mortality(person)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

class LifestyleAdviceRequest(PatientData):
    exercise_days_per_week: Optional[float] = Field(0, ge=0, le=7, description="Number of exercise days per week")
    diet_quality:           Optional[str]   = Field("average", description="Diet quality: poor/average/good")
    alcohol_units_per_week: Optional[float] = Field(0, ge=0,       description="Alcohol units per week")
    stress_level:           Optional[str]   = Field("moderate", description="Stress level: low/moderate/high")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    patient_data: LifestyleAdviceRequest
    history: List[ChatMessage]
    message: str

def _build_ai_user_profile(patient: LifestyleAdviceRequest) -> dict:
    sex_str = "female" if patient.female == 1.0 else "male" if patient.female == 0.0 else "not specified"
    smoking_map = {0: "never", 1: "former", 2: "current"}
    smoking_str = smoking_map.get(patient.smoking, "not specified")
    diabetes_str = "yes" if patient.diabetes_dx == 1.0 else "no" if patient.diabetes_dx == 0.0 else "not specified"
    family_hx_str = "yes" if patient.family_history == 1.0 else "no" if patient.family_history == 0.0 else "not specified"

    return {
        "age": patient.age,
        "sex": sex_str,
        "bmi": patient.bmi,
        "systolic_bp": patient.sbp,
        "diastolic_bp": patient.dbp,
        "cholesterol_total": patient.total_chol,
        "hdl_cholesterol": patient.hdl,
        "smoker": smoking_str,
        "diabetic": diabetes_str,
        "family_history": family_hx_str,
        "exercise_days_per_week": patient.exercise_days_per_week,
        "diet_quality": patient.diet_quality,
        "alcohol_units_per_week": patient.alcohol_units_per_week,
        "stress_level": patient.stress_level,
        "sleep_hours": patient.sleep_hours,
        "sedentary_minutes_per_day": patient.sedentary_min,
    }

def _get_risk_predictions_context(patient: LifestyleAdviceRequest) -> tuple[dict, str]:
    X = patient_to_row(patient)
    proba = float(MODEL.predict_proba(X)[0, 1])
    prevalent_risk_lvl = "high" if proba >= 0.20 else "moderate" if proba >= 0.08 else "low"
    
    mort_person = {}
    patient_dict = patient.model_dump()
    for f in MORT_FEATURES:
        if f in patient_dict and patient_dict[f] is not None:
            mort_person[f] = patient_dict[f]
            
    mortality_pred = _predict_mortality(mort_person)
    
    context = {
        "prevalent_cvd_risk": f"{proba:.1%}",
        "prevalent_cvd_risk_category": prevalent_risk_lvl,
        "mortality_10y_risk": f"{mortality_pred['horizon_cvd_death_risk']['10y']:.1%}",
        "mortality_risk_category": mortality_pred['risk_band'],
        "median_years_to_cvd_death": mortality_pred['median_years_to_cvd_death'] if mortality_pred['median_years_to_cvd_death'] is not None else "undefined"
    }
    
    context_str = f"""- Prevalent cardiovascular disease risk (chance of existing/past CVD): {context['prevalent_cvd_risk']} (Category: {context['prevalent_cvd_risk_category']})
- 10-year CVD death (mortality) risk: {context['mortality_10y_risk']} (Category: {context['mortality_risk_category']})
- Median years to CVD death: {context['median_years_to_cvd_death']}"""

    return context, context_str

@app.post("/predict/suggestions", summary="Get personalized lifestyle suggestions based on patient data and predictions")
def predict_suggestions(patient: LifestyleAdviceRequest):
    if not groq_client:
        raise HTTPException(status_code=503, detail="Groq AI client is not available. Please verify the environment configuration.")

    try:
        user_profile = _build_ai_user_profile(patient)
        _, predictions_context_str = _get_risk_predictions_context(patient)
        
        prompt = f"""
You are a preventive-health advisor helping someone understand how to reduce their cardiovascular risk.

## Patient profile
{json.dumps(user_profile, indent=2)}

## Risk assessment
{predictions_context_str}

## Your task
Based ONLY on the specific data points above, suggest 3-5 concrete, prioritised lifestyle changes
that would have the greatest measurable impact on reducing this person's cardiovascular risk.

Rules:
- Be specific to their numbers (e.g. if BMI is 31 mention weight; if smoker, mention cessation; if sedentary time is high, mention physical activity).
- Skip factors that are already healthy for this person - don't suggest things they already do well.
- For each suggestion include: what to change, why it matters for THEM, and one simple first step.
- Keep the total response under 300 words.
- Do NOT diagnose, prescribe medication, or replace a doctor's advice.
- End with a one-sentence reminder to consult a healthcare professional.

Format: numbered list, plain text, no markdown headers.
"""
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        suggestions = response.choices[0].message.content
        return {
            "suggestions": suggestions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {e}")

@app.post("/predict/chat", summary="Chat with an AI health coach about risk predictions and lifestyle tips")
def predict_chat(request: ChatRequest):
    if not groq_client:
        raise HTTPException(status_code=503, detail="Groq AI client is not available. Please verify the environment configuration.")

    try:
        user_profile = _build_ai_user_profile(request.patient_data)
        _, predictions_context_str = _get_risk_predictions_context(request.patient_data)
        
        system_prompt = f"""
You are a friendly and knowledgeable preventive-health chatbot helping a patient understand their cardiovascular health results and make positive lifestyle changes.

Here are the details of the patient you are chatting with:

## Patient profile
{json.dumps(user_profile, indent=2)}

## Risk assessment
{predictions_context_str}

Instructions:
1. Answer the patient's questions about their results, risk levels, and lifestyle recommendations.
2. Be encouraging, empathetic, and clear. Avoid overly dense medical jargon.
3. Be specific to their numbers. If they ask about physical activity, check their sedentary minutes and exercise days. If they ask about weight, check their BMI.
4. Do NOT diagnose, prescribe medication, or give specific clinical treatments. Recommend they consult their doctor for clinical decisions.
5. Keep answers concise (under 200 words per response).
6. End with a subtle reminder if they ask about changing medication/diagnoses.
"""
        messages = [{"role": "system", "content": system_prompt}]
        for msg in request.history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": request.message})
        
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=512,
        )
        reply = response.choices[0].message.content
        return {
            "reply": reply
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
