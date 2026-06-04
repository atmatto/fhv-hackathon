import os
import numpy as np
import pandas as pd
import joblib
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
                "log-odds of CVD. Positive = pushes prediction higher; "
                "negative = pushes it lower."
            ),
        }
    else:
        response["explanation"] = {"error": shap_error}

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
