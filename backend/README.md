# CVD Risk Prediction

Predicts prevalent cardiovascular disease (heart attack or stroke ever
diagnosed) and CVD mortality risk from patient risk factors, trained on
the NHANES dataset.

## Usage

```bash
uv run api.py
```

Set the Groq API key in `backend/.env` to enable AI lifestyle suggestions:

```
GROQ_API_KEY=your_key_here
```

---

## API endpoints

| Method | Path                      | Description                                         |
|--------|---------------------------|-----------------------------------------------------|
| GET    | `/`                       | Health check, model metadata, SHAP ranking          |
| GET    | `/features`               | Prevalent CVD model fields with units & importance  |
| GET    | `/mortality/features`     | Mortality model fields with importance & medians    |
| POST   | `/predict`                | Prevalent CVD probability                           |
| POST   | `/predict/explain`        | Prevalent CVD probability + per-feature SHAP values |
| POST   | `/mortality/predict`      | CVD death risk by horizon & survival curve          |
| POST   | `/predict/suggestions`    | Personalized lifestyle suggestions (AI)             |
| POST   | `/predict/chat`           | Interactive AI health coach (multi-turn)            |
| POST   | `/predict/comprehensive`  | All of the above in a single call                   |

All input fields are **optional** — missing values are median-imputed. More
fields → more accurate prediction.

---

## Unified input schema

Every `POST` endpoint (except `/predict/chat`, which wraps this in
`patient_data`) accepts the same JSON body.

```json
{
  "age":                    65,
  "female":                  0,
  "race":                    3,
  "education":               4,
  "income":                2.5,
  "sbp":                   148,
  "dbp":                    90,
  "pulse":                  72,
  "bmi":                  29.0,
  "waist":               101.0,
  "total_chol":            218,
  "hdl":                    42,
  "non_hdl":               176,
  "hba1c":                 7.1,
  "diabetes_dx":             1,
  "family_history":          1,
  "smoking":                 1,
  "sedentary_min":         300,
  "sleep_hours":           5.5,

  "htn_dx":                  1,
  "highchol_dx":             1,
  "uacr":                 10.0,
  "creatinine":            1.2,
  "ldl":                   130,
  "uric_acid":             5.5,
  "wbc":                   7.0,
  "crp":                   2.5,
  "insulin":              10.0,
  "bun":                  16.0,
  "glucose":             110.0,

  "exercise_days_per_week":  1,
  "diet_quality":        "poor",
  "alcohol_units_per_week": 12,
  "stress_level":        "high"
}
```

### Field reference

#### Core clinical fields

| Field            | Type    | Valid range     | Description                                                         |
|------------------|---------|-----------------|---------------------------------------------------------------------|
| `age`            | float   | 18 – 120        | Age in years                                                        |
| `female`         | 0 / 1   | —               | Sex: 1 = female, 0 = male                                           |
| `race`           | int     | —               | 1=Mexican Am., 2=Other Hispanic, 3=NH White, 4=NH Black, 6=NH Asian, 7=Other |
| `education`      | 1 – 5   | —               | 1 = < 9th grade … 5 = college graduate                             |
| `income`         | float   | ≥ 0             | Family income ÷ poverty line                                        |
| `sbp`            | float   | 50 – 300 mmHg   | Systolic blood pressure (mean of readings)                          |
| `dbp`            | float   | 20 – 200 mmHg   | Diastolic blood pressure (mean of readings)                         |
| `pulse`          | float   | 20 – 300 bpm    | Resting pulse                                                       |
| `bmi`            | float   | 10 – 80 kg/m²   | Body mass index                                                     |
| `waist`          | float   | 40 – 250 cm     | Waist circumference                                                 |
| `total_chol`     | float   | 50 – 600 mg/dL  | Total cholesterol                                                   |
| `hdl`            | float   | 10 – 200 mg/dL  | HDL cholesterol                                                     |
| `non_hdl`        | float   | 10 – 500 mg/dL  | Non-HDL cholesterol (auto-derived from `total_chol − hdl` if omitted) |
| `hba1c`          | float   | 2 – 20 %        | Glycohemoglobin (HbA1c)                                             |
| `diabetes_dx`    | 0 / 1   | —               | Doctor-diagnosed diabetes                                           |
| `family_history` | 0 / 1   | —               | Family history of CVD                                               |
| `smoking`        | 0/1/2   | —               | 0 = never, 1 = former, 2 = current smoker                          |
| `sedentary_min`  | float   | ≥ 0 min/day     | Sedentary activity time                                             |
| `sleep_hours`    | float   | 0 – 24 h/night  | Usual sleep duration                                                |

#### Mortality-specific fields

| Field         | Type  | Valid range    | Description                              |
|---------------|-------|----------------|------------------------------------------|
| `htn_dx`      | 0 / 1 | —              | Doctor-diagnosed hypertension            |
| `highchol_dx` | 0 / 1 | —              | Doctor-diagnosed high cholesterol        |
| `uacr`        | float | ≥ 0 mg/g       | Urine albumin-to-creatinine ratio        |
| `creatinine`  | float | ≥ 0 mg/dL      | Serum creatinine                         |
| `ldl`         | float | 10 – 500 mg/dL | LDL cholesterol                          |
| `uric_acid`   | float | ≥ 0 mg/dL      | Uric acid                                |
| `wbc`         | float | ≥ 0 10³/µL     | White blood cell count                   |
| `crp`         | float | ≥ 0 mg/L       | High-sensitivity CRP                     |
| `insulin`     | float | ≥ 0 µU/mL      | Fasting insulin                          |
| `bun`         | float | ≥ 0 mg/dL      | Blood urea nitrogen                      |
| `glucose`     | float | ≥ 0 mg/dL      | Fasting glucose                          |

#### Lifestyle / AI fields

| Field                    | Type   | Valid range  | Description                              |
|--------------------------|--------|--------------|------------------------------------------|
| `exercise_days_per_week` | float  | 0 – 7        | Exercise days per week                   |
| `diet_quality`           | string | —            | `"poor"` / `"average"` / `"good"`        |
| `alcohol_units_per_week` | float  | ≥ 0          | Alcohol units per week                   |
| `stress_level`           | string | —            | `"low"` / `"moderate"` / `"high"`        |

---

## Example requests

### Comprehensive prediction (recommended)

Runs all three models in a single call. AI suggestions degrade gracefully if
Groq is unavailable.

```bash
curl -s -X POST http://localhost:8000/predict/comprehensive \
  -H "Content-Type: application/json" \
  -d '{
    "age": 65,       "female": 0,          "race": 3,
    "education": 4,  "income": 2.5,        "sbp": 148,
    "dbp": 90,       "pulse": 72,          "bmi": 29.0,
    "waist": 101.0,  "total_chol": 218,    "hdl": 42,
    "non_hdl": 176,  "hba1c": 7.1,         "diabetes_dx": 1,
    "family_history": 1, "smoking": 1,     "sedentary_min": 300,
    "sleep_hours": 5.5,
    "htn_dx": 1,     "highchol_dx": 1,    "uacr": 10.0,
    "creatinine": 1.2, "ldl": 130,         "uric_acid": 5.5,
    "wbc": 7.0,      "crp": 2.5,           "insulin": 10.0,
    "bun": 16.0,     "glucose": 110.0,
    "exercise_days_per_week": 1, "diet_quality": "poor",
    "alcohol_units_per_week": 12, "stress_level": "high"
  }' | python -m json.tool
```

**Response:**

```json
{
    "prevalent_cvd": {
        "cvd_probability": 0.283,
        "risk_level": "high",
        "label": "Prevalent CVD (heart attack or stroke ever diagnosed)",
        "features_provided": ["age", "sbp", "dbp", "bmi", "..."],
        "features_imputed": []
    },
    "mortality": {
        "cvd_index": 0.4281,
        "horizon_cvd_death_risk": {
            "1y": 0.0029,
            "5y": 0.0187,
            "10y": 0.0418
        },
        "median_years_to_cvd_death": null,
        "survival_curve": {"0": 1.0, "1": 0.9971, "...": "..."},
        "risk_band": "low",
        "recommend_doctor_visit": false
    },
    "ai_suggestions": {
        "status": "success",
        "suggestions": "1. Reduce alcohol intake from 12 units per week...",
        "error_detail": null
    }
}
```

The `ai_suggestions.status` field is one of:
- `"success"` — suggestions generated successfully
- `"error"` — Groq request failed (detail in `error_detail`)
- `"disabled"` — `GROQ_API_KEY` not set

---

### Prevalent CVD prediction only

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 65, "female": 0, "sbp": 148, "dbp": 90, "bmi": 29.0,
       "total_chol": 218, "hdl": 42, "hba1c": 7.1, "diabetes_dx": 1,
       "smoking": 1, "family_history": 1}' | python -m json.tool
```

**Response:**

```json
{
    "cvd_probability": 0.283,
    "risk_level": "high",
    "label": "Prevalent CVD (heart attack or stroke ever diagnosed)",
    "features_provided": ["age", "sbp", "dbp", "bmi", "total_chol", "hdl", "hba1c", "diabetes_dx", "smoking"],
    "features_imputed": ["race", "education", "income", "waist", "ldl", "..."]
}
```

### Prevalent CVD with SHAP explanation

```bash
curl -s -X POST http://localhost:8000/predict/explain \
  -H "Content-Type: application/json" \
  -d '{"age": 65, "female": 0, "sbp": 148, "diabetes_dx": 1}' \
  | python -m json.tool
```

The `/predict/explain` response adds an `explanation` key containing:
- **`shap_values`** — each feature's additive contribution to CVD probability
- **`top_risk_drivers`** — features pushing the prediction up
- **`top_protective_factors`** — features pulling it down

### CVD mortality prediction only

```bash
curl -s -X POST http://localhost:8000/mortality/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 80, "sbp": 170, "htn_dx": 1, "diabetes_dx": 1, "creatinine": 1.8}' \
  | python -m json.tool
```

**Response:**
```json
{
    "cvd_index": 0.9819,
    "horizon_cvd_death_risk": {
        "1y": 0.0099,
        "5y": 0.0593,
        "10y": 0.1177
    },
    "median_years_to_cvd_death": null,
    "survival_curve": {
        "0": 1.0, "1": 0.9901, "5": 0.9407, "10": 0.8823, "15": 0.854
    },
    "risk_band": "moderate",
    "recommend_doctor_visit": false
}
```

- **`risk_band`** — `"low"` (< 5%), `"moderate"` (5–15%), `"elevated"` (≥ 15%) based on 10-year risk
- **`recommend_doctor_visit`** — `true` when 10-year risk ≥ 15%

### AI lifestyle suggestions

```bash
curl -s -X POST http://localhost:8000/predict/suggestions \
  -H "Content-Type: application/json" \
  -d '{"age": 65, "sbp": 148, "bmi": 29.0, "diabetes_dx": 1, "smoking": 1,
       "exercise_days_per_week": 1, "diet_quality": "poor",
       "alcohol_units_per_week": 12, "stress_level": "high", "sleep_hours": 5.5
      }' | python -m json.tool
```

**Response:**
```json
{
    "suggestions": "1. Reduce alcohol intake from 12 units per week...\n2. Increase exercise days..."
}
```

### AI health coach chat

The `patient_data` field accepts the same unified schema as all other endpoints.

```bash
curl -s -X POST http://localhost:8000/predict/chat \
  -H "Content-Type: application/json" \
  -d '{
    "patient_data": {
      "age": 65, "sbp": 148, "bmi": 29.0, "diabetes_dx": 1,
      "exercise_days_per_week": 1, "diet_quality": "poor",
      "alcohol_units_per_week": 12, "stress_level": "high", "sleep_hours": 5.5
    },
    "history": [
      {"role": "user",      "content": "What is my biggest risk factor?"},
      {"role": "assistant", "content": "Your high blood pressure at 148 mmHg..."}
    ],
    "message": "What should I do first?"
  }' | python -m json.tool
```

**Response:**
```json
{
    "reply": "Given your profile, the most impactful first step would be..."
}
```

Pass previous turns in `history` to maintain conversation context.
