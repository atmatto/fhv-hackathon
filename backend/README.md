# CVD Risk Prediction

Predicts prevalent cardiovascular disease (heart attack or stroke ever
diagnosed) from patient risk factors, trained on the NHANES dataset.

## Usage

```bash
uv run api.py
```

## API endpoints

| Method | Path              | Description                                  |
|--------|-------------------|----------------------------------------------|
| GET    | `/`               | Health check, model metadata, SHAP ranking   |
| GET    | `/features`       | All accepted fields with units & importance  |
| POST   | `/predict`        | CVD probability for a patient                |
| POST   | `/predict/explain`| Probability + per-feature SHAP attributions  |

All input fields are **optional** — missing values are median-imputed by the
pipeline. More fields → more accurate prediction.

---

## Example requests

### Basic prediction

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }' | python -m json.tool
```

**Response:**
```json
{
  "cvd_probability": 0.414,
  "risk_level": "high",
  "label": "Prevalent CVD (heart attack or stroke ever diagnosed)",
  "features_provided": ["age", "female", "sbp", "dbp", "bmi", ...],
  "features_imputed":  ["race", "education", "income", ...]
}
```

### Prediction with explanation

```bash
curl -s -X POST http://localhost:8000/predict/explain \
  -H "Content-Type: application/json" \
  -d '{"age": 65, "female": 0, "sbp": 148, "diabetes_dx": 1}' \
  | python -m json.tool
```

The `/predict/explain` response includes a full SHAP breakdown:
- **`shap_values`** — each feature's additive contribution to CVD probability
- **`top_risk_drivers`** — features pushing the prediction up
- **`top_protective_factors`** — features pulling it down

---

## Input fields

| Field            | Unit         | Description                                    |
|------------------|--------------|------------------------------------------------|
| `age`            | years        | Age at screening                               |
| `female`         | 0 / 1        | Sex (1 = female, 0 = male)                     |
| `race`           | code         | 1=Mexican Am., 2=Other Hispanic, 3=NH White,   |
|                  |              | 4=NH Black, 6=NH Asian, 7=Other                |
| `education`      | 1–5          | 1=<9th grade … 5=college graduate              |
| `income`         | ratio        | Family income ÷ poverty line                   |
| `sbp`            | mmHg         | Systolic BP (mean of readings)                 |
| `dbp`            | mmHg         | Diastolic BP (mean of readings)                |
| `pulse`          | bpm          | Resting pulse                                  |
| `bmi`            | kg/m²        | Body mass index                                |
| `waist`          | cm           | Waist circumference                            |
| `total_chol`     | mg/dL        | Total cholesterol                              |
| `hdl`            | mg/dL        | HDL cholesterol                                |
| `non_hdl`        | mg/dL        | Non-HDL cholesterol (auto-derived if omitted)  |
| `hba1c`          | %            | Glycohemoglobin (HbA1c)                        |
| `diabetes_dx`    | 0 / 1        | Doctor-diagnosed diabetes                      |
| `family_history` | 0 / 1        | Family history of CVD                          |
| `smoking`        | 0 / 1 / 2    | Never / former / current smoker                |
| `sedentary_min`  | min/day      | Sedentary activity time                        |
| `sleep_hours`    | hours/night  | Usual sleep duration                           |

1. **Percentile context** — "Your SBP of 148 mmHg is in the 82nd percentile
   for this dataset" (store training distribution in the bundle).
2. **Counterfactual** — "If you reduced SBP by 20 mmHg, predicted risk
   would drop from 41% to ~28%" (run the model on a perturbed input).
3. **Risk factor narrative** — pass SHAP values + patient data to an LLM
   for a patient-friendly summary.
4. **Calibration** — post-hoc Platt scaling to make probabilities better
   calibrated for clinical interpretation.
