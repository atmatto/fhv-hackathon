<template>
  <form @submit.prevent="$emit('submit')">

    <h2 class="title">Health Risk Input Form</h2>

    <div class="grid">

      <div class="field">
        <label>Age</label>
        <input :value="form.age" @input="update('age', +$event.target.value)"
          type="number" min="18" max="120" required />
      </div>

      <div class="field">
        <label>Female</label>
        <label class="toggle">
          <input type="checkbox" :checked="form.female"
            @change="update('female', $event.target.checked)" />
          <span>{{ form.female ? "Yes" : "No" }}</span>
        </label>
      </div>

      <div class="field">
        <label>Systolic BP</label>
        <input :value="form.sbp" @input="update('sbp', +$event.target.value)"
          type="number" min="50" required />
      </div>

      <div class="field">
        <label>Diastolic BP</label>
        <input :value="form.dbp" @input="update('dbp', +$event.target.value)"
          type="number" min="20" required />
      </div>

      <div class="field">
        <label>BMI</label>
        <input :value="form.bmi" @input="update('bmi', +$event.target.value)"
          type="number" step="0.1" min="10" required />
      </div>

      <div class="field">
        <label>Total Cholesterol</label>
        <input :value="form.total_chol" @input="update('total_chol', +$event.target.value)"
          type="number" min="50" required />
      </div>

      <div class="field">
        <label>HDL</label>
        <input :value="form.hdl" @input="update('hdl', +$event.target.value)"
          type="number" min="10" required />
      </div>

      <div class="field">
        <label>HbA1c</label>
        <input :value="form.hba1c" @input="update('hba1c', +$event.target.value)"
          type="number" step="0.1" min="2" required />
      </div>

      <div class="field">
        <label>Diabetes Diagnosis</label>
        <label class="toggle">
          <input type="checkbox" :checked="form.diabetes_dx"
            @change="update('diabetes_dx', $event.target.checked)" />
          <span>{{ form.diabetes_dx ? "Yes" : "No" }}</span>
        </label>
      </div>

      <div class="field">
        <label>Smoking Status</label>
        <select :value="form.smoking" @change="update('smoking', +$event.target.value)">
          <option value="0">Never</option>
          <option value="1">Former</option>
          <option value="2">Current</option>
        </select>
      </div>

      <div class="field">
        <label>Family History</label>
        <label class="toggle">
          <input type="checkbox" :checked="form.family_history"
            @change="update('family_history', $event.target.checked)" />
          <span>{{ form.family_history ? "Yes" : "No" }}</span>
        </label>
      </div>

    </div>

    <!-- Advanced Inputs foldout -->
    <div class="advanced-section">
      <button type="button" class="advanced-toggle" @click="showAdvanced = !showAdvanced" :aria-expanded="showAdvanced">
        <span class="advanced-toggle-label">
          <svg class="adv-icon" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path fill-rule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clip-rule="evenodd" />
          </svg>
          Advanced Clinical & Demographics
        </span>
        <svg class="chevron" :class="{ open: showAdvanced }" viewBox="0 0 20 20" fill="currentColor">
          <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
        </svg>
      </button>

      <div class="advanced-body" :class="{ expanded: showAdvanced }">
        <p class="advanced-hint">
          These fields are optional. If left blank, they will be imputed by the model.
        </p>
        <div class="grid">
          <div class="field">
            <label>Resting Pulse <span class="unit">(bpm)</span></label>
            <input :value="form.pulse" @input="update('pulse', $event.target.value === '' ? null : +$event.target.value)" type="number" min="20" max="300" />
          </div>
          <div class="field">
            <label>Waist Circumference <span class="unit">(cm)</span></label>
            <input :value="form.waist" @input="update('waist', $event.target.value === '' ? null : +$event.target.value)" type="number" min="40" max="250" step="0.1" />
          </div>
          <div class="field">
            <label>Non-HDL Cholesterol <span class="unit">(mg/dL)</span></label>
            <input :value="form.non_hdl" @input="update('non_hdl', $event.target.value === '' ? null : +$event.target.value)" type="number" min="10" max="500" placeholder="Auto-derived if blank" />
          </div>
          <div class="field">
            <label>Sedentary Time <span class="unit">(min/day)</span></label>
            <input :value="form.sedentary_min" @input="update('sedentary_min', $event.target.value === '' ? null : +$event.target.value)" type="number" min="0" max="1440" />
          </div>
          <div class="field">
            <label>Sleep Duration <span class="unit">(hrs/night)</span></label>
            <input :value="form.sleep_hours" @input="update('sleep_hours', $event.target.value === '' ? null : +$event.target.value)" type="number" min="0" max="24" step="0.5" />
          </div>
          <div class="field">
            <label>Education Level <span class="unit">(1–5)</span></label>
            <select :value="form.education || ''" @change="update('education', $event.target.value || null)">
              <option value="">— Not specified —</option>
              <option value="1">1 – Less than 9th grade</option>
              <option value="2">2 – 9th–11th grade</option>
              <option value="3">3 – High school / GED</option>
              <option value="4">4 – Some college / AA</option>
              <option value="5">5 – College graduate</option>
            </select>
          </div>
          <div class="field">
            <label>Income-to-Poverty Ratio</label>
            <input :value="form.income" @input="update('income', $event.target.value === '' ? null : +$event.target.value)" type="number" min="0" step="0.1" placeholder="e.g. 2.5" />
          </div>
          <div class="field">
            <label>Race / Ethnicity</label>
            <select :value="form.race || ''" @change="update('race', $event.target.value || null)">
              <option value="">— Not specified —</option>
              <option value="1">Mexican American</option>
              <option value="2">Other Hispanic</option>
              <option value="3">Non-Hispanic White</option>
              <option value="4">Non-Hispanic Black</option>
              <option value="6">Non-Hispanic Asian</option>
              <option value="7">Other / Multiracial</option>
            </select>
          </div>
        </div>
      </div>
    </div>
    <!-- end Advanced Inputs -->

    <!-- Mortality & Lab Values foldout -->
    <div class="advanced-section">
      <button type="button" class="advanced-toggle" @click="showLabs = !showLabs" :aria-expanded="showLabs">
        <span class="advanced-toggle-label">
          <svg class="adv-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
          </svg>
          Mortality & Lab Values
        </span>
        <svg class="chevron" :class="{ open: showLabs }" viewBox="0 0 20 20" fill="currentColor">
          <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
        </svg>
      </button>

      <div class="advanced-body" :class="{ expanded: showLabs }">
        <p class="advanced-hint">
          Additional lab markers and conditions to improve mortality prediction horizon.
        </p>
        <div class="grid">
          <div class="field">
            <label>Hypertension Diagnosis</label>
            <label class="toggle">
              <input type="checkbox" :checked="form.htn_dx" @change="update('htn_dx', $event.target.checked)" />
              <span>{{ form.htn_dx ? "Yes" : "No" }}</span>
            </label>
          </div>
          <div class="field">
            <label>High Chol. Diagnosis</label>
            <label class="toggle">
              <input type="checkbox" :checked="form.highchol_dx" @change="update('highchol_dx', $event.target.checked)" />
              <span>{{ form.highchol_dx ? "Yes" : "No" }}</span>
            </label>
          </div>
          <div class="field">
            <label>UACR <span class="unit">(mg/g)</span></label>
            <input :value="form.uacr" @input="update('uacr', $event.target.value === '' ? null : +$event.target.value)" type="number" min="0" step="0.1" />
          </div>
          <div class="field">
            <label>Creatinine <span class="unit">(mg/dL)</span></label>
            <input :value="form.creatinine" @input="update('creatinine', $event.target.value === '' ? null : +$event.target.value)" type="number" min="0" step="0.1" />
          </div>
          <div class="field">
            <label>LDL Cholesterol <span class="unit">(mg/dL)</span></label>
            <input :value="form.ldl" @input="update('ldl', $event.target.value === '' ? null : +$event.target.value)" type="number" min="10" max="500" />
          </div>
          <div class="field">
            <label>Uric Acid <span class="unit">(mg/dL)</span></label>
            <input :value="form.uric_acid" @input="update('uric_acid', $event.target.value === '' ? null : +$event.target.value)" type="number" min="0" step="0.1" />
          </div>
          <div class="field">
            <label>WBC <span class="unit">(10³/µL)</span></label>
            <input :value="form.wbc" @input="update('wbc', $event.target.value === '' ? null : +$event.target.value)" type="number" min="0" step="0.1" />
          </div>
          <div class="field">
            <label>CRP <span class="unit">(mg/L)</span></label>
            <input :value="form.crp" @input="update('crp', $event.target.value === '' ? null : +$event.target.value)" type="number" min="0" step="0.1" />
          </div>
          <div class="field">
            <label>Insulin <span class="unit">(µU/mL)</span></label>
            <input :value="form.insulin" @input="update('insulin', $event.target.value === '' ? null : +$event.target.value)" type="number" min="0" step="0.1" />
          </div>
          <div class="field">
            <label>BUN <span class="unit">(mg/dL)</span></label>
            <input :value="form.bun" @input="update('bun', $event.target.value === '' ? null : +$event.target.value)" type="number" min="0" step="0.1" />
          </div>
          <div class="field">
            <label>Fasting Glucose <span class="unit">(mg/dL)</span></label>
            <input :value="form.glucose" @input="update('glucose', $event.target.value === '' ? null : +$event.target.value)" type="number" min="0" step="0.1" />
          </div>
        </div>
      </div>
    </div>
    <!-- end Mortality & Lab Values -->

    <!-- Lifestyle & AI Factors foldout -->
    <div class="advanced-section">
      <button type="button" class="advanced-toggle" @click="showLifestyle = !showLifestyle" :aria-expanded="showLifestyle">
        <span class="advanced-toggle-label">
          <svg class="adv-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          Lifestyle & AI Factors
        </span>
        <svg class="chevron" :class="{ open: showLifestyle }" viewBox="0 0 20 20" fill="currentColor">
          <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
        </svg>
      </button>

      <div class="advanced-body" :class="{ expanded: showLifestyle }">
        <p class="advanced-hint">
          Habits and lifestyle factors to generate customized health coaching recommendations.
        </p>
        <div class="grid">
          <div class="field">
            <label>Exercise Days/Week</label>
            <input :value="form.exercise_days_per_week" @input="update('exercise_days_per_week', $event.target.value === '' ? null : +$event.target.value)" type="number" min="0" max="7" step="0.5" />
          </div>
          <div class="field">
            <label>Diet Quality</label>
            <select :value="form.diet_quality || ''" @change="update('diet_quality', $event.target.value || null)">
              <option value="">— Not specified —</option>
              <option value="poor">Poor</option>
              <option value="average">Average</option>
              <option value="good">Good</option>
            </select>
          </div>
          <div class="field">
            <label>Alcohol Units/Week</label>
            <input :value="form.alcohol_units_per_week" @input="update('alcohol_units_per_week', $event.target.value === '' ? null : +$event.target.value)" type="number" min="0" />
          </div>
          <div class="field">
            <label>Stress Level</label>
            <select :value="form.stress_level || ''" @change="update('stress_level', $event.target.value || null)">
              <option value="">— Not specified —</option>
              <option value="low">Low</option>
              <option value="moderate">Moderate</option>
              <option value="high">High</option>
            </select>
          </div>
        </div>
      </div>
    </div>
    <!-- end Lifestyle & AI Factors -->

    <button class="btn" type="submit">Submit</button>

    <!-- Results -->
    <div v-if="result" class="results">
      <div class="result-card prob">
        <span class="result-label">CVD Probability</span>
        <span class="result-value">{{ (result.cvd_probability * 100).toFixed(1) }}%</span>
      </div>
      <div class="result-card risk" :class="result.risk_level">
        <span class="result-label">Risk Level</span>
        <span class="result-value capitalize">{{ result.risk_level }}</span>
      </div>
      <div class="result-card full">
        <span class="result-label">Assessed For</span>
        <span class="result-desc">{{ result.label }}</span>
      </div>

      <template v-if="result.explanation && !result.explanation.error">
        <div class="result-card full drivers">
          <span class="result-label">Top Risk Drivers</span>
          <ul class="factor-list">
            <li v-for="f in result.explanation.top_risk_drivers" :key="f.feature" class="factor risk-factor">
              <span class="factor-name">{{ f.label }}</span>
              <span class="factor-shap">+{{ f.shap.toFixed(3) }}</span>
            </li>
          </ul>
        </div>
        <div class="result-card full protectors">
          <span class="result-label">Top Protective Factors</span>
          <ul class="factor-list">
            <li v-for="f in result.explanation.top_protective_factors" :key="f.feature" class="factor protect-factor">
              <span class="factor-name">{{ f.label }}</span>
              <span class="factor-shap">{{ f.shap.toFixed(3) }}</span>
            </li>
          </ul>
        </div>
        <div class="result-card full">
          <span class="result-label">Note</span>
          <span class="result-desc">{{ result.explanation.note }}</span>
        </div>
      </template>

      <div v-else-if="result.explanation?.error" class="result-card full">
        <span class="result-label">Explanation Unavailable</span>
        <span class="result-desc">{{ result.explanation.error }}</span>
      </div>
    </div>

  </form>
</template>

<script setup>
import { ref } from "vue"

// Props & emits
const props = defineProps({
  form:   { type: Object,  required: true },
  result: { type: Object,  default: null  },
})

const emit = defineEmits(["update:form", "submit"])

function update(key, value) {
  emit("update:form", { ...props.form, [key]: value })
}

// Local UI state not shared with other tabs
const showAdvanced = ref(false)
const showLabs = ref(false)
const showLifestyle = ref(false)
</script>

<style scoped>
* {
  font-family: monospace;
}

.title {
  margin-top: 0;
  margin-bottom: 20px;
  font-size: 20px;
  font-weight: 600;
  color: #111827;
}

.grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

label {
  font-size: 13px;
  color: #374151;
}

input,
select {
  padding: 10px;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  outline: none;
  background: white;
  font-size: 14px;
  color: #111827;
  font-family: monospace;
}

input:focus,
select:focus {
  border-color: #6366f1;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
}

.toggle {
  display: flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
  user-select: none;
}

.toggle input {
  width: 18px;
  height: 18px;
}

/* Advanced foldout */
.advanced-section {
  margin-top: 24px;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  overflow: hidden;
}

.advanced-toggle {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 13px 16px;
  background: #f8fafc;
  border: none;
  cursor: pointer;
  font-size: 14px;
  font-weight: 600;
  color: #374151;
  transition: background 0.15s;
  gap: 8px;
  font-family: monospace;
}

.advanced-toggle:hover {
  background: #f1f5f9;
}

.advanced-toggle-label {
  display: flex;
  align-items: center;
  gap: 8px;
}

.adv-icon {
  width: 16px;
  height: 16px;
  color: #6366f1;
  flex-shrink: 0;
}

.chevron {
  width: 18px;
  height: 18px;
  color: #94a3b8;
  flex-shrink: 0;
  transition: transform 0.25s ease;
}

.chevron.open {
  transform: rotate(180deg);
}

.advanced-body {
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.35s ease, padding 0.2s ease;
  padding: 0 16px;
}

.advanced-body.expanded {
  max-height: 1000px;
  padding: 16px;
}

.advanced-hint {
  margin: 0 0 14px;
  font-size: 12px;
  color: #94a3b8;
  line-height: 1.5;
}

.unit {
  font-weight: 400;
  color: #9ca3af;
  font-size: 11px;
}

/* Submit button */
.btn {
  margin-top: 20px;
  width: 100%;
  padding: 12px;
  border: none;
  border-radius: 10px;
  background: #4f46e5;
  color: white;
  font-weight: 600;
  cursor: pointer;
  transition: 0.2s;
  font-family: monospace;
}

.btn:hover {
  background: #4338ca;
}

/* Results */
.results {
  margin-top: 20px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.result-card {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.result-card.full {
  grid-column: 1 / -1;
}

.result-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #94a3b8;
}

.result-value {
  font-size: 28px;
  font-weight: 700;
  color: #0f172a;
}

.result-desc {
  font-size: 14px;
  color: #374151;
  line-height: 1.5;
}

.capitalize {
  text-transform: capitalize;
}

.result-card.risk.low             { border-color: #86efac; background: #f0fdf4; }
.result-card.risk.low .result-value { color: #16a34a; }

.result-card.risk.moderate             { border-color: #fde68a; background: #fffbeb; }
.result-card.risk.moderate .result-value { color: #d97706; }

.result-card.risk.high             { border-color: #fca5a5; background: #fef2f2; }
.result-card.risk.high .result-value { color: #dc2626; }

.factor-list {
  list-style: none;
  padding: 0;
  margin: 8px 0 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.factor {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  border-radius: 8px;
  font-size: 14px;
}

.risk-factor   { background: #fef2f2; color: #991b1b; }
.protect-factor { background: #f0fdf4; color: #166534; }

.factor-name { font-weight: 500; }
.factor-shap { font-family: monospace; font-size: 13px; opacity: 0.8; }

@media (max-width: 640px) {
  .grid { grid-template-columns: 1fr; }
}
</style>