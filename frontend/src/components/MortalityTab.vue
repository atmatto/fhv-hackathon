<template>
  <div class="tab-container">
    <h2 class="title">Mortality Statistics</h2>

    <div v-if="!mortalityResult" class="empty-state">
      <div class="empty-icon">📊</div>
      <p>Submit the form on the first tab to generate mortality predictions.</p>
    </div>

    <div v-else class="results">
      <!-- ── Risk Band ── -->
      <div class="result-card risk" :class="riskClass">
        <span class="result-label">10-Year Risk Band</span>
        <span class="result-value capitalize">{{ mortalityResult.risk_band }}</span>
      </div>

      <!-- ── Doctor Visit ── -->
      <div class="result-card visit" :class="{ 'recommend': mortalityResult.recommend_doctor_visit }">
        <span class="result-label">Doctor Visit</span>
        <span class="result-value">{{ mortalityResult.recommend_doctor_visit ? 'Recommended' : 'Not Urgent' }}</span>
      </div>

      <!-- ── Death Risk Horizons ── -->
      <div v-if="mortalityResult.horizon_cvd_death_risk" class="result-card full horizons">
        <span class="result-label">CVD Death Risk Horizons</span>
        <div class="horizon-grid">
          <div class="horizon-item">
            <span class="h-label">1 Year</span>
            <span class="h-val">{{ formatPct(mortalityResult.horizon_cvd_death_risk['1y']) }}</span>
          </div>
          <div class="horizon-item">
            <span class="h-label">5 Years</span>
            <span class="h-val">{{ formatPct(mortalityResult.horizon_cvd_death_risk['5y']) }}</span>
          </div>
          <div class="horizon-item">
            <span class="h-label">10 Years</span>
            <span class="h-val">{{ formatPct(mortalityResult.horizon_cvd_death_risk['10y']) }}</span>
          </div>
        </div>
      </div>

      <!-- ── Survival Curve Line Chart (SVG) ── -->
      <div v-if="mortalityResult.survival_curve" class="result-card full chart-card">
        <span class="result-label">Survival Probability Curve</span>
        <div class="chart-container">
          <svg viewBox="0 0 600 240" class="line-chart" preserveAspectRatio="xMidYMid meet">
            
            <!-- Y Axis Title -->
            <text x="15" y="120" text-anchor="middle" class="axis-title-y" transform="rotate(-90, 15, 120)">Survival Probability</text>
            
            <!-- X Axis Title -->
            <text x="300" y="230" text-anchor="middle" class="axis-title-x">Time (Years)</text>
            
            <!-- X Axis Line -->
            <line x1="30" y1="190" x2="570" y2="190" stroke="#e5e7eb" stroke-width="2" stroke-linecap="round" />

            <!-- Connecting Data Path -->
            <path :d="chartPath" fill="none" stroke="#4f46e5" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />

            <!-- Points, Guidelines, and Text Labels -->
            <g v-for="(pt, i) in chartPoints" :key="i">
              
              <!-- Vertical dashed drop line to X-axis -->
              <line 
                :x1="pt.cx" :y1="pt.cy + 5" 
                :x2="pt.cx" y2="190" 
                stroke="#d1d5db" 
                stroke-width="1.5" 
                stroke-dasharray="4" 
              />
              
              <!-- Data point circle -->
              <circle 
                :cx="pt.cx" 
                :cy="pt.cy" 
                r="5" 
                fill="#ffffff" 
                stroke="#4f46e5" 
                stroke-width="2.5" 
              />
              
              <!-- Percentage value above point -->
              <text :x="pt.cx" :y="pt.cy - 14" text-anchor="middle" class="chart-val">
                {{ i % 2 == 0 ? pt.val : "" }}
              </text>
              
              <!-- Year label below X-axis -->
              <text :x="pt.cx" y="215" text-anchor="middle" class="chart-label">
                {{ pt.x }}
              </text>
              
            </g>
            
          </svg>
        </div>
      </div>

      <!-- ── Median Years ── -->
      <div v-if="mortalityResult.median_years_to_cvd_death" class="result-card full">
        <span class="result-label">Median Years to CVD Death</span>
        <span class="result-value-small">{{ mortalityResult.median_years_to_cvd_death }} years</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from "vue"

const props = defineProps({
  mortalityResult: {
    type: Object,
    default: null
  }
})

// Maps the mortality 'risk_band' to the shared CSS color themes
const riskClass = computed(() => {
  if (!props.mortalityResult?.risk_band) return ''
  const band = props.mortalityResult.risk_band.toLowerCase()
  if (band === 'low') return 'low'
  if (band === 'elevated') return 'high' 
  return 'moderate'
})

function formatPct(val) {
  if (val == null) return '--'
  return (val * 100).toFixed(1) + '%'
}

// ── SVG Line Chart Logic ──
const chartPoints = computed(() => {
  if (!props.mortalityResult?.survival_curve) return []
  
  const curve = props.mortalityResult.survival_curve
  // Extract and sort years (X axis)
  const xData = Object.keys(curve).map(Number).sort((a, b) => a - b)
  if (xData.length === 0) return []
  
  const yData = xData.map(x => curve[x])

  // Chart configuration ranges
  const minX = Math.min(...xData)
  const maxX = Math.max(...xData) || minX + 1

  // Dynamic Y-axis scaling to make the line drop visible even if survival is high 
  const minYVal = Math.min(...yData)
  const minY = Math.max(0, minYVal - 0.05) // Add 5% padding below lowest point
  const maxY = 1.0
  const rangeY = maxY - minY || 0.1

  // SVG dimensions & paddings matching viewBox="0 0 600 240"
  const svgWidth = 600
  const svgHeight = 240
  const paddingLeft = 40
  const paddingRight = 40
  const paddingTop = 45   // Space for text above max Y point
  const paddingBottom = 50 // Space for X-axis & labels

  const usableWidth = svgWidth - paddingLeft - paddingRight
  const usableHeight = svgHeight - paddingTop - paddingBottom

  return xData.map((x, index) => {
    const y = yData[index]
    
    // Calculate X coordinate
    const cx = paddingLeft + ((x - minX) / (maxX - minX)) * usableWidth
    
    // Calculate Y coordinate (inverted because SVG Y goes top-down)
    const cy = paddingTop + (1 - (y - minY) / rangeY) * usableHeight
    
    return {
      x,
      y,
      cx,
      cy,
      val: (y * 100).toFixed(1) + '%'
    }
  })
})

// Generate SVG `<path d="...">` command to draw the line
const chartPath = computed(() => {
  const pts = chartPoints.value
  if (pts.length < 2) return ''
  return pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.cx} ${p.cy}`).join(' ')
})
</script>

<style scoped>
* { font-family: monospace; }

.title {
  margin-top: 0;
  margin-bottom: 20px;
  font-size: 20px;
  font-weight: 600;
  color: #111827;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 20px;
  text-align: center;
  color: #6b7280;
  background: #f9fafb;
  border: 1px dashed #d1d5db;
  border-radius: 12px;
}

.empty-icon { font-size: 32px; margin-bottom: 12px; }

/* ── Results Grid ── */
.results {
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
.result-card.full { grid-column: 1 / -1; }

.result-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #94a3b8;
}

.result-value { font-size: 28px; font-weight: 700; color: #0f172a; }
.result-value-small { font-size: 20px; font-weight: 700; color: #0f172a; }
.capitalize { text-transform: capitalize; }

/* Shared risk colors */
.result-card.risk.low { border-color: #86efac; background: #f0fdf4; }
.result-card.risk.low .result-value { color: #16a34a; }
.result-card.risk.moderate { border-color: #fde68a; background: #fffbeb; }
.result-card.risk.moderate .result-value { color: #d97706; }
.result-card.risk.high { border-color: #fca5a5; background: #fef2f2; }
.result-card.risk.high .result-value { color: #dc2626; }

/* Doctor Visit colors */
.result-card.visit.recommend { border-color: #fca5a5; background: #fef2f2; }
.result-card.visit.recommend .result-value { color: #dc2626; }

/* ── Horizons ── */
.horizon-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  margin-top: 8px;
}
.horizon-item {
  display: flex;
  flex-direction: column;
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 10px;
  align-items: center;
}
.h-label { font-size: 12px; color: #6b7280; margin-bottom: 4px; }
.h-val { font-size: 18px; font-weight: 700; color: #4f46e5; }

/* ── Survival Curve Line Chart ── */
.chart-container {
  width: 100%;
  margin-top: 12px;
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  overflow: hidden;
}

.line-chart {
  display: block;
  width: 100%;
  height: auto;
  overflow: visible;
}

.chart-val {
  font-size: 12px;
  font-weight: 700;
  fill: #4f46e5;
}

.chart-label {
  font-size: 13px;
  font-weight: 600;
  fill: #6b7280;
}

/* Axis titles and labels */
.axis-title-x {
  font-size: 12px;
  font-weight: 600;
  fill: #374151;
}

.axis-title-y {
  font-size: 12px;
  font-weight: 600;
  fill: #374151;
}

.axis-tick {
  font-size: 10px;
  font-weight: 500;
  fill: #6b7280;
}

@media (max-width: 640px) {
  .results { grid-template-columns: 1fr; }
}
</style>