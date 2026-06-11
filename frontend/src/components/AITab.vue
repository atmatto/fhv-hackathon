<template>
  <div class="tab-container">
    <h2 class="title">AI Lifestyle Insights</h2>

    <div v-if="!AIResult" class="empty-state">
      <div class="empty-icon">✨</div>
      <p>Submit the form on the first tab to generate personalized AI recommendations.</p>
    </div>

    <div v-else class="results">
      <!-- AI Disabled -->
      <div v-if="AIResult.status === 'disabled'" class="result-card full warn">
        <span class="result-label">Feature Disabled</span>
        <span class="result-desc">The AI generation is currently disabled (API key not set on the backend).</span>
      </div>

      <!-- AI Error -->
      <div v-else-if="AIResult.status === 'error'" class="result-card full error">
        <span class="result-label">Generation Failed</span>
        <span class="result-desc">{{ AIResult.error_detail || 'An unknown error occurred while contacting the AI.' }}</span>
      </div>

      <!-- AI Success -->
      <div v-else-if="AIResult.suggestions" class="result-card full ai-content">
        <div class="ai-header">
          <span class="result-label">Actionable Plan</span>
        </div>
        <div class="suggestions-text" v-html="formattedSuggestions"></div>
      </div>

      <!-- Fallback -->
      <div v-else class="result-card full">
        <span class="result-label">No insights available</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from "vue"

const props = defineProps({
  AIResult: {
    type: Object,
    default: null
  }
})

// Computes the text while replacing newlines and formatting **bold** tags 
// (since LLMs natively respond with markdown asterisks)
const formattedSuggestions = computed(() => {
  if (!props.AIResult || !props.AIResult.suggestions) return ''
  
  let text = props.AIResult.suggestions

  // Basic markdown sanitization for bold
  text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
  
  return text
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

/* Results Grid */
.results {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.result-card {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.result-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #94a3b8;
}

.result-desc {
  font-size: 14px;
  color: #374151;
  line-height: 1.5;
}

.warn {
  border-color: #fde68a;
  background: #fffbeb;
}

.error {
  border-color: #fca5a5;
  background: #fef2f2;
}
.error .result-label { color: #dc2626; }

/* AI Suggestions Text */
.ai-header {
  border-bottom: 1px solid #e2e8f0;
  padding-bottom: 12px;
  margin-bottom: 4px;
}

.suggestions-text {
  font-size: 17px;
  line-height: 1.6;
  color: #1f2937;
  white-space: pre-wrap;
  font-family: 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
  font-weight: 400;
}

/* Scoped specifically targeting any v-html generated <strong> tags */
:deep(.suggestions-text strong) {
  color: #4f46e5;
  font-weight: 700;
}
</style>