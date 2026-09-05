function safeJsonParse(s, fallback) {
  try {
    return JSON.parse(s)
  } catch {
    return fallback
  }
}

function initDropzone() {
  const dz = document.querySelector('[data-dropzone]')
  const input = document.querySelector('[data-file-input]')
  if (!dz || !input) return

  dz.addEventListener('click', () => input.click())
  dz.addEventListener('dragover', (e) => {
    e.preventDefault()
    dz.classList.add('ring-4', 'ring-blue-100', 'border-blue-300')
  })
  dz.addEventListener('dragleave', () => {
    dz.classList.remove('ring-4', 'ring-blue-100', 'border-blue-300')
  })
  dz.addEventListener('drop', (e) => {
    e.preventDefault()
    dz.classList.remove('ring-4', 'ring-blue-100', 'border-blue-300')
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]
    if (!file) return
    const dt = new DataTransfer()
    dt.items.add(file)
    input.files = dt.files
  })
}

function setProgress(pct) {
  const bar = document.getElementById('progressBar')
  const label = document.getElementById('progressLabel')
  if (bar) bar.style.width = `${pct}%`
  if (label) label.textContent = `${pct}%`
}

function appendLog(line) {
  const box = document.getElementById('logBox')
  if (!box) return
  const div = document.createElement('div')
  div.textContent = line
  box.appendChild(div)
  box.scrollTop = box.scrollHeight
}

function setStep(step, state) {
  const el = document.querySelector(`[data-step="${step}"]`)
  if (!el) return
  el.classList.remove('is-on', 'is-done')
  if (state === 'on') el.classList.add('is-on')
  if (state === 'done') el.classList.add('is-done')
}

function initTraining() {
  const cfg = window.__TRAINING__
  if (!cfg) return
  const btn = document.getElementById('btnTrain')
  if (!btn) return
  const timeLabel = document.getElementById('timeLabel')

  function stream(runId) {
    const es = new EventSource(`/admin/training/stream/${runId}`)
    es.onmessage = (ev) => {
      const payload = safeJsonParse(ev.data, null)
      if (!payload) return
      if (typeof payload.progress === 'number') setProgress(payload.progress)
      if (payload.log) appendLog(payload.log)
      if (payload.step) {
        if (payload.step.state === 'on') setStep(payload.step.name, 'on')
        if (payload.step.state === 'done') setStep(payload.step.name, 'done')
      }
      if (payload.elapsed_seconds != null && timeLabel) {
        timeLabel.textContent = `${payload.elapsed_seconds.toFixed(1)} s`
      }
      if (payload.status === 'done' || payload.status === 'error') {
        es.close()
        btn.disabled = false
        if (payload.status === 'done') {
          appendLog('Training selesai. Silakan buka Evaluasi Model.')
        }
        if (payload.status === 'error') {
          appendLog(`Training gagal: ${payload.error || 'unknown error'}`)
        }
      }
    }
    es.onerror = () => {
      es.close()
    }
  }

  if (cfg.lastRunId) {
    stream(cfg.lastRunId)
  }

  btn.addEventListener('click', async () => {
    if (!cfg.canTrain) return
    btn.disabled = true
    setProgress(0)
    const logBox = document.getElementById('logBox')
    if (logBox) logBox.innerHTML = ''
    for (const s of ['read','clean','encode','split','train','eval','cm','fi','shap','save']) {
      setStep(s, '')
    }
    appendLog('Memulai training...')
    const res = await fetch('/admin/training/start', { method: 'POST' })
    const data = await res.json().catch(() => null)
    if (!data || !data.run_id) {
      btn.disabled = false
      appendLog('Gagal memulai training.')
      return
    }
    stream(data.run_id)
  })
}

function buildStackedBarConfusion(canvasId, cm) {
  const el = document.getElementById(canvasId)
  if (!el || !cm) return
  const labels = cm.labels || []
  const matrix = cm.matrix || []
  const datasets = []
  for (let trueIdx = 0; trueIdx < labels.length; trueIdx++) {
    const row = matrix[trueIdx] || []
    datasets.push({
      label: `True: ${labels[trueIdx]}`,
      data: labels.map((_, predIdx) => row[predIdx] || 0),
    })
  }
  const colors = ['#2563eb','#60a5fa','#93c5fd','#10b981','#f59e0b','#ef4444','#a855f7','#0ea5e9']
  datasets.forEach((d, i) => {
    d.backgroundColor = colors[i % colors.length]
    d.borderWidth = 0
    d.stack = 'cm'
  })

  new Chart(el, {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom' } },
      scales: {
        x: { stacked: true, ticks: { color: '#475569' }, grid: { color: 'rgba(148,163,184,.2)' } },
        y: { stacked: true, ticks: { color: '#475569' }, grid: { color: 'rgba(148,163,184,.2)' } },
      },
    },
  })
}

function buildFeatureImportance(canvasId, fi) {
  const el = document.getElementById(canvasId)
  if (!el || !fi) return
  const labels = fi.labels || []
  const values = fi.values || []
  new Chart(el, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Importance',
        data: values,
        backgroundColor: '#2563eb',
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#475569' }, grid: { color: 'rgba(148,163,184,.2)' } },
        y: { ticks: { color: '#475569' }, grid: { display: false } },
      },
    },
  })
}

function buildProbabilities(canvasId, probs) {
  const el = document.getElementById(canvasId)
  if (!el || !probs) return
  new Chart(el, {
    type: 'bar',
    data: {
      labels: probs.labels || [],
      datasets: [{
        label: 'Probability',
        data: probs.values || [],
        backgroundColor: '#2563eb',
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, max: 1, ticks: { color: '#475569' }, grid: { color: 'rgba(148,163,184,.2)' } },
        x: { ticks: { color: '#475569' }, grid: { display: false } },
      },
    },
  })
}

function initCharts() {
  if (window.__MODEL_CHARTS__) {
    const d = window.__MODEL_CHARTS__
    buildStackedBarConfusion('cmChart', d.confusion)
    buildFeatureImportance('fiChart', d.feature_importance)
  }
  if (window.__EVAL__) {
    const d = window.__EVAL__
    buildStackedBarConfusion('cmChart', d.confusion)
    buildFeatureImportance('fiChart', d.feature_importance)
  }
  if (window.__TRANSPARENCY__) {
    const d = window.__TRANSPARENCY__
    buildFeatureImportance('fiChart', d.feature_importance)
  }
  if (window.__RESULT__) {
    const d = window.__RESULT__
    buildProbabilities('probChart', d.probabilities)
    buildFeatureImportance('contribChart', d.contributions)
  }
}

initDropzone()
initTraining()
initCharts()

