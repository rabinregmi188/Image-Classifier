const state = {
  file: null,
  previewUrl: "",
  latestResult: null,
}

const dropZone = document.getElementById("dropZone")
const fileInput = document.getElementById("fileInput")
const previewState = document.getElementById("previewState")
const previewImage = document.getElementById("previewImage")
const previewName = document.getElementById("previewName")
const analyzeButton = document.getElementById("analyzeButton")
const sampleButton = document.getElementById("sampleButton")
const statusBanner = document.getElementById("statusBanner")
const resultContent = document.getElementById("resultContent")
const emptyResult = document.getElementById("emptyResult")
const resultImage = document.getElementById("resultImage")
const resultLabel = document.getElementById("resultLabel")
const resultConfidence = document.getElementById("resultConfidence")
const resultDimensions = document.getElementById("resultDimensions")
const predictionBars = document.getElementById("predictionBars")
const featureGrid = document.getElementById("featureGrid")
const historyGrid = document.getElementById("historyGrid")
const totalPredictions = document.getElementById("totalPredictions")
const averageConfidence = document.getElementById("averageConfidence")
const topLabel = document.getElementById("topLabel")

function setStatus(message, type = "info") {
  statusBanner.textContent = message
  statusBanner.className = `status-banner ${type}`
  statusBanner.classList.remove("hidden")
}

function clearStatus() {
  statusBanner.classList.add("hidden")
}

function setSelectedFile(file) {
  state.file = file
  if (state.previewUrl) {
    URL.revokeObjectURL(state.previewUrl)
  }
  state.previewUrl = URL.createObjectURL(file)
  previewImage.src = state.previewUrl
  previewName.textContent = file.name
  previewState.classList.remove("hidden")
  clearStatus()
}

function bindDragAndDrop() {
  ;["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault()
      dropZone.classList.add("dragover")
    })
  })

  ;["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault()
      dropZone.classList.remove("dragover")
    })
  })

  dropZone.addEventListener("drop", (event) => {
    const [file] = event.dataTransfer.files
    if (file) {
      setSelectedFile(file)
    }
  })

  fileInput.addEventListener("change", (event) => {
    const [file] = event.target.files
    if (file) {
      setSelectedFile(file)
    }
  })
}

function renderResult(result) {
  state.latestResult = result
  emptyResult.classList.add("hidden")
  resultContent.classList.remove("hidden")
  resultImage.src = result.image_url
  resultLabel.textContent = result.label
  resultConfidence.textContent = `Confidence: ${Math.round(result.confidence * 100)}%`
  resultDimensions.textContent = `${result.width} x ${result.height}`

  predictionBars.innerHTML = result.predictions
    .map(
      (item) => `
        <div class="prediction-row">
          <header>
            <span>${item.label}</span>
            <strong>${Math.round(item.confidence * 100)}%</strong>
          </header>
          <div class="bar-track">
            <div class="bar-fill" style="width: ${Math.max(item.confidence * 100, 4)}%"></div>
          </div>
        </div>
      `
    )
    .join("")

  featureGrid.innerHTML = Object.entries(result.features)
    .map(
      ([key, value]) => `
        <article class="feature-card">
          <p>${formatFeatureName(key)}</p>
          <strong>${Math.round(value * 100)}%</strong>
        </article>
      `
    )
    .join("")
}

function formatFeatureName(name) {
  return name
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ")
}

async function analyzeCurrentFile() {
  if (!state.file) {
    setStatus("Choose an image before running prediction.", "error")
    return
  }

  const formData = new FormData()
  formData.append("file", state.file)

  setStatus("Analyzing image and storing the prediction...", "info")
  analyzeButton.disabled = true

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    })
    const payload = await response.json()
    if (!response.ok) {
      throw new Error(payload.detail || "Prediction failed.")
    }

    renderResult(payload)
    setStatus(`Prediction complete: ${payload.label}`, "info")
    await Promise.all([loadHistory(), loadStats()])
  } catch (error) {
    setStatus(error.message || "Prediction failed.", "error")
  } finally {
    analyzeButton.disabled = false
  }
}

async function loadHistory() {
  const response = await fetch("/api/history")
  const payload = await response.json()

  if (!payload.items.length) {
    historyGrid.innerHTML = `
      <div class="empty-state">Prediction history will appear here after your first upload.</div>
    `
    return
  }

  historyGrid.innerHTML = payload.items
    .map(
      (item) => `
        <article class="history-item">
          <img src="/uploads/${item.stored_name}" alt="${item.label}" />
          <div class="history-meta">
            <div class="history-chip">${item.label}</div>
            <strong>${Math.round(item.confidence * 100)}% confidence</strong>
            <p>${item.original_name}</p>
            <p>${new Date(item.created_at).toLocaleString()}</p>
          </div>
        </article>
      `
    )
    .join("")
}

async function loadStats() {
  const response = await fetch("/api/stats")
  const payload = await response.json()
  totalPredictions.textContent = String(payload.total_predictions)
  averageConfidence.textContent = `${Math.round((payload.average_confidence || 0) * 100)}%`
  topLabel.textContent = payload.top_labels[0]?.label || "None yet"
}

function createSampleImage() {
  const canvas = document.createElement("canvas")
  canvas.width = 640
  canvas.height = 640
  const context = canvas.getContext("2d")

  const gradient = context.createLinearGradient(0, 0, 640, 640)
  gradient.addColorStop(0, "#ff8f6b")
  gradient.addColorStop(0.4, "#ff4db8")
  gradient.addColorStop(1, "#4b6bff")
  context.fillStyle = gradient
  context.fillRect(0, 0, 640, 640)

  context.fillStyle = "rgba(255,255,255,0.22)"
  context.beginPath()
  context.arc(470, 180, 86, 0, Math.PI * 2)
  context.fill()

  context.fillStyle = "#102144"
  context.fillRect(0, 460, 640, 180)

  context.fillStyle = "#29d8ff"
  context.beginPath()
  context.moveTo(70, 470)
  context.lineTo(230, 280)
  context.lineTo(360, 470)
  context.closePath()
  context.fill()

  context.fillStyle = "#0a1733"
  context.beginPath()
  context.moveTo(220, 470)
  context.lineTo(410, 220)
  context.lineTo(620, 470)
  context.closePath()
  context.fill()

  return new Promise((resolve) => {
    canvas.toBlob((blob) => {
      resolve(new File([blob], "sample-scene.png", { type: "image/png" }))
    }, "image/png")
  })
}

analyzeButton.addEventListener("click", analyzeCurrentFile)

sampleButton.addEventListener("click", async () => {
  const sampleFile = await createSampleImage()
  setSelectedFile(sampleFile)
  setStatus("Sample image loaded. Run prediction to test the app.", "info")
})

bindDragAndDrop()
loadHistory()
loadStats()
