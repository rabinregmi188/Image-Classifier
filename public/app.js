const STORAGE_KEY = "image-classifier-history-v1"

const prototypes = [
  {
    label: "Landscape",
    vector: [0.62, 0.28, 0.5, 0.25, 0.36, 0.48, 0.26, 0.34, 0.1, 0.42],
  },
  {
    label: "Portrait",
    vector: [0.58, 0.24, 0.4, 0.42, 0.16, 0.12, 0.18, 0.82, 0.05, 0.71],
  },
  {
    label: "Document",
    vector: [0.86, 0.22, 0.05, 0.12, 0.1, 0.08, 0.64, 0.24, 0.92, 0.38],
  },
  {
    label: "Night",
    vector: [0.18, 0.34, 0.25, 0.08, 0.56, 0.1, 0.22, 0.31, 0.02, 0.48],
  },
  {
    label: "Abstract",
    vector: [0.52, 0.58, 0.81, 0.4, 0.32, 0.28, 0.71, 0.44, 0.06, 0.35],
  },
]

const state = {
  file: null,
  previewUrl: "",
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

function getHistory() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]")
  } catch {
    return []
  }
}

function saveHistory(items) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(items))
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

function formatFeatureName(name) {
  return name
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ")
}

function softmax(values) {
  const max = Math.max(...values)
  const exps = values.map((value) => Math.exp(value - max))
  const sum = exps.reduce((total, value) => total + value, 0)
  return exps.map((value) => value / sum)
}

function distance(a, b) {
  return Math.sqrt(a.reduce((total, value, index) => total + (value - b[index]) ** 2, 0))
}

function extractFeatures(imageData, width, height) {
  const data = imageData.data
  const totalPixels = width * height

  let brightnessSum = 0
  let brightnessSquaredSum = 0
  let saturationSum = 0
  let warmCount = 0
  let coolCount = 0
  let greenCount = 0
  let documentBrightCount = 0
  let documentLowSatCount = 0

  const brightnessMap = new Float32Array(totalPixels)

  for (let i = 0; i < totalPixels; i += 1) {
    const offset = i * 4
    const red = data[offset] / 255
    const green = data[offset + 1] / 255
    const blue = data[offset + 2] / 255

    const brightness = (red + green + blue) / 3
    const maxChannel = Math.max(red, green, blue)
    const minChannel = Math.min(red, green, blue)
    const saturation = maxChannel - minChannel

    brightnessMap[i] = brightness
    brightnessSum += brightness
    brightnessSquaredSum += brightness * brightness
    saturationSum += saturation
    if (red > green && green > blue * 0.7) warmCount += 1
    if (blue > red && blue > green * 0.8) coolCount += 1
    if (green > red && green > blue) greenCount += 1
    if (brightness > 0.82) documentBrightCount += 1
    if (saturation < 0.08) documentLowSatCount += 1
  }

  const brightness = brightnessSum / totalPixels
  const variance = Math.max(brightnessSquaredSum / totalPixels - brightness ** 2, 0)
  const contrast = Math.sqrt(variance)
  const saturation = saturationSum / totalPixels

  let edgeSum = 0
  let edgeCount = 0

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x
      if (x + 1 < width) {
        edgeSum += Math.abs(brightnessMap[index] - brightnessMap[index + 1])
        edgeCount += 1
      }
      if (y + 1 < height) {
        edgeSum += Math.abs(brightnessMap[index] - brightnessMap[index + width])
        edgeCount += 1
      }
    }
  }

  const centerY1 = Math.floor(height * 0.25)
  const centerY2 = Math.max(Math.floor(height * 0.75), centerY1 + 1)
  const centerX1 = Math.floor(width * 0.25)
  const centerX2 = Math.max(Math.floor(width * 0.75), centerX1 + 1)

  let centerSum = 0
  let centerCount = 0

  for (let y = centerY1; y < centerY2; y += 1) {
    for (let x = centerX1; x < centerX2; x += 1) {
      centerSum += brightnessMap[y * width + x]
      centerCount += 1
    }
  }

  return {
    brightness,
    contrast,
    saturation,
    warm_ratio: warmCount / totalPixels,
    cool_ratio: coolCount / totalPixels,
    green_ratio: greenCount / totalPixels,
    edge_density: edgeCount ? edgeSum / edgeCount : 0,
    portrait_bias: Math.min(height / Math.max(width, 1), 2) / 2,
    document_bias: (documentBrightCount / totalPixels) * 0.65 + (documentLowSatCount / totalPixels) * 0.35,
    center_focus: centerCount ? centerSum / centerCount : brightness,
  }
}

async function classifyFile(file) {
  const imageUrl = URL.createObjectURL(file)
  const image = await new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = imageUrl
  })

  const width = image.naturalWidth || image.width
  const height = image.naturalHeight || image.height
  const canvas = document.createElement("canvas")
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext("2d")
  context.drawImage(image, 0, 0, width, height)
  const imageData = context.getImageData(0, 0, width, height)

  const features = extractFeatures(imageData, width, height)
  const featureVector = [
    features.brightness,
    features.contrast,
    features.saturation,
    features.warm_ratio,
    features.cool_ratio,
    features.green_ratio,
    features.edge_density,
    features.portrait_bias,
    features.document_bias,
    features.center_focus,
  ]

  const logits = prototypes.map((prototype) => -4 * distance(prototype.vector, featureVector))
  const probabilities = softmax(logits)

  const predictions = prototypes
    .map((prototype, index) => ({
      label: prototype.label,
      confidence: Number(probabilities[index].toFixed(4)),
    }))
    .sort((left, right) => right.confidence - left.confidence)

  URL.revokeObjectURL(imageUrl)

  return {
    label: predictions[0].label,
    confidence: predictions[0].confidence,
    predictions: predictions.slice(0, 3),
    features: Object.fromEntries(
      Object.entries(features).map(([key, value]) => [key, Number(value.toFixed(4))])
    ),
    width,
    height,
  }
}

function renderResult(result) {
  emptyResult.classList.add("hidden")
  resultContent.classList.remove("hidden")
  resultImage.src = state.previewUrl
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

function renderHistory() {
  const items = getHistory()

  if (!items.length) {
    historyGrid.innerHTML = `
      <div class="empty-state">Prediction history will appear here after your first upload.</div>
    `
    return
  }

  historyGrid.innerHTML = items
    .map(
      (item) => `
        <article class="history-item">
          <div class="history-thumb" aria-hidden="true">${item.label.slice(0, 1)}</div>
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

function renderStats() {
  const items = getHistory()
  totalPredictions.textContent = String(items.length)

  const average =
    items.reduce((total, item) => total + item.confidence, 0) / (items.length || 1)
  averageConfidence.textContent = `${Math.round(average * 100)}%`

  const counts = items.reduce((accumulator, item) => {
    accumulator[item.label] = (accumulator[item.label] || 0) + 1
    return accumulator
  }, {})

  const top = Object.entries(counts).sort((left, right) => right[1] - left[1])[0]
  topLabel.textContent = top ? top[0] : "None yet"
}

async function analyzeCurrentFile() {
  if (!state.file) {
    setStatus("Choose an image before running prediction.", "error")
    return
  }

  setStatus("Analyzing image locally...", "info")
  analyzeButton.disabled = true

  try {
    const result = await classifyFile(state.file)
    renderResult(result)

    const items = [
      {
        id: Date.now(),
        original_name: state.file.name,
        label: result.label,
        confidence: result.confidence,
        created_at: new Date().toISOString(),
      },
      ...getHistory(),
    ].slice(0, 12)

    saveHistory(items)
    renderHistory()
    renderStats()
    setStatus(`Prediction complete: ${result.label}`, "info")
  } catch (error) {
    setStatus(error.message || "Prediction failed.", "error")
  } finally {
    analyzeButton.disabled = false
  }
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
renderHistory()
renderStats()
