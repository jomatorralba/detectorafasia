/* Hanning window */
function hanning(n) {
  const w = new Float32Array(n)
  for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (n - 1)))
  return w
}

/* Radix-2 Cooley-Tukey FFT — returns magnitude of first n/2 bins */
function rfft(input) {
  const n = input.length
  const re = new Float64Array(n)
  const im = new Float64Array(n)
  for (let i = 0; i < n; i++) re[i] = input[i]

  // Bit-reversal permutation
  let j = 0
  for (let i = 1; i < n; i++) {
    let bit = n >> 1
    while (j & bit) { j ^= bit; bit >>= 1 }
    j ^= bit
    if (i < j) {
      ;[re[i], re[j]] = [re[j], re[i]]
      ;[im[i], im[j]] = [im[j], im[i]]
    }
  }

  // Butterfly stages
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len
    const wRe = Math.cos(ang), wIm = Math.sin(ang)
    for (let i = 0; i < n; i += len) {
      let tRe = 1, tIm = 0
      for (let k = 0; k < (len >> 1); k++) {
        const uRe = re[i + k], uIm = im[i + k]
        const half = i + k + (len >> 1)
        const vRe = re[half] * tRe - im[half] * tIm
        const vIm = re[half] * tIm + im[half] * tRe
        re[i + k] = uRe + vRe;  im[i + k] = uIm + vIm
        re[half]  = uRe - vRe;  im[half]  = uIm - vIm
        const nTRe = tRe * wRe - tIm * wIm
        tIm = tRe * wIm + tIm * wRe; tRe = nTRe
      }
    }
  }

  const mag = new Float32Array(n >> 1)
  for (let i = 0; i < (n >> 1); i++) mag[i] = Math.sqrt(re[i] ** 2 + im[i] ** 2)
  return mag
}

/* Short-Time Fourier Transform */
export function computeSpectrogram(channelData, sampleRate, fftSize = 512, hopSize = 128) {
  const win = hanning(fftSize)
  const frame = new Float32Array(fftSize)
  const frames = []

  for (let i = 0; i + fftSize <= channelData.length; i += hopSize) {
    for (let j = 0; j < fftSize; j++) frame[j] = (channelData[i + j] ?? 0) * win[j]
    frames.push(rfft(frame.slice()))
  }

  return { frames, sampleRate, fftSize, hopSize }
}

/* First voiced frame (5% of peak energy, with a small pre-roll) */
export function findOnsetFrame(frames, threshold = 0.05) {
  const energy = frames.map(f => { let s = 0; f.forEach(v => { s += v * v }); return s })
  const maxE = Math.max(...energy)
  if (maxE === 0) return 0
  const thr = maxE * threshold
  const idx = energy.findIndex(e => e > thr)
  return Math.max(0, (idx < 0 ? 0 : idx) - 3) // 3-frame pre-roll
}

/* Magma colormap (black → purple → orange → white) — for patient */
function magmaColor(t) {
  const stops = [[0,0,4],[87,15,110],[187,55,84],[249,142,9],[252,253,191]]
  t = Math.max(0, Math.min(1, t))
  const pos = t * (stops.length - 1)
  const lo = Math.floor(pos), hi = Math.min(Math.ceil(pos), stops.length - 1)
  const f = pos - lo
  return stops[lo].map((v, i) => Math.round(v + f * (stops[hi][i] - v)))
}

/* Cyan colormap (black → teal) — for reference */
function cyanColor(t) {
  t = Math.max(0, Math.min(1, t))
  return [Math.round(t * 20), Math.round(t * 210), Math.round(t * 190)]
}

/* Render frames → ImageData (onset-aligned, normalized 60 dB range) */
function renderFrames(frames, onsetFrame, w, h, fftSize, sampleRate, maxFreqHz, colorFn) {
  const maxBin = Math.min(
    Math.floor((maxFreqHz / (sampleRate / 2)) * (fftSize / 2)),
    (fftSize >> 1)
  )
  const active = frames.slice(onsetFrame)
  const nF = active.length

  // dB per frame/bin
  const dbF = active.map(f =>
    Array.from({ length: maxBin }, (_, b) => 20 * Math.log10(Math.max(f[b], 1e-8)))
  )
  const all = dbF.flat()
  const maxDb = Math.max(...all)
  const minDb = maxDb - 60

  const img = new ImageData(w, h)
  for (let x = 0; x < w; x++) {
    const fi = Math.min(nF - 1, Math.floor((x / w) * nF))
    for (let y = 0; y < h; y++) {
      const bi = Math.min(maxBin - 1, Math.floor(((h - 1 - y) / h) * maxBin))
      const t  = Math.max(0, Math.min(1, (dbF[fi][bi] - minDb) / (maxDb - minDb)))
      const [r, g, b] = colorFn(t)
      const idx = (y * w + x) * 4
      img.data[idx] = r; img.data[idx+1] = g; img.data[idx+2] = b; img.data[idx+3] = 255
    }
  }
  return img
}

/* Blend patient (magma) + reference (cyan) on black background — additive */
export function blendSpectrograms(patSpec, patOnset, refSpec, refOnset, w, h, maxFreqHz = 8000) {
  const patImg = renderFrames(patSpec.frames, patOnset, w, h, patSpec.fftSize, patSpec.sampleRate, maxFreqHz, magmaColor)
  const refImg = renderFrames(refSpec.frames, refOnset, w, h, refSpec.fftSize, refSpec.sampleRate, maxFreqHz, cyanColor)

  const out = new ImageData(w, h)
  for (let i = 0; i < out.data.length; i += 4) {
    out.data[i]   = Math.min(255, patImg.data[i]   + Math.round(refImg.data[i]   * 0.65))
    out.data[i+1] = Math.min(255, patImg.data[i+1] + Math.round(refImg.data[i+1] * 0.65))
    out.data[i+2] = Math.min(255, patImg.data[i+2] + Math.round(refImg.data[i+2] * 0.65))
    out.data[i+3] = 255
  }
  return out
}

/* Just patient, no reference */
export function renderPatientSpectrogram(patSpec, patOnset, w, h, maxFreqHz = 8000) {
  return renderFrames(patSpec.frames, patOnset, w, h, patSpec.fftSize, patSpec.sampleRate, maxFreqHz, magmaColor)
}
