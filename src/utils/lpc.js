// Análisis de formantes por LPC (Linear Predictive Coding)
// para extracción del espacio vocálico F1/F2/F3.
//
// Pipeline:
//   1. Decimación 48 kHz → 16 kHz (FIR low-pass + downsample)
//   2. Pre-énfasis (α = 0.97)
//   3. Zona estable 30–80 % de la vocal
//   4. Frames 25 ms / hop 10 ms con ventana Hamming
//   5. Autocorrelación → Levinson-Durbin → LPC (p = 14)
//   6. Espectro LPC → picos espectrales → F1, F2, F3
//   7. Mediana entre frames para robustez
//
// Referencias:
//   Markel & Gray (1976) "Linear Prediction of Speech"
//   Snell & Milinazzo (1993) — detección de formantes por picos espectrales
//   Sapir et al. (2010) — VSA y FCR como biomarcadores de disartria hipocinética

// ─── Decimación con filtro FIR anti-aliasing ───────────────────────

function decimateTo16k(pcm, fromSr) {
  const toSr = 16000
  if (fromSr === toSr) return { pcm, sr: toSr }

  const dec = Math.round(fromSr / toSr)  // p.ej. 3 para 48 kHz→16 kHz
  const cutoff = 0.5 / dec               // frecuencia de corte normalizada
  const M = 31                           // orden del filtro
  const h = new Float32Array(M + 1)
  const center = M / 2
  let sum = 0
  for (let i = 0; i <= M; i++) {
    const sinc = i === center
      ? 2 * cutoff
      : Math.sin(2 * Math.PI * cutoff * (i - center)) / (Math.PI * (i - center))
    const win = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / M)  // Hamming
    h[i] = sinc * win
    sum += h[i]
  }
  for (let i = 0; i <= M; i++) h[i] /= sum

  const outLen = Math.floor((pcm.length - M) / dec)
  const out = new Float32Array(outLen)
  for (let i = 0; i < outLen; i++) {
    let s = 0
    for (let j = 0; j <= M; j++) s += h[j] * (pcm[i * dec + j] || 0)
    out[i] = s
  }
  return { pcm: out, sr: toSr }
}

// ─── Pre-énfasis ─────────────────────────────────────────────────────

function preEmphasis(pcm, alpha = 0.97) {
  const out = new Float32Array(pcm.length)
  out[0] = pcm[0]
  for (let i = 1; i < pcm.length; i++) out[i] = pcm[i] - alpha * pcm[i - 1]
  return out
}

// ─── Autocorrelación ─────────────────────────────────────────────────

function autocorr(frame, p) {
  const r = new Float32Array(p + 1)
  const N = frame.length
  for (let k = 0; k <= p; k++) {
    let s = 0
    for (let i = 0; i < N - k; i++) s += frame[i] * frame[i + k]
    r[k] = s
  }
  return r
}

// ─── Levinson-Durbin ─────────────────────────────────────────────────

function levinsonDurbin(r, p) {
  const a = new Float32Array(p + 1)  // a[0]=1, a[1..p]=coeficientes LPC
  a[0] = 1
  let E = r[0]
  if (E === 0) return a

  for (let m = 1; m <= p; m++) {
    let lambda = 0
    for (let j = 1; j < m; j++) lambda += a[j] * r[m - j]
    const km = -(r[m] + lambda) / E
    const aPrev = a.slice(0, m)
    a[m] = km
    for (let j = 1; j < m; j++) a[j] = aPrev[j] + km * aPrev[m - j]
    E *= (1 - km * km)
    if (E <= 0) break
  }
  return a
}

// ─── Espectro LPC |1/A(e^jω)|² ───────────────────────────────────────

function lpcSpectrum(a, nBins) {
  const p = a.length - 1
  const spec = new Float32Array(nBins)
  for (let bin = 0; bin < nBins; bin++) {
    const omega = Math.PI * bin / nBins
    let re = 0, im = 0
    for (let k = 0; k <= p; k++) {
      re += a[k] * Math.cos(k * omega)
      im -= a[k] * Math.sin(k * omega)
    }
    const mag2 = re * re + im * im
    spec[bin] = mag2 > 0 ? 1 / mag2 : 0
  }
  return spec
}

// ─── Detección de picos → formantes ──────────────────────────────────

function spectralPeaks(spec, sr, nBins, minHz, maxHz, nOut) {
  const minB = Math.floor(minHz * nBins / (sr / 2))
  const maxB = Math.ceil(maxHz  * nBins / (sr / 2))
  const peaks = []
  for (let i = minB + 1; i < maxB - 1 && i < spec.length; i++) {
    if (spec[i] > spec[i - 1] && spec[i] > spec[i + 1]) {
      // interpolación parabólica para submuestra precision
      const denom = spec[i - 1] - 2 * spec[i] + spec[i + 1]
      const delta = denom !== 0 ? 0.5 * (spec[i - 1] - spec[i + 1]) / denom : 0
      const hz    = (i + delta) * (sr / 2) / nBins
      peaks.push({ hz, val: spec[i] })
    }
  }
  peaks.sort((a, b) => a.hz - b.hz)
  return peaks.slice(0, nOut).map(p => Math.round(p.hz))
}

// ─── API pública ─────────────────────────────────────────────────────

export function extractFormants(pcm, sr) {
  // 1. Decimación a 16 kHz
  const { pcm: pcm16, sr: sr16 } = decimateTo16k(pcm, sr)

  // 2. Encontrar zona fonada con RMS (umbral 10 % del máximo)
  const FL = Math.round(sr16 * 0.025)
  const HL = Math.round(sr16 * 0.010)
  const nF = Math.floor((pcm16.length - FL) / HL) + 1
  let maxRms = 0
  const rmsArr = new Float32Array(nF)
  for (let i = 0; i < nF; i++) {
    let s = 0
    for (let j = 0; j < FL; j++) s += (pcm16[i * HL + j] || 0) ** 2
    rmsArr[i] = Math.sqrt(s / FL)
    if (rmsArr[i] > maxRms) maxRms = rmsArr[i]
  }
  if (maxRms === 0) return null

  const thr = maxRms * 0.10
  let fst = -1, lst = -1
  for (let i = 0; i < nF; i++) {
    if (rmsArr[i] > thr) { if (fst < 0) fst = i; lst = i }
  }
  if (fst < 0 || lst - fst < 10) return null

  // Zona estable: 30–80 %
  const len     = lst - fst
  const stabSt  = Math.round(fst + len * 0.30) * HL
  const stabEnd = Math.round(fst + len * 0.80) * HL

  // 3. Pre-énfasis sobre la zona estable
  const seg = preEmphasis(pcm16.subarray(stabSt, stabEnd))

  // 4. LPC frame a frame
  const p    = 14    // orden LPC (estándar para sr=16 kHz)
  const BINS = 512
  const allF1 = [], allF2 = [], allF3 = []

  for (let i = 0; i + FL <= seg.length; i += HL) {
    const raw = seg.subarray(i, i + FL)
    // Ventana Hamming
    const win = new Float32Array(FL)
    for (let j = 0; j < FL; j++)
      win[j] = raw[j] * (0.54 - 0.46 * Math.cos(2 * Math.PI * j / (FL - 1)))

    const r  = autocorr(win, p)
    if (r[0] === 0) continue
    const a  = levinsonDurbin(r, p)
    const sp = lpcSpectrum(a, BINS)
    const fmts = spectralPeaks(sp, sr16, BINS, 200, 3500, 3)

    if (fmts[0]) allF1.push(fmts[0])
    if (fmts[1]) allF2.push(fmts[1])
    if (fmts[2]) allF3.push(fmts[2])
  }

  if (allF1.length < 3 || allF2.length < 3) return null

  const median = arr => {
    const s = [...arr].sort((a, b) => a - b)
    return s[Math.floor(s.length / 2)]
  }

  return {
    F1: median(allF1),
    F2: median(allF2),
    F3: allF3.length >= 3 ? median(allF3) : null,
  }
}
