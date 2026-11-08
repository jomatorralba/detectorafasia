// Seguimiento de formantes trama a trama (para análisis en directo).
// Método similar al de Praat (To Formant, Burg/LPC):
//   1. Diezmado a ~12 kHz con filtro FIR antialiasing
//   2. Pre-énfasis desde 50 Hz
//   3. Ventana Hamming de 25 ms
//   4. LPC por autocorrelación (Levinson-Durbin)
//   5. Raíces del polinomio LPC (Durand-Kerner) → frecuencia y ancho de banda
//   6. Se conservan las resonancias con ancho de banda razonable, ordenadas: F1, F2, F3, F4
// Usar raíces (y no picos del espectro) evita que F1 y F2 se fundan cuando están
// muy próximos, como en /o/ y /u/.

export const VOICES = {
  hombre: { label: 'Hombre', ceiling: 5000 },
  mujer:  { label: 'Mujer',  ceiling: 5500 },
  nino:   { label: 'Niño/a', ceiling: 8000 },
}

function lowpassFir(cutoffNorm, taps) {
  const h = new Float64Array(taps)
  const c = (taps - 1) / 2
  let s = 0
  for (let i = 0; i < taps; i++) {
    const x = i - c
    const sinc = x === 0 ? 2 * cutoffNorm : Math.sin(2 * Math.PI * cutoffNorm * x) / (Math.PI * x)
    h[i] = sinc * (0.42 - 0.5 * Math.cos(2 * Math.PI * i / (taps - 1)) + 0.08 * Math.cos(4 * Math.PI * i / (taps - 1))) // Blackman
    s += h[i]
  }
  for (let i = 0; i < taps; i++) h[i] /= s
  return h
}

function levinson(r, p) {
  const a = new Float64Array(p + 1); a[0] = 1
  let E = r[0]
  if (E <= 0) return null
  const tmp = new Float64Array(p + 1)
  for (let m = 1; m <= p; m++) {
    let acc = r[m]
    for (let j = 1; j < m; j++) acc += a[j] * r[m - j]
    const k = -acc / E
    tmp.set(a)
    for (let j = 1; j < m; j++) a[j] = tmp[j] + k * tmp[m - j]
    a[m] = k
    E *= 1 - k * k
    if (E <= 0) return null
  }
  return a
}

// Raíces de z^p + a1 z^(p-1) + ... + ap (Durand-Kerner)
function polyRoots(a) {
  const p = a.length - 1
  const re = new Float64Array(p), im = new Float64Array(p)
  for (let i = 0; i < p; i++) {          // arranque en círculo de radio 0.9
    const ang = 2 * Math.PI * i / p + 0.4
    re[i] = 0.9 * Math.cos(ang); im[i] = 0.9 * Math.sin(ang)
  }
  for (let it = 0; it < 200; it++) {
    let delta = 0
    for (let i = 0; i < p; i++) {
      // valor del polinomio en z_i (Horner)
      let vr = 1, vi = 0
      for (let k = 1; k <= p; k++) {
        const nr = vr * re[i] - vi * im[i] + a[k]
        vi = vr * im[i] + vi * re[i]
        vr = nr
      }
      // producto (z_i - z_j)
      let dr = 1, di = 0
      for (let j = 0; j < p; j++) {
        if (j === i) continue
        const xr = re[i] - re[j], xi = im[i] - im[j]
        const nr = dr * xr - di * xi
        di = dr * xi + di * xr
        dr = nr
      }
      const den = dr * dr + di * di
      if (den === 0) continue
      const qr = (vr * dr + vi * di) / den, qi = (vi * dr - vr * di) / den
      re[i] -= qr; im[i] -= qi
      delta = Math.max(delta, Math.abs(qr) + Math.abs(qi))
    }
    if (delta < 1e-10) break
  }
  return { re, im }
}

export function createFormantTracker(sampleRate, ceiling = 5500, { winMs = 25, nFormants = 5 } = {}) {
  // Remuestreo (fraccionario) a 2 × techo, como Praat: así el orden LPC = 2 × nº de formantes
  const fs     = Math.min(sampleRate, 2 * ceiling)
  const ratio  = sampleRate / fs                         // muestras de entrada por muestra de salida
  const fc     = 0.5 / ratio * 0.95                      // corte (ciclos/muestra de entrada)
  const half   = Math.ceil(6 / (2 * fc))                 // semianchura del núcleo sinc
  const nOut   = Math.round(fs * winMs / 1000)
  const order  = 2 * nFormants
  const alpha  = Math.exp(-2 * Math.PI * 50 / fs)
  const ham    = Float64Array.from({ length: nOut }, (_, i) => 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (nOut - 1)))
  const need   = Math.ceil((nOut + 1) * ratio) + 2 * half + 2   // muestras originales necesarias

  const kernel = x => {
    if (Math.abs(x) >= half) return 0
    const s = x === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * x) / (Math.PI * x)
    return s * (0.5 + 0.5 * Math.cos(Math.PI * x / half))   // ventana Hann
  }

  /** frame: Float32Array de `need` muestras a la frecuencia original (la última es la más reciente) */
  function analyze(frame) {
    // remuestreo: x[n] en el instante t = base + n·ratio (en muestras de entrada)
    const x = new Float64Array(nOut + 1)
    const base = frame.length - 1 - half - nOut * ratio
    for (let n = 0; n <= nOut; n++) {
      if (ratio === 1) { x[n] = frame[Math.round(base + n)] ?? 0; continue }
      const t = base + n * ratio, k0 = Math.ceil(t - half), k1 = Math.floor(t + half)
      let s = 0
      for (let k = k0; k <= k1; k++) s += (frame[k] ?? 0) * kernel(t - k)
      x[n] = s
    }
    // pre-énfasis + ventana
    const w = new Float64Array(nOut)
    for (let n = 0; n < nOut; n++) w[n] = (x[n + 1] - alpha * x[n]) * ham[n]
    // autocorrelación
    const r = new Float64Array(order + 1)
    for (let k = 0; k <= order; k++) { let s = 0; for (let n = k; n < nOut; n++) s += w[n] * w[n - k]; r[k] = s }
    if (r[0] <= 1e-12) return []
    r[0] *= 1 + 1e-9                                     // estabilidad numérica
    const a = levinson(r, order)
    if (!a) return []
    const { re, im } = polyRoots(a)
    const out = []
    for (let i = 0; i < re.length; i++) {
      if (im[i] <= 0) continue                          // una de cada par conjugado
      const mag = Math.hypot(re[i], im[i])
      const f   = Math.atan2(im[i], re[i]) * fs / (2 * Math.PI)
      const bw  = -Math.log(Math.max(mag, 1e-9)) * fs / Math.PI
      if (f > 90 && f < ceiling - 50 && bw > 0 && bw < 600) out.push({ f, bw })
    }
    out.sort((p, q) => p.f - q.f)
    return out.slice(0, 4)
  }

  return { need, fs, order, analyze }
}
