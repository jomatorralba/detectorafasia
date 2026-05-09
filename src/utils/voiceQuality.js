// Calidad vocal: jitter, shimmer y HNR sobre vocal sostenida /a/
//
// Referencias:
//   Titze (1995) Workshop on Acoustic Voice Analysis — normas jitter/shimmer
//   Boersma (1993) JIPA 23:101-109 — HNR por autocorrelación (algoritmo de Praat)

import { yinPitch } from './yin.js'

export function calcVoiceQuality(pcm, sr) {
  // Framing: frames 40ms / hop 5ms (resolución fina para perturbación)
  const FN = Math.round(sr * 0.040)
  const HN = Math.round(sr * 0.005)
  const n  = Math.floor((pcm.length - FN) / HN) + 1

  // Encontrar zona fonada (RMS > 5% del máximo)
  let maxRms = 0
  const rms = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let j = 0; j < FN; j++) s += (pcm[i * HN + j] || 0) ** 2
    rms[i] = Math.sqrt(s / FN)
    if (rms[i] > maxRms) maxRms = rms[i]
  }
  if (maxRms === 0) return null

  const thr = maxRms * 0.05
  let first = -1, last = -1
  for (let i = 0; i < n; i++) {
    if (rms[i] > thr) { if (first < 0) first = i; last = i }
  }
  if (first < 0 || last - first < 20) return null

  // Zona estable: 20–80 % de la región fonada
  const len = last - first
  const stableStart = Math.round(first + len * 0.20)
  const stableEnd   = Math.round(first + len * 0.80)

  // Extraer F0 y amplitud por frame en la zona estable
  const f0s = [], amps = []
  for (let i = stableStart; i < stableEnd; i++) {
    const start = i * HN
    if (start + FN > pcm.length) break
    const frame = pcm.subarray(start, start + FN)
    const f0 = yinPitch(frame, sr)
    if (f0 > 60 && f0 < 600) {
      f0s.push(f0)
      let s = 0
      for (let k = 0; k < FN; k++) s += frame[k] ** 2
      amps.push(Math.sqrt(s / FN))
    }
  }
  if (f0s.length < 5) return null

  // ── Jitter local (%) ─────────────────────────────────────────────
  // Perturbación ciclo-a-ciclo del periodo: mean|T_i − T_{i+1}| / mean(T_i)
  const periods = f0s.map(f => 1 / f)
  let jitterSum = 0
  for (let i = 0; i < periods.length - 1; i++) jitterSum += Math.abs(periods[i] - periods[i + 1])
  const meanPeriod = periods.reduce((a, b) => a + b, 0) / periods.length
  const jitter = (jitterSum / (periods.length - 1)) / meanPeriod * 100

  // ── Shimmer local (%) ────────────────────────────────────────────
  // Perturbación ciclo-a-ciclo de la amplitud RMS
  let shimmerSum = 0
  for (let i = 0; i < amps.length - 1; i++) shimmerSum += Math.abs(amps[i] - amps[i + 1])
  const meanAmp = amps.reduce((a, b) => a + b, 0) / amps.length
  const shimmer = meanAmp > 0 ? (shimmerSum / (amps.length - 1)) / meanAmp * 100 : 0

  // ── HNR (dB) — método Boersma (1993) ─────────────────────────────
  // Autocorrelación normalizada en el lag T0; HNR = 10·log10(r(T0)/(1−r(T0)))
  const meanF0 = f0s.reduce((a, b) => a + b, 0) / f0s.length
  const T0     = Math.round(sr / meanF0)

  const s0 = stableStart * HN
  const s1 = Math.min(stableEnd * HN, pcm.length)
  let r0 = 0, rT = 0
  for (let i = s0; i < s1 - T0; i++) {
    r0 += pcm[i] * pcm[i]
    rT += pcm[i] * pcm[i + T0]
  }
  const rNorm    = r0 > 0 ? rT / r0 : 0
  const rClamped = Math.max(0.0001, Math.min(0.9999, rNorm))
  const hnr      = 10 * Math.log10(rClamped / (1 - rClamped))

  return {
    jitter:  +jitter.toFixed(2),
    shimmer: +shimmer.toFixed(2),
    hnr:     +hnr.toFixed(1),
  }
}
