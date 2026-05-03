import { computeSpectrogram, findOnsetFrame } from './spectrogram'

const FFT_SIZE = 512
const HOP_SIZE = 128

export function analyzeAudio({ channelData, sampleRate, duration }) {
  const spec       = computeSpectrogram(channelData, sampleRate, FFT_SIZE, HOP_SIZE)
  const onsetFrame = findOnsetFrame(spec.frames)
  const onsetSec   = (onsetFrame * HOP_SIZE) / sampleRate

  const onsetSample = onsetFrame * HOP_SIZE
  const trimmed     = channelData.slice(onsetSample)

  // RMS intensity
  let sumSq = 0
  for (let i = 0; i < trimmed.length; i++) sumSq += trimmed[i] ** 2
  const rms   = Math.sqrt(sumSq / Math.max(trimmed.length, 1))
  const rmsDb = 20 * Math.log10(Math.max(rms, 1e-8))

  // Peak amplitude
  let peak = 0
  for (let i = 0; i < trimmed.length; i++) {
    const a = Math.abs(trimmed[i])
    if (a > peak) peak = a
  }
  const peakDb = 20 * Math.log10(Math.max(peak, 1e-8))

  // Intensity variability — coefficient of variation of RMS per 20 ms frame
  const frameLen = Math.floor(0.020 * sampleRate)
  const frameRms = []
  for (let i = 0; i + frameLen <= trimmed.length; i += frameLen) {
    let s = 0
    for (let j = i; j < i + frameLen; j++) s += trimmed[j] ** 2
    frameRms.push(Math.sqrt(s / frameLen))
  }
  const meanFRms = frameRms.length
    ? frameRms.reduce((a, b) => a + b, 0) / frameRms.length
    : 0
  const varFRms = frameRms.length
    ? frameRms.reduce((a, b) => a + (b - meanFRms) ** 2, 0) / frameRms.length
    : 0
  const cvIntensity = meanFRms > 0 ? Math.sqrt(varFRms) / meanFRms : 0

  // F0 via autocorrelation on first 40 ms
  const f0 = estimateF0(trimmed, sampleRate)

  // Spectral centroid from onset-aligned mean spectrum
  const activeFrames = spec.frames.slice(onsetFrame)
  const meanSpec     = new Float32Array(FFT_SIZE / 2)
  if (activeFrames.length > 0) {
    activeFrames.forEach(f => { for (let i = 0; i < meanSpec.length; i++) meanSpec[i] += f[i] })
    for (let i = 0; i < meanSpec.length; i++) meanSpec[i] /= activeFrames.length
  }
  const centroid = spectralCentroid(meanSpec, sampleRate / FFT_SIZE)

  return { duration, onsetSec, rmsDb, peakDb, cvIntensity, f0, centroid, spec, onsetFrame }
}

function estimateF0(data, sampleRate) {
  const minHz  = 70, maxHz = 400
  const winLen = Math.min(Math.floor(0.040 * sampleRate), data.length)
  if (winLen < 10) return null

  const minLag = Math.floor(sampleRate / maxHz)
  const maxLag = Math.min(Math.floor(sampleRate / minHz), winLen - 1)
  if (minLag >= maxLag) return null

  let sumE = 0
  for (let i = 0; i < winLen; i++) sumE += data[i] ** 2
  if (sumE < 1e-10) return null

  let best = -Infinity, bestLag = -1
  for (let lag = minLag; lag <= maxLag; lag++) {
    let s = 0
    for (let i = 0; i + lag < winLen; i++) s += data[i] * data[i + lag]
    if (s > best) { best = s; bestLag = lag }
  }
  return bestLag > 0 ? Math.round(sampleRate / bestLag) : null
}

function spectralCentroid(spectrum, freqPerBin) {
  let weighted = 0, total = 0
  for (let i = 0; i < spectrum.length; i++) {
    weighted += i * freqPerBin * spectrum[i]
    total    += spectrum[i]
  }
  return total > 0 ? Math.round(weighted / total) : 0
}

export function acousticDistance(patSpec, patOnset, refSpec, refOnset) {
  const patFrames = patSpec.frames.slice(patOnset)
  const refFrames = refSpec.frames.slice(refOnset)
  if (!patFrames.length || !refFrames.length) return null

  const len    = patFrames[0].length
  const patMean = new Float32Array(len)
  const refMean = new Float32Array(len)

  patFrames.forEach(f => { for (let i = 0; i < len; i++) patMean[i] += f[i] })
  refFrames.forEach(f => { for (let i = 0; i < len; i++) refMean[i] += f[i] })
  for (let i = 0; i < len; i++) {
    patMean[i] = Math.log(Math.max(patMean[i] / patFrames.length, 1e-8))
    refMean[i] = Math.log(Math.max(refMean[i] / refFrames.length, 1e-8))
  }

  const pm = patMean.reduce((a, b) => a + b, 0) / len
  const rm = refMean.reduce((a, b) => a + b, 0) / len
  let sp = 0, sq = 0, rq = 0
  for (let i = 0; i < len; i++) {
    const p = patMean[i] - pm, r = refMean[i] - rm
    sp += p * r; sq += p * p; rq += r * r
  }
  const corr = sq > 0 && rq > 0 ? sp / Math.sqrt(sq * rq) : 0
  return Math.max(0, Math.min(1, (1 - corr) / 2))
}
