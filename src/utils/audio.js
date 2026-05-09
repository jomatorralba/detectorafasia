export async function decodeAudio(blob) {
  const ac = new AudioContext()
  const buf = await ac.decodeAudioData(await blob.arrayBuffer())
  ac.close()
  return {
    channelData: buf.getChannelData(0),
    sampleRate: buf.sampleRate,
    duration: buf.duration,
  }
}

export function calcTMF(channelData, sampleRate) {
  const fl = Math.floor(0.025 * sampleRate)
  const hl = Math.floor(0.010 * sampleRate)
  const n  = Math.max(1, Math.floor((channelData.length - fl) / hl) + 1)

  let maxRms = 0
  const rms = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let j = 0; j < fl; j++) s += (channelData[i * hl + j] || 0) ** 2
    rms[i] = Math.sqrt(s / fl)
    if (rms[i] > maxRms) maxRms = rms[i]
  }

  if (maxRms === 0) return 0
  const thr = maxRms * 0.05
  let first = -1, last = -1
  for (let i = 0; i < n; i++) {
    if (rms[i] > thr) { if (first < 0) first = i; last = i }
  }
  return first < 0 ? 0 : (last - first) * hl / sampleRate
}

export function calcDDK(channelData, sampleRate) {
  const fl = Math.floor(0.020 * sampleRate)
  const hl = Math.floor(0.010 * sampleRate)
  const n  = Math.max(1, Math.floor((channelData.length - fl) / hl) + 1)

  const energy = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let j = 0; j < fl; j++) s += (channelData[i * hl + j] || 0) ** 2
    energy[i] = s / fl
  }

  // Onset strength = positive energy derivative
  const str = new Float32Array(n)
  let maxStr = 0
  for (let i = 1; i < n; i++) {
    str[i] = Math.max(0, energy[i] - energy[i - 1])
    if (str[i] > maxStr) maxStr = str[i]
  }

  if (maxStr === 0) return { nPataka: 0, onsetTimes: [] }

  const thr     = maxStr * 0.30
  const minDist = Math.floor(0.08 * sampleRate / hl) // 80 ms min between onsets

  const onsets = []
  for (let i = 1; i < n - 1; i++) {
    if (
      str[i] > thr &&
      str[i] >= str[i - 1] &&
      str[i] >= str[i + 1] &&
      (!onsets.length || i - onsets.at(-1) > minDist)
    ) onsets.push(i)
  }

  return {
    nPataka: onsets.length / 3,
    onsetTimes: onsets.map(i => i * hl / sampleRate),
  }
}

export function getEnvelope(channelData, n = 200) {
  const chunk = Math.max(1, Math.floor(channelData.length / n))
  const env = new Float32Array(n)
  const maxVal = Math.max(...channelData.map(Math.abs)) || 1
  for (let i = 0; i < n; i++) {
    let max = 0
    for (let j = 0; j < chunk; j++) {
      const abs = Math.abs(channelData[i * chunk + j] || 0)
      if (abs > max) max = abs
    }
    env[i] = max / maxVal
  }
  return env
}

export function gaussianRef(refDuration, recordedDuration, n = 200) {
  const tMax = Math.max(recordedDuration * 1.1, refDuration * 1.4)
  const mu   = refDuration / 2
  const sig  = refDuration / 4
  const ref  = new Float32Array(n)
  const times = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    times[i] = (i / n) * tMax
    ref[i]   = Math.exp(-0.5 * ((times[i] - mu) / sig) ** 2) * 0.85
  }
  return { ref, times, tMax }
}

// Análisis de pausas en lectura de texto continuo
// Ref: Tjaden & Wilding (2004); Duffy (2013) Motor Speech Disorders
export function calcPauseStats(channelData, sampleRate, totalSyllables) {
  const FL = Math.floor(0.025 * sampleRate)   // 25 ms
  const HL = Math.floor(0.010 * sampleRate)   // 10 ms hop
  const n  = Math.max(1, Math.floor((channelData.length - FL) / HL) + 1)

  let maxRms = 0
  const rms = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let j = 0; j < FL; j++) s += (channelData[i * HL + j] || 0) ** 2
    rms[i] = Math.sqrt(s / FL)
    if (rms[i] > maxRms) maxRms = rms[i]
  }
  if (maxRms === 0) return null

  const thr = maxRms * 0.05
  const voiced = Array.from(rms, v => v > thr ? 1 : 0)

  // Agrupar runs de silencio ≥ 250 ms (excluyendo inicial y final)
  const pauseMinFrames = Math.round(0.250 / 0.010)
  const pauses = []
  let i = 1  // saltar primer frame
  while (i < voiced.length - 1) {
    if (voiced[i] === 0) {
      let j = i
      while (j < voiced.length - 1 && voiced[j] === 0) j++
      if (j - i >= pauseMinFrames) pauses.push((j - i) * HL / sampleRate)
      i = j
    } else {
      i++
    }
  }

  const totalDur    = channelData.length / sampleRate
  const silenceDur  = voiced.filter(v => v === 0).length * HL / sampleRate
  const speechDur   = Math.max(0.1, totalDur - silenceDur)

  return {
    silence_pct:       +( silenceDur / totalDur ).toFixed(3),
    num_pauses:        pauses.length,
    mean_pause_dur:    pauses.length > 0 ? +(pauses.reduce((a, b) => a + b, 0) / pauses.length).toFixed(2) : 0,
    articulation_rate: +(totalSyllables / speechDur).toFixed(1),
  }
}

export function getWaveformPoints(channelData, sampleRate, maxPoints = 3000) {
  const step = Math.max(1, Math.floor(channelData.length / maxPoints))
  const dur  = channelData.length / sampleRate
  const pts  = []
  for (let i = 0; i < channelData.length; i += step)
    pts.push({ t: (i / channelData.length) * dur, v: channelData[i] })
  return pts
}
