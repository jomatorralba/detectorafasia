// Motor de análisis en directo: recibe muestras PCM en trozos de cualquier tamaño
// y produce columnas a intervalos fijos (hop) con:
//   - espectro en dBFS (STFT, ventana Hann)
//   - mínimo / máximo de la forma de onda en ese intervalo (oscilograma)
//   - RMS (dBFS) y F0 (YIN)
// El resultado no depende de cómo lleguen troceadas las muestras.

import { hanning, rfft } from './spectrogram.js'
import { yinPitch } from './yin.js'

export const MODES = {
  wide:   { label: 'Banda ancha',   winMs: 5,  desc: 'Formantes y transiciones (ventana 5 ms)' },
  narrow: { label: 'Banda estrecha', winMs: 30, desc: 'Armónicos y F0 (ventana 30 ms)' },
}

const nextPow2 = n => 2 ** Math.ceil(Math.log2(n))

export class LiveAnalyzer {
  constructor(sampleRate, { mode = 'wide', hopMs = 5, f0EveryCols = 4, f0WinMs = 40 } = {}) {
    this.sr       = sampleRate
    this.hop      = Math.round(sampleRate * hopMs / 1000)
    this.winLen   = Math.round(sampleRate * MODES[mode].winMs / 1000)
    this.nfft     = Math.max(1024, nextPow2(this.winLen))
    this.win      = hanning(this.winLen)
    this.winSum   = this.win.reduce((a, b) => a + b, 0)
    this.binHz    = sampleRate / this.nfft
    this.f0Len    = Math.round(sampleRate * f0WinMs / 1000)
    this.f0Every  = f0EveryCols

    // buffer circular: cubre la ventana más larga que necesitamos
    this.cap   = nextPow2(Math.max(this.winLen, this.f0Len, this.hop) * 2 + sampleRate * 0.1)
    this.buf   = new Float32Array(this.cap)
    this.total = 0             // muestras recibidas
    this.next  = this.hop      // final (exclusivo) de la próxima columna
    this.cols  = 0
    this.f0    = 0
    this.frame = new Float32Array(this.nfft)
  }

  _read(end, len, out) {
    // copia las `len` muestras que terminan en `end` (posición absoluta)
    const start = end - len
    for (let i = 0; i < len; i++) {
      const p = start + i
      out[i] = p < 0 ? 0 : this.buf[p & (this.cap - 1)]
    }
    return out
  }

  /** Últimas n muestras (para el osciloscopio de detalle). */
  recent(n) {
    n = Math.min(n, this.total, this.cap)
    return this._read(this.total, n, new Float32Array(n))
  }

  /** Añade muestras; devuelve las columnas completas generadas. */
  push(samples) {
    const out = []
    let i = 0
    while (i < samples.length) {
      // copiar hasta completar la siguiente columna
      const need = Math.min(samples.length - i, this.next - this.total)
      for (let k = 0; k < need; k++) this.buf[(this.total + k) & (this.cap - 1)] = samples[i + k]
      this.total += need
      i += need
      if (this.total === this.next) {
        out.push(this._column(this.next))
        this.next += this.hop
      }
    }
    return out
  }

  _column(end) {
    // forma de onda del intervalo
    let mn = Infinity, mx = -Infinity, ss = 0
    for (let p = end - this.hop; p < end; p++) {
      const v = p < 0 ? 0 : this.buf[p & (this.cap - 1)]
      if (v < mn) mn = v
      if (v > mx) mx = v
      ss += v * v
    }
    const rms   = Math.sqrt(ss / this.hop)
    const rmsDb = 20 * Math.log10(Math.max(rms, 1e-7))

    // espectro (ventana terminada en `end`, zero-padding hasta nfft)
    const f = this.frame
    f.fill(0)
    for (let k = 0; k < this.winLen; k++) {
      const p = end - this.winLen + k
      f[k] = (p < 0 ? 0 : this.buf[p & (this.cap - 1)]) * this.win[k]
    }
    const mag = rfft(f)
    const db  = new Float32Array(mag.length)
    const norm = 2 / this.winSum
    for (let b = 0; b < mag.length; b++) db[b] = 20 * Math.log10(Math.max(mag[b] * norm, 1e-7))

    // F0 cada f0Every columnas, sólo si hay señal suficiente
    if (this.cols % this.f0Every === 0) {
      const fr = this._read(end, this.f0Len, new Float32Array(this.f0Len))
      let e = 0
      for (let k = 0; k < fr.length; k++) e += fr[k] * fr[k]
      this.f0 = Math.sqrt(e / fr.length) > 0.003 ? yinPitch(fr, this.sr) : 0
    }

    this.cols++
    return { t: end / this.sr, min: mn, max: mx, rmsDb, db, f0: this.f0 }
  }
}
