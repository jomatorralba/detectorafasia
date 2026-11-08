import { useState, useRef, useEffect, useCallback } from 'react'
import { Mic, Square, Pause, Play, Download, AlertTriangle, SlidersHorizontal } from 'lucide-react'
import { LiveAnalyzer, MODES } from '../utils/liveAnalysis.js'
import { VOICES } from '../utils/formantTrack.js'
import { magmaColor } from '../utils/spectrogram.js'

// ─── Parámetros de visualización ──────────────────────────────────────
const HOP_MS    = 5                          // una columna cada 5 ms
const WINDOW_S  = 6                          // segundos visibles
const COLS      = (WINDOW_S * 1000) / HOP_MS // 1200 columnas
const SPEC_H    = 320                        // filas del espectrograma
const OSC_H     = 120
const SCOPE_MS  = 40                         // ventana del osciloscopio de detalle
const FREQS     = [4000, 5000, 8000]
const GAINS     = [1, 2, 5, 10, 20]
const F0_MIN    = 75                         // escala de la línea azul (Hz), como Praat
const F0_MAX    = 500
const INT_MIN   = -70                        // escala de la línea verde (dBFS)
const NF        = 4                          // formantes guardados por columna

// Capas superpuestas (mismos colores que Praat)
const LAYERS = {
  f0:  { label: 'F0 (tono)',   color: '#2563eb' },
  fmt: { label: 'Formantes',   color: '#e11d48' },
  int: { label: 'Intensidad',  color: '#16a34a' },
}

// Temas: grises tipo Praat (por defecto) o color
const THEMES = {
  gris: {
    label: 'Grises (tipo Praat)',
    lut: Array.from({ length: 256 }, (_, i) => { const v = Math.round(255 * (1 - i / 255) ** 1.15); return [v, v, v] }),
    range: 50, specBg: '#ffffff', oscBg: '#ffffff', trace: '#0e6170', zero: 'rgba(0,0,0,0.12)',
    grid: 'rgba(0,0,0,0.08)', axis: 'rgba(0,0,0,0.55)', border: '1px solid #e5e7eb', hint: 'rgba(0,0,0,0.45)',
  },
  color: {
    label: 'Color',
    lut: Array.from({ length: 256 }, (_, i) => magmaColor(i / 255)),
    range: 70, specBg: '#000004', oscBg: '#070a0f', trace: 'rgba(251,146,60,0.95)', zero: 'rgba(255,255,255,0.08)',
    grid: 'rgba(255,255,255,0.07)', axis: 'rgba(255,255,255,0.55)', border: 'none', hint: 'rgba(255,255,255,0.55)',
  },
}

// Procesador de audio: envía bloques de 512 muestras al hilo principal
const WORKLET = `
class LiveTap extends AudioWorkletProcessor {
  constructor () { super(); this.b = new Float32Array(512); this.n = 0 }
  process (inputs) {
    const ch = inputs[0] && inputs[0][0]
    if (ch) for (let i = 0; i < ch.length; i++) {
      this.b[this.n++] = ch[i]
      if (this.n === 512) { this.port.postMessage(this.b.slice(0)); this.n = 0 }
    }
    return true
  }
}
registerProcessor('live-tap', LiveTap)`

// posición vertical (0 = arriba, 1 = abajo) de cada escala propia
const f0Y  = v => 1 - (Math.min(Math.max(v, F0_MIN), F0_MAX) - F0_MIN) / (F0_MAX - F0_MIN)
const intY = v => 1 - (Math.min(v, 0) - INT_MIN) / -INT_MIN

const newHistory = () => ({ f0: new Float32Array(COLS), fmt: new Float32Array(COLS * NF), int: new Float32Array(COLS).fill(-200), head: 0 })

// ─── UI auxiliar ──────────────────────────────────────────────────────
function Seg({ options, value, onChange }) {
  return (
    <div style={{ display: 'inline-flex', border: '1px solid #d1d5db', borderRadius: 8, overflow: 'hidden' }}>
      {options.map(o => (
        <button key={o.value} onClick={() => onChange(o.value)}
          style={{
            padding: '5px 10px', fontSize: 12, border: 'none', cursor: 'pointer',
            background: value === o.value ? '#116b70' : '#fff',
            color: value === o.value ? '#fff' : '#555',
            borderRight: '1px solid #e5e7eb',
          }}>{o.label}</button>
      ))}
    </div>
  )
}

const label   = { fontSize: 11, color: '#888', marginRight: 6 }
const axisTxt = { position: 'absolute', fontSize: 9, fontFamily: 'monospace', pointerEvents: 'none' }

// ─── Página ───────────────────────────────────────────────────────────
export function EnDirecto() {
  const [running, setRunning] = useState(false)
  const [frozen,  setFrozen]  = useState(false)
  const [mode,    setMode]    = useState('wide')
  const [maxFreq, setMaxFreq] = useState(5000)
  const [gain,    setGain]    = useState(2)
  const [voice,   setVoice]   = useState('mujer')
  const [theme,   setTheme]   = useState('gris')
  const [layers,  setLayers]  = useState({ f0: true, fmt: true, int: true })
  const [read,    setRead]    = useState({ f0: 0, db: -Infinity, fmts: null, clip: false })
  const [error,   setError]   = useState('')
  const [sr,      setSr]      = useState(0)
  const [showCfg, setShowCfg] = useState(false)
  const [showDetail, setShowDetail] = useState(false)

  const specRef  = useRef(null)
  const ovRef    = useRef(null)
  const oscRef   = useRef(null)
  const scopeRef = useRef(null)
  const audio    = useRef(null)    // { ctx, stream, node, sink }
  const queue    = useRef([])
  const anRef    = useRef(null)
  const rafRef   = useRef(0)
  const histRef  = useRef(newHistory())
  const settings = useRef({})
  settings.current = { frozen, maxFreq, gain, layers, th: THEMES[theme] }
  const peakDb   = useRef(-40)
  const lastClip = useRef(-1)
  const lastRead = useRef(0)
  const recent   = useRef([])      // últimas columnas (200 ms) para lecturas numéricas

  const th = THEMES[theme]

  // ─── Limpieza de lienzos ────────────────────────────────────────────
  const clearAll = useCallback(() => {
    const t = settings.current.th
    const fill = (c, bg) => { if (!c) return; const g = c.getContext('2d'); g.fillStyle = bg; g.fillRect(0, 0, c.width, c.height) }
    fill(specRef.current, t.specBg)
    fill(oscRef.current, t.oscBg)
    fill(scopeRef.current, t.oscBg)
    const o = oscRef.current
    if (o) { const g = o.getContext('2d'); g.fillStyle = t.zero; g.fillRect(0, OSC_H / 2, o.width, 1) }
    histRef.current = newHistory()
    drawOverlay()
  }, [])

  const newAnalyzer = (rate) => new LiveAnalyzer(rate, { mode, hopMs: HOP_MS, ceiling: VOICES[voice].ceiling })

  // Cambios de análisis, voz, escala o tema: analizador nuevo y lienzos limpios
  useEffect(() => {
    if (audio.current) anRef.current = newAnalyzer(audio.current.ctx.sampleRate)
    clearAll()
  }, [mode, voice, maxFreq, theme, clearAll]) // eslint-disable-line

  // Mostrar/ocultar capas: redibujar al momento (también con la imagen congelada)
  useEffect(() => { drawOverlay() }, [layers]) // eslint-disable-line

  // ─── Dibujo de columnas (espectrograma + oscilograma) ───────────────
  const drawColumns = (cols, an) => {
    const { maxFreq: mf, gain: gn, th: t } = settings.current
    const n = cols.length
    const spec = specRef.current, osc = oscRef.current
    const gs = spec.getContext('2d'), go = osc.getContext('2d')

    gs.drawImage(spec, -n, 0)
    go.drawImage(osc, -n, 0)
    go.fillStyle = t.oscBg; go.fillRect(COLS - n, 0, n, OSC_H)
    go.fillStyle = t.zero;  go.fillRect(COLS - n, OSC_H / 2, n, 1)

    const img = gs.createImageData(n, SPEC_H)
    const nyq = an.sr / 2
    for (let k = 0; k < n; k++) {
      const c = cols[k]
      let cmax = -200
      const top = Math.min(c.db.length - 1, Math.floor(Math.min(mf, nyq) / an.binHz))
      for (let b = 2; b <= top; b++) if (c.db[b] > cmax) cmax = c.db[b]
      peakDb.current = cmax > peakDb.current ? cmax : Math.max(-60, peakDb.current - 0.02)
      const floor = peakDb.current - t.range

      for (let y = 0; y < SPEC_H; y++) {
        const f = ((SPEC_H - 1 - y) + 0.5) / SPEC_H * mf
        const idx = (y * n + k) * 4
        let v = 0
        if (f < nyq) {
          const pos = f / an.binHz, b0 = Math.floor(pos), fr = pos - b0
          const d = c.db[b0] * (1 - fr) + (c.db[b0 + 1] ?? c.db[b0]) * fr
          v = Math.max(0, Math.min(255, Math.round((d - floor) / t.range * 255)))
        }
        const [r, g, bl] = t.lut[v]
        img.data[idx] = r; img.data[idx + 1] = g; img.data[idx + 2] = bl; img.data[idx + 3] = 255
      }

      const clip = c.max >= 0.99 || c.min <= -0.99
      if (clip) lastClip.current = c.t
      const y1 = (0.5 - Math.min(1, c.max * gn) * 0.46) * OSC_H
      const y2 = (0.5 - Math.max(-1, c.min * gn) * 0.46) * OSC_H
      go.fillStyle = clip ? '#ef4444' : t.trace
      go.fillRect(COLS - n + k, Math.floor(y1), 1, Math.max(1, Math.ceil(y2 - y1)))
    }
    gs.putImageData(img, COLS - n, 0)
  }

  // ─── Capas: F0 (azul), formantes (rojo), intensidad (verde) ─────────
  const pushHistory = cols => {
    const h = histRef.current
    for (const c of cols) {
      const i = h.head
      h.f0[i]  = c.f0 || 0
      h.int[i] = c.intDb
      for (let k = 0; k < NF; k++) h.fmt[i * NF + k] = c.formants?.[k] || 0
      h.head = (i + 1) % COLS
    }
  }

  function drawOverlay() {
    const c = ovRef.current; if (!c) return
    const { maxFreq: mf, layers: L } = settings.current
    const g = c.getContext('2d'), h = histRef.current
    g.clearRect(0, 0, COLS, SPEC_H)
    const at = x => (h.head + x) % COLS          // x = 0 (más antiguo) … COLS-1 (ahora)

    if (L.fmt) {
      g.fillStyle = LAYERS.fmt.color
      for (let x = 0; x < COLS; x += 2) {         // un punto cada 10 ms, como Praat
        const i = at(x)
        for (let k = 0; k < NF; k++) {
          const f = h.fmt[i * NF + k]
          if (f > 0 && f < mf) g.fillRect(x - 1.5, SPEC_H - f / mf * SPEC_H - 1.5, 3, 3)
        }
      }
    }
    const line = (val, toY, color, width) => {
      g.strokeStyle = color; g.lineWidth = width; g.lineJoin = 'round'; g.beginPath()
      let pen = false
      for (let x = 0; x < COLS; x++) {
        const y = toY(val(at(x)))
        if (y == null) { pen = false; continue }
        pen ? g.lineTo(x, y) : g.moveTo(x, y)
        pen = true
      }
      g.stroke()
    }
    if (L.int) line(i => h.int[i], v => v > INT_MIN ? intY(v) * SPEC_H : null, LAYERS.int.color, 2)
    if (L.f0)  line(i => h.f0[i],  v => v > 0 ? f0Y(v) * SPEC_H : null, LAYERS.f0.color, 2.5)
  }

  const drawScope = an => {
    const c = scopeRef.current; if (!c || !showDetailRef.current) return
    const t = settings.current.th
    const g = c.getContext('2d'), w = c.width, h = c.height
    const span = Math.round(an.sr * SCOPE_MS / 1000)
    const raw  = an.recent(span * 2)
    if (raw.length < span * 2) return
    let start = 0
    for (let i = span; i > 1; i--) if (raw[i - 1] < 0 && raw[i] >= 0) { start = i; break }
    const seg = raw.subarray(start, start + span)
    let pk = 0.02; for (let i = 0; i < seg.length; i++) pk = Math.max(pk, Math.abs(seg[i]))
    g.fillStyle = t.oscBg; g.fillRect(0, 0, w, h)
    g.fillStyle = t.zero; g.fillRect(0, h / 2, w, 1)
    for (let ms = 10; ms < SCOPE_MS; ms += 10) g.fillRect(Math.round(ms / SCOPE_MS * w), 0, 1, h)
    g.strokeStyle = t.trace; g.lineWidth = 1.5; g.beginPath()
    for (let i = 0; i < seg.length; i++) {
      const x = i / (span - 1) * w, y = (0.5 - seg[i] / pk * 0.44) * h
      i ? g.lineTo(x, y) : g.moveTo(x, y)
    }
    g.stroke()
  }
  const showDetailRef = useRef(false)
  showDetailRef.current = showDetail

  // ─── Bucle de animación ─────────────────────────────────────────────
  const loop = () => {
    rafRef.current = requestAnimationFrame(loop)
    const an = anRef.current; if (!an) return
    const chunks = queue.current; queue.current = []
    let cols = []
    for (const ch of chunks) cols.push(...an.push(ch))
    if (!cols.length || settings.current.frozen) return
    if (cols.length > COLS) cols = cols.slice(-COLS)   // pestaña oculta: sólo lo visible

    const rec = recent.current
    rec.push(...cols)
    if (rec.length > 40) rec.splice(0, rec.length - 40)  // últimos 200 ms

    drawColumns(cols, an)
    pushHistory(cols)
    drawOverlay()
    drawScope(an)

    const now = performance.now()
    if (now - lastRead.current < 100) return             // lecturas numéricas a ~10 Hz
    lastRead.current = now
    const med = arr => { const s = arr.filter(v => v > 0).sort((a, b) => a - b); return s.length >= 10 ? s[s.length >> 1] : 0 }
    const voiced = rec.filter(c => c.formants)
    const last = cols[cols.length - 1]
    setRead({
      f0:   med(rec.map(c => c.f0)),
      fmts: voiced.length >= 10 ? [0, 1, 2].map(k => med(voiced.map(c => c.formants[k] || 0))) : null,
      db:   last.intDb,
      clip: last.t - lastClip.current < 1,
    })
  }

  // ─── Micrófono ──────────────────────────────────────────────────────
  const start = async () => {
    setError('')
    try {
      // Sin cancelación de eco, supresión de ruido ni control de ganancia:
      // alteran el espectro y la envolvente de la señal.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
      })
      const ctx = new AudioContext()
      await ctx.resume()
      const src  = ctx.createMediaStreamSource(stream)
      const sink = ctx.createGain(); sink.gain.value = 0
      let node
      if (ctx.audioWorklet) {
        const url = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }))
        await ctx.audioWorklet.addModule(url)
        URL.revokeObjectURL(url)
        node = new AudioWorkletNode(ctx, 'live-tap', { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1, channelCountMode: 'explicit' })
        node.port.onmessage = e => queue.current.push(e.data)
      } else {
        node = ctx.createScriptProcessor(1024, 1, 1)
        node.onaudioprocess = e => queue.current.push(e.inputBuffer.getChannelData(0).slice(0))
      }
      src.connect(node); node.connect(sink); sink.connect(ctx.destination)

      audio.current = { ctx, stream, node, sink }
      anRef.current = newAnalyzer(ctx.sampleRate)
      queue.current = []
      recent.current = []
      peakDb.current = -40
      setSr(ctx.sampleRate)
      clearAll()
      setFrozen(false)
      setRunning(true)
      cancelAnimationFrame(rafRef.current)
      rafRef.current = requestAnimationFrame(loop)
    } catch (e) {
      console.error(e)
      setError('No se puede acceder al micrófono. Comprueba los permisos del navegador.')
      stop()
    }
  }

  const stop = () => {
    cancelAnimationFrame(rafRef.current)
    const a = audio.current
    if (a) {
      a.stream.getTracks().forEach(t => t.stop())
      try { a.node.disconnect() } catch { /* ya desconectado */ }
      a.ctx.close()
    }
    audio.current = null
    anRef.current = null
    queue.current = []
    setRunning(false)
    setFrozen(false)
  }

  useEffect(() => { clearAll(); return () => stop() }, []) // eslint-disable-line

  // ─── Exportar imagen ────────────────────────────────────────────────
  const savePng = () => {
    const pad = 48, padR = 60, W = COLS + pad + padR, H = 62 + SPEC_H + 24 + OSC_H + 30
    const c = document.createElement('canvas'); c.width = W; c.height = H
    const g = c.getContext('2d')
    g.fillStyle = '#fff'; g.fillRect(0, 0, W, H)
    g.fillStyle = '#111'; g.font = '600 15px system-ui'; g.textAlign = 'left'
    g.fillText('Espectrograma y oscilograma', pad, 22)
    g.fillStyle = '#777'; g.font = '12px system-ui'
    g.fillText(`${new Date().toLocaleString('es-ES')} · ${MODES[mode].label} · 0–${maxFreq / 1000} kHz · voz: ${VOICES[voice].label.toLowerCase()} · oscilograma ×${gain} · ${sr} Hz`, pad, 38)
    // leyenda
    let lx = pad
    g.font = '12px system-ui'
    for (const [k, l] of Object.entries(LAYERS)) {
      if (!layers[k]) continue
      g.fillStyle = l.color; g.fillRect(lx, 47, 14, 3)
      g.fillStyle = '#444'; g.fillText(l.label, lx + 18, 52); lx += 30 + g.measureText(l.label).width
    }
    const y0 = 62
    g.drawImage(specRef.current, pad, y0)
    g.drawImage(ovRef.current, pad, y0)
    g.strokeStyle = '#ccc'; g.strokeRect(pad - 0.5, y0 - 0.5, COLS + 1, SPEC_H + 1)
    g.drawImage(oscRef.current, pad, y0 + SPEC_H + 24)
    g.strokeRect(pad - 0.5, y0 + SPEC_H + 23.5, COLS + 1, OSC_H + 1)
    g.fillStyle = '#555'; g.font = '10px monospace'; g.textAlign = 'right'
    for (let f = 0; f <= maxFreq; f += 1000) {
      const y = y0 + SPEC_H - f / maxFreq * SPEC_H
      g.fillText(`${f} Hz`, pad - 4, Math.min(y0 + SPEC_H, Math.max(y0 + 8, y + 3)))
    }
    g.textAlign = 'left'
    if (layers.f0)  { g.fillStyle = LAYERS.f0.color;  g.fillText(`${F0_MAX} Hz`, pad + COLS + 4, y0 + 9);  g.fillText(`${F0_MIN} Hz`, pad + COLS + 4, y0 + SPEC_H) }
    if (layers.int) { g.fillStyle = LAYERS.int.color; g.fillText('0 dB', pad + COLS + 4, y0 + 21); g.fillText(`${INT_MIN} dB`, pad + COLS + 4, y0 + SPEC_H - 12) }
    g.fillStyle = '#555'; g.textAlign = 'center'
    for (let s = 0; s <= WINDOW_S; s++) {
      const x = pad + COLS - s / WINDOW_S * COLS
      g.fillText(s ? `−${s} s` : '0 s', Math.min(W - 14, Math.max(pad + 12, x)), y0 + SPEC_H + 15)
      g.fillText(s ? `−${s} s` : '0 s', Math.min(W - 14, Math.max(pad + 12, x)), y0 + SPEC_H + 24 + OSC_H + 15)
    }
    const a = document.createElement('a')
    a.download = `espectrograma_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.png`
    a.href = c.toDataURL('image/png'); a.click()
  }

  // ─── Ejes ───────────────────────────────────────────────────────────
  const fStep = maxFreq > 5000 ? 2000 : 1000
  const freqTicks = []
  for (let f = fStep; f < maxFreq; f += fStep) freqTicks.push(f)
  const hasF0 = running && read.f0 > 0
  const hasDb = running && isFinite(read.db) && read.db > -90

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Barra principal */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center' }}>
        {!running ? (
          <button onClick={start} style={btn('#dc2626')}><Mic size={15}/> Empezar</button>
        ) : (
          <button onClick={stop} style={btn('#b91c1c')}><Square size={12} fill="white"/> Detener</button>
        )}
        {running && (
          <button onClick={() => setFrozen(f => !f)} style={btnGhost(frozen)}>
            {frozen ? <><Play size={13}/> Reanudar</> : <><Pause size={13}/> Congelar</>}
          </button>
        )}
        {sr > 0 && (
          <button onClick={savePng} style={btnGhost()}><Download size={13}/> Guardar imagen</button>
        )}
        <div style={{ flex: 1 }}/>
        <button onClick={() => setShowCfg(v => !v)} style={btnGhost(showCfg)}>
          <SlidersHorizontal size={13}/> Ajustes
        </button>
      </div>

      {showCfg && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, padding: '12px 14px', background: '#f9fafb', border: '1px solid #e8ecf0', borderRadius: 10 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'center' }}>
            <span><span style={label}>Voz (para formantes)</span>
              <Seg value={voice} onChange={setVoice} options={Object.entries(VOICES).map(([k, v]) => ({ value: k, label: v.label }))}/>
            </span>
            <span><span style={label}>Análisis</span>
              <Seg value={mode} onChange={setMode} options={Object.entries(MODES).map(([k, m]) => ({ value: k, label: m.label }))}/>
            </span>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'center' }}>
            <span><span style={label}>Hasta</span>
              <Seg value={maxFreq} onChange={setMaxFreq} options={FREQS.map(f => ({ value: f, label: `${f / 1000} kHz` }))}/>
            </span>
            <span><span style={label}>Aspecto</span>
              <Seg value={theme} onChange={setTheme} options={Object.entries(THEMES).map(([k, t]) => ({ value: k, label: t.label }))}/>
            </span>
            <span><span style={label}>Tamaño de la onda</span>
              <Seg value={gain} onChange={setGain} options={GAINS.map(g => ({ value: g, label: `×${g}` }))}/>
            </span>
          </div>
          <label style={{ fontSize: 12, color: '#555', display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
            <input type="checkbox" checked={showDetail} onChange={e => setShowDetail(e.target.checked)}/>
            Mostrar detalle de la onda (últimos {SCOPE_MS} ms)
          </label>
          <div style={{ fontSize: 11, color: '#999', lineHeight: 1.5 }}>
            {MODES[mode].desc}. Formantes por LPC (techo {VOICES[voice].ceiling} Hz, cada 10 ms, sólo en tramos sonoros).
            F0 en escala {F0_MIN}–{F0_MAX} Hz (eje azul de la derecha). Intensidad en dB relativos al máximo del micrófono (no calibrada en dB SPL).
            {sr > 0 && <> Muestreo {sr} Hz.</>} Procesado automático del navegador desactivado.
          </div>
        </div>
      )}

      {error && (
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', padding: '8px 12px', background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#b91c1c', fontSize: 13 }}>
          <AlertTriangle size={14}/> {error}
        </div>
      )}

      {/* Lecturas */}
      {running && (
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'baseline', columnGap: 22, rowGap: 6, minHeight: 30 }}>
          <Reading label="F0" color={LAYERS.f0.color} value={hasF0 ? Math.round(read.f0) : '—'} unit="Hz"/>
          {['F1', 'F2', 'F3'].map((n, k) => (
            <Reading key={n} label={n} color={LAYERS.fmt.color} value={read.fmts?.[k] ? Math.round(read.fmts[k]) : '—'} unit="Hz" small/>
          ))}
          <Reading label="Intensidad" color={LAYERS.int.color} value={hasDb ? read.db.toFixed(0) : '—'} unit="dB"/>
          {read.clip && (
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 12, fontWeight: 600, color: '#b91c1c', background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 6, padding: '3px 8px', alignSelf: 'center' }}>
              <AlertTriangle size={12}/> Señal saturada: aleja el micrófono
            </span>
          )}
          {frozen && <span style={{ fontSize: 12, fontWeight: 600, color: '#116b70', alignSelf: 'center' }}>Imagen congelada</span>}
        </div>
      )}

      {/* Espectrograma + capas */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8, marginBottom: 6 }}>
          <div style={title}>Espectrograma</div>
          <div style={{ flex: 1 }}/>
          {Object.entries(LAYERS).map(([k, l]) => (
            <button key={k} onClick={() => setLayers(s => ({ ...s, [k]: !s[k] }))}
              title={layers[k] ? 'Ocultar' : 'Mostrar'}
              style={{
                display: 'inline-flex', alignItems: 'center', gap: 6, padding: '3px 9px', borderRadius: 999, fontSize: 12, cursor: 'pointer',
                border: `1px solid ${layers[k] ? l.color : '#d1d5db'}`, background: layers[k] ? `${l.color}12` : '#fff',
                color: layers[k] ? '#222' : '#aaa',
              }}>
              <span style={{ width: 14, height: k === 'fmt' ? 6 : 3, borderRadius: 2, background: layers[k] ? l.color : '#d1d5db',
                ...(k === 'fmt' ? { background: 'none', borderTop: `3px dotted ${layers[k] ? l.color : '#d1d5db'}`, height: 0 } : {}) }}/>
              {l.label}
            </button>
          ))}
        </div>
        <div style={{ position: 'relative', borderRadius: 10, overflow: 'hidden', background: th.specBg, border: th.border }}>
          <canvas ref={specRef} width={COLS} height={SPEC_H} style={{ width: '100%', height: SPEC_H, display: 'block' }}/>
          <canvas ref={ovRef} width={COLS} height={SPEC_H} style={{ position: 'absolute', inset: 0, width: '100%', height: SPEC_H, display: 'block' }}/>
          {freqTicks.map(f => (
            <div key={f} style={{ position: 'absolute', left: 0, right: 0, top: `${(1 - f / maxFreq) * 100}%`, borderTop: `1px solid ${th.grid}`, pointerEvents: 'none' }}>
              <span style={{ ...axisTxt, color: th.axis, left: 6, top: -12 }}>{f / 1000} kHz</span>
            </div>
          ))}
          {/* escalas de las capas (derecha) */}
          {layers.f0 && <>
            <span style={{ ...axisTxt, right: 6, top: 3, color: LAYERS.f0.color, fontWeight: 700 }}>{F0_MAX} Hz</span>
            <span style={{ ...axisTxt, right: 6, bottom: 3, color: LAYERS.f0.color, fontWeight: 700 }}>{F0_MIN} Hz</span>
          </>}
          {layers.int && <>
            <span style={{ ...axisTxt, right: 6, top: 15, color: LAYERS.int.color, fontWeight: 700 }}>0 dB</span>
            <span style={{ ...axisTxt, right: 6, bottom: 15, color: LAYERS.int.color, fontWeight: 700 }}>{INT_MIN} dB</span>
          </>}
          {running && !frozen && layers.f0 && hasF0 && (
            <EndTag color={LAYERS.f0.color} top={f0Y(read.f0)}>{Math.round(read.f0)} Hz</EndTag>
          )}
          {running && !frozen && layers.int && hasDb && read.db > INT_MIN && (
            <EndTag color={LAYERS.int.color} top={intY(read.db)}>{read.db.toFixed(0)} dB</EndTag>
          )}
          {!running && (
            <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: th.hint, fontSize: 14 }}>
              Pulsa <b style={{ margin: '0 5px' }}>Empezar</b> y habla
            </div>
          )}
        </div>
      </div>

      {/* Oscilograma */}
      <div>
        <div style={{ ...title, marginBottom: 6 }}>Oscilograma <span style={sub}>· forma de la onda, mismo tiempo</span></div>
        <div style={{ position: 'relative', borderRadius: 10, overflow: 'hidden', background: th.oscBg, border: th.border }}>
          <canvas ref={oscRef} width={COLS} height={OSC_H} style={{ width: '100%', height: OSC_H, display: 'block' }}/>
        </div>
        <TimeAxis ticks={[0, 2, 4, 6]}/>
      </div>

      {/* Osciloscopio de detalle (opcional) */}
      <div style={{ display: showDetail ? 'block' : 'none' }}>
        <div style={{ ...title, marginBottom: 6 }}>Detalle de la onda <span style={sub}>· últimos {SCOPE_MS} ms</span></div>
        <div style={{ position: 'relative', borderRadius: 10, overflow: 'hidden', background: th.oscBg, border: th.border }}>
          <canvas ref={scopeRef} width={800} height={100} style={{ width: '100%', height: 100, display: 'block' }}/>
          <span style={{ ...axisTxt, color: th.axis, right: 6, bottom: 4 }}>10 ms / div</span>
        </div>
      </div>
    </div>
  )
}

function EndTag({ color, top, children }) {
  return (
    <span style={{
      position: 'absolute', right: 4, top: `${top * 100}%`, transform: 'translateY(-50%)',
      fontSize: 10, fontWeight: 700, fontFamily: 'monospace', color: '#fff', background: color,
      padding: '1px 5px', borderRadius: 4, pointerEvents: 'none', transition: 'top 0.1s linear',
    }}>{children}</span>
  )
}

function Reading({ label, value, unit, color, small }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'baseline', gap: 5 }}>
      <span style={{ fontSize: 11, color, fontWeight: 700, letterSpacing: '0.03em' }}>{label}</span>
      <span style={{ fontSize: small ? 18 : 22, fontWeight: 700, color: '#222', fontVariantNumeric: 'tabular-nums', minWidth: small ? 40 : 36, textAlign: 'right' }}>{value}</span>
      <span style={{ fontSize: 11, color: '#aaa' }}>{unit}</span>
    </span>
  )
}

function TimeAxis({ ticks }) {
  return (
    <div style={{ position: 'relative', height: 14, fontSize: 9, fontFamily: 'monospace', color: '#aaa' }}>
      {ticks.map(s => (
        <span key={s} style={{
          position: 'absolute', top: 2, left: `${(1 - s / WINDOW_S) * 100}%`, whiteSpace: 'nowrap',
          transform: s === 0 ? 'translateX(-100%)' : s === WINDOW_S ? 'none' : 'translateX(-50%)',
        }}>{s ? `−${s} s` : '0 s'}</span>
      ))}
    </div>
  )
}

const title = { fontSize: 13, fontWeight: 600, color: '#333' }
const sub   = { fontWeight: 400, color: '#999', fontSize: 12 }

function btn(bg, disabled) {
  return {
    display: 'inline-flex', alignItems: 'center', gap: 6, padding: '7px 14px', borderRadius: 8,
    border: 'none', background: disabled ? '#e5e7eb' : bg, color: disabled ? '#9ca3af' : '#fff',
    fontSize: 13, fontWeight: 600, cursor: disabled ? 'default' : 'pointer',
  }
}

function btnGhost(active) {
  return {
    display: 'inline-flex', alignItems: 'center', gap: 6, padding: '7px 12px', borderRadius: 8,
    border: `1px solid ${active ? '#116b70' : '#d1d5db'}`, background: active ? '#eef7f7' : '#fff',
    color: active ? '#116b70' : '#444', fontSize: 13, cursor: 'pointer',
  }
}
