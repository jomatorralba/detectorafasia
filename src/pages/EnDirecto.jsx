import { useState, useRef, useEffect, useCallback } from 'react'
import { Mic, Square, Pause, Play, Download, AlertTriangle, SlidersHorizontal } from 'lucide-react'
import { LiveAnalyzer, MODES } from '../utils/liveAnalysis.js'
import { magmaColor } from '../utils/spectrogram.js'

// ─── Parámetros de visualización ──────────────────────────────────────
const HOP_MS      = 5                       // una columna cada 5 ms
const WINDOW_S    = 6                       // segundos visibles
const COLS        = (WINDOW_S * 1000) / HOP_MS // 1200 columnas
const SPEC_H      = 300                     // filas del espectrograma
const OSC_H       = 120
const SCOPE_MS    = 40                      // ventana del osciloscopio de detalle
const FREQS       = [4000, 5000, 8000]
const GAINS       = [1, 2, 5, 10, 20]
const DYN_RANGE   = 70                      // dB visibles bajo el pico

// LUT de color (magma, igual que el resto de la app)
const LUT = Array.from({ length: 256 }, (_, i) => magmaColor(i / 255))

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

// ─── UI auxiliar ──────────────────────────────────────────────────────
function Seg({ options, value, onChange, disabled }) {
  return (
    <div style={{ display: 'inline-flex', border: '1px solid #d1d5db', borderRadius: 8, overflow: 'hidden' }}>
      {options.map(o => (
        <button key={o.value} disabled={disabled} onClick={() => onChange(o.value)}
          style={{
            padding: '5px 10px', fontSize: 12, border: 'none', cursor: disabled ? 'default' : 'pointer',
            background: value === o.value ? '#116b70' : '#fff',
            color: value === o.value ? '#fff' : '#555',
            borderRight: '1px solid #e5e7eb',
          }}>{o.label}</button>
      ))}
    </div>
  )
}

const label = { fontSize: 11, color: '#888', marginRight: 6 }
const axisTxt = { position: 'absolute', fontSize: 9, fontFamily: 'monospace', color: 'rgba(255,255,255,0.55)', pointerEvents: 'none' }

// ─── Página ───────────────────────────────────────────────────────────
export function EnDirecto() {
  const [running, setRunning] = useState(false)
  const [frozen,  setFrozen]  = useState(false)
  const [mode,    setMode]    = useState('wide')
  const [maxFreq, setMaxFreq] = useState(5000)
  const [gain,    setGain]    = useState(2)
  const [read,    setRead]    = useState({ f0: 0, db: -Infinity, clip: false })
  const [error,   setError]   = useState('')
  const [sr,      setSr]      = useState(0)
  const [showCfg, setShowCfg] = useState(false)
  const [showDetail, setShowDetail] = useState(false)

  const specRef  = useRef(null)
  const oscRef   = useRef(null)
  const scopeRef = useRef(null)
  const audio    = useRef(null)    // { ctx, stream, node, sink }
  const queue    = useRef([])
  const anRef    = useRef(null)
  const rafRef   = useRef(0)
  const settings = useRef({ frozen, maxFreq, gain })
  settings.current = { frozen, maxFreq, gain }
  const peakDb   = useRef(-40)
  const lastClip = useRef(-1)
  const lastRead = useRef(0)
  const f0Hist   = useRef([])

  const clearCanvases = useCallback(() => {
    for (const c of [specRef.current, oscRef.current, scopeRef.current]) {
      if (!c) continue
      const g = c.getContext('2d'); g.fillStyle = c === specRef.current ? '#000004' : '#070a0f'
      g.fillRect(0, 0, c.width, c.height)
    }
    drawOscGrid()
  }, [])

  function drawOscGrid() {
    const c = oscRef.current; if (!c) return
    const g = c.getContext('2d')
    g.fillStyle = 'rgba(255,255,255,0.08)'; g.fillRect(0, OSC_H / 2, c.width, 1)
  }

  // Al cambiar modo de análisis: nuevo analizador y lienzo limpio
  useEffect(() => {
    if (audio.current) anRef.current = new LiveAnalyzer(audio.current.ctx.sampleRate, { mode, hopMs: HOP_MS })
    clearCanvases()
  }, [mode, clearCanvases])

  // Al cambiar escala de frecuencia: limpiar espectrograma
  useEffect(() => {
    const c = specRef.current; if (!c) return
    const g = c.getContext('2d'); g.fillStyle = '#000004'; g.fillRect(0, 0, c.width, c.height)
  }, [maxFreq])

  // ─── Dibujo de columnas ─────────────────────────────────────────────
  const drawColumns = (cols, an) => {
    const { maxFreq: mf, gain: gn } = settings.current
    const n = cols.length
    const spec = specRef.current, osc = oscRef.current
    const gs = spec.getContext('2d'), go = osc.getContext('2d')

    // desplazar a la izquierda n píxeles
    gs.drawImage(spec, -n, 0)
    go.drawImage(osc, -n, 0)
    go.fillStyle = '#070a0f'; go.fillRect(COLS - n, 0, n, OSC_H)
    go.fillStyle = 'rgba(255,255,255,0.08)'; go.fillRect(COLS - n, OSC_H / 2, n, 1)

    const img = gs.createImageData(n, SPEC_H)
    const nyq = an.sr / 2
    for (let k = 0; k < n; k++) {
      const c = cols[k]
      // pico adaptativo para el rango de color (sube rápido, baja lento)
      let cmax = -200
      const top = Math.min(c.db.length - 1, Math.floor(Math.min(mf, nyq) / an.binHz))
      for (let b = 2; b <= top; b++) if (c.db[b] > cmax) cmax = c.db[b]
      peakDb.current = cmax > peakDb.current ? cmax : Math.max(-60, peakDb.current - 0.02)
      const ceil = peakDb.current, floor = ceil - DYN_RANGE

      for (let y = 0; y < SPEC_H; y++) {
        const f = ((SPEC_H - 1 - y) + 0.5) / SPEC_H * mf
        const idx = (y * n + k) * 4
        let v = 0
        if (f < nyq) {
          const pos = f / an.binHz, b0 = Math.floor(pos), fr = pos - b0
          const d = c.db[b0] * (1 - fr) + (c.db[b0 + 1] ?? c.db[b0]) * fr
          v = Math.max(0, Math.min(255, Math.round((d - floor) / DYN_RANGE * 255)))
        }
        const [r, g, bl] = LUT[v]
        img.data[idx] = r; img.data[idx + 1] = g; img.data[idx + 2] = bl; img.data[idx + 3] = 255
      }

      // oscilograma: barra min→max
      const clip = c.max >= 0.99 || c.min <= -0.99
      if (clip) lastClip.current = c.t
      const y1 = (0.5 - Math.min(1, c.max * gn) * 0.46) * OSC_H
      const y2 = (0.5 - Math.max(-1, c.min * gn) * 0.46) * OSC_H
      go.fillStyle = clip ? '#ef4444' : 'rgba(251,146,60,0.95)'
      go.fillRect(COLS - n + k, Math.floor(y1), 1, Math.max(1, Math.ceil(y2 - y1)))
    }
    gs.putImageData(img, COLS - n, 0)
  }

  const drawScope = an => {
    const c = scopeRef.current; if (!c) return
    const g = c.getContext('2d'), w = c.width, h = c.height
    const span = Math.round(an.sr * SCOPE_MS / 1000)
    const raw  = an.recent(span * 2)
    if (raw.length < span * 2) return
    // disparo en cruce por cero ascendente para imagen estable
    let start = 0
    for (let i = span; i > 1; i--) if (raw[i - 1] < 0 && raw[i] >= 0) { start = i; break }
    const seg = raw.subarray(start, start + span)
    let pk = 0.02; for (let i = 0; i < seg.length; i++) pk = Math.max(pk, Math.abs(seg[i]))
    g.fillStyle = '#070a0f'; g.fillRect(0, 0, w, h)
    g.fillStyle = 'rgba(255,255,255,0.08)'; g.fillRect(0, h / 2, w, 1)
    for (let ms = 10; ms < SCOPE_MS; ms += 10) g.fillRect(Math.round(ms / SCOPE_MS * w), 0, 1, h)
    g.strokeStyle = 'rgba(56,189,248,0.95)'; g.lineWidth = 1.5; g.beginPath()
    for (let i = 0; i < seg.length; i++) {
      const x = i / (span - 1) * w, y = (0.5 - seg[i] / pk * 0.44) * h
      i ? g.lineTo(x, y) : g.moveTo(x, y)
    }
    g.stroke()
  }

  // ─── Bucle de animación ─────────────────────────────────────────────
  const loop = () => {
    rafRef.current = requestAnimationFrame(loop)
    const an = anRef.current; if (!an) return
    const chunks = queue.current; queue.current = []
    const cols = []
    for (const ch of chunks) cols.push(...an.push(ch))
    if (!cols.length || settings.current.frozen) return
    const hist = f0Hist.current
    for (const c of cols) hist.push(c.f0)
    if (hist.length > 40) hist.splice(0, hist.length - 40)   // últimos 200 ms
    // si la pestaña estuvo oculta puede llegar mucho de golpe: sólo lo visible
    drawColumns(cols.length > COLS ? cols.slice(-COLS) : cols, an)
    drawScope(an)
    const now = performance.now()
    if (now - lastRead.current < 100) return   // lecturas numéricas a ~10 Hz
    lastRead.current = now
    const last = cols[cols.length - 1]
    // F0: mediana de las últimas columnas sonoras para una lectura estable
    const f0s = hist.filter(Boolean).sort((a, b) => a - b)
    setRead({
      f0: f0s.length >= 10 ? f0s[Math.floor(f0s.length / 2)] : 0,
      db: last.rmsDb,
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
      anRef.current = new LiveAnalyzer(ctx.sampleRate, { mode, hopMs: HOP_MS })
      queue.current = []
      peakDb.current = -40
      setSr(ctx.sampleRate)
      clearCanvases()
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

  useEffect(() => { clearCanvases(); return () => stop() }, []) // eslint-disable-line

  // ─── Exportar imagen ────────────────────────────────────────────────
  const savePng = () => {
    const pad = 44, W = COLS + pad + 12, H = 40 + SPEC_H + 24 + OSC_H + 30
    const c = document.createElement('canvas'); c.width = W; c.height = H
    const g = c.getContext('2d')
    g.fillStyle = '#fff'; g.fillRect(0, 0, W, H)
    g.fillStyle = '#111'; g.font = '600 15px system-ui'
    g.fillText('Espectrograma y oscilograma', pad, 22)
    g.fillStyle = '#777'; g.font = '12px system-ui'
    g.fillText(`${new Date().toLocaleString('es-ES')} · ${MODES[mode].label} · 0–${maxFreq / 1000} kHz · oscilograma ×${gain} · ${sr} Hz`, pad, 36)
    const y0 = 44
    g.drawImage(specRef.current, pad, y0)
    g.drawImage(oscRef.current, pad, y0 + SPEC_H + 24)
    g.fillStyle = '#555'; g.font = '10px monospace'; g.textAlign = 'right'
    for (let f = 0; f <= maxFreq; f += 1000) {
      const y = y0 + SPEC_H - f / maxFreq * SPEC_H
      g.fillText(`${f / 1000}k`, pad - 4, Math.min(y0 + SPEC_H, Math.max(y0 + 8, y + 3)))
    }
    g.textAlign = 'center'
    for (let s = 0; s <= WINDOW_S; s++) {
      const x = pad + COLS - s / WINDOW_S * COLS
      g.fillText(s ? `−${s} s` : '0 s', Math.min(W - 14, Math.max(pad + 12, x)), y0 + SPEC_H + 15)
      g.fillText(s ? `−${s} s` : '0 s', Math.min(W - 14, Math.max(pad + 12, x)), y0 + SPEC_H + 24 + OSC_H + 15)
    }
    const a = document.createElement('a')
    a.download = `espectrograma_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.png`
    a.href = c.toDataURL('image/png'); a.click()
  }

  // ─── Ejes (HTML, nítidos a cualquier tamaño) ────────────────────────
  const fStep = maxFreq > 5000 ? 2000 : 1000
  const freqTicks = []
  for (let f = fStep; f < maxFreq; f += fStep) freqTicks.push(f)
  const timeTicks = [0, 2, 4, 6]
  const hasF0 = running && read.f0 > 0
  const hasDb = running && isFinite(read.db) && read.db > -90

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Barra principal: sólo lo imprescindible */}
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
            <span><span style={label}>Análisis</span>
              <Seg value={mode} onChange={setMode} options={Object.entries(MODES).map(([k, m]) => ({ value: k, label: m.label }))}/>
            </span>
            <span><span style={label}>Hasta</span>
              <Seg value={maxFreq} onChange={setMaxFreq} options={FREQS.map(f => ({ value: f, label: `${f / 1000} kHz` }))}/>
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
            {MODES[mode].desc}. {sr > 0 && <>Muestreo {sr} Hz · columna cada {HOP_MS} ms · rango {DYN_RANGE} dB. </>}
            Procesado automático del navegador desactivado (eco, ruido, ganancia).
          </div>
        </div>
      )}

      {error && (
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', padding: '8px 12px', background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#b91c1c', fontSize: 13 }}>
          <AlertTriangle size={14}/> {error}
        </div>
      )}

      {/* Lectura compacta */}
      {running && <div style={{ display: 'flex', alignItems: 'baseline', gap: 22, minHeight: 30 }}>
        <Reading label="Tono (F0)" value={hasF0 ? Math.round(read.f0) : '—'} unit="Hz"/>
        <Reading label="Volumen" value={hasDb ? read.db.toFixed(0) : '—'} unit="dB"/>
        {running && read.clip && (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 12, fontWeight: 600, color: '#b91c1c', background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 6, padding: '3px 8px', alignSelf: 'center' }}>
            <AlertTriangle size={12}/> Señal saturada: aleja el micrófono
          </span>
        )}
        {frozen && <span style={{ fontSize: 12, fontWeight: 600, color: '#116b70', alignSelf: 'center' }}>Imagen congelada</span>}
      </div>}

      {/* Espectrograma */}
      <div>
        <div style={title}>Espectrograma <span style={sub}>· frecuencias de la voz</span></div>
        <div style={{ position: 'relative', borderRadius: 10, overflow: 'hidden', background: '#000004' }}>
          <canvas ref={specRef} width={COLS} height={SPEC_H} style={{ width: '100%', height: SPEC_H, display: 'block' }}/>
          {freqTicks.map(f => (
            <div key={f} style={{ position: 'absolute', left: 0, right: 0, top: `${(1 - f / maxFreq) * 100}%`, borderTop: '1px solid rgba(255,255,255,0.07)', pointerEvents: 'none' }}>
              <span style={{ ...axisTxt, left: 6, top: -12 }}>{f / 1000} kHz</span>
            </div>
          ))}
          {!running && (
            <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'rgba(255,255,255,0.55)', fontSize: 14 }}>
              Pulsa <b style={{ margin: '0 5px', color: 'rgba(255,255,255,0.85)' }}>Empezar</b> y habla
            </div>
          )}
        </div>
      </div>

      {/* Oscilograma */}
      <div>
        <div style={title}>Oscilograma <span style={sub}>· forma de la onda, mismo tiempo</span></div>
        <div style={{ position: 'relative', borderRadius: 10, overflow: 'hidden', background: '#070a0f' }}>
          <canvas ref={oscRef} width={COLS} height={OSC_H} style={{ width: '100%', height: OSC_H, display: 'block' }}/>
        </div>
        <TimeAxis ticks={timeTicks}/>
      </div>

      {/* Osciloscopio de detalle (opcional) */}
      <div style={{ display: showDetail ? 'block' : 'none' }}>
        <div style={title}>Detalle de la onda <span style={sub}>· últimos {SCOPE_MS} ms</span></div>
        <div style={{ position: 'relative', borderRadius: 10, overflow: 'hidden', background: '#070a0f' }}>
          <canvas ref={scopeRef} width={800} height={100} style={{ width: '100%', height: 100, display: 'block' }}/>
          <span style={{ ...axisTxt, right: 6, bottom: 4 }}>10 ms / div</span>
        </div>
      </div>
    </div>
  )
}

function Reading({ label, value, unit }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'baseline', gap: 6 }}>
      <span style={{ fontSize: 11, color: '#999', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</span>
      <span style={{ fontSize: 22, fontWeight: 700, color: '#222', fontVariantNumeric: 'tabular-nums', minWidth: 44, textAlign: 'right' }}>{value}</span>
      <span style={{ fontSize: 12, color: '#aaa' }}>{unit}</span>
    </span>
  )
}

const title = { fontSize: 13, fontWeight: 600, color: '#333', marginBottom: 6 }
const sub   = { fontWeight: 400, color: '#999', fontSize: 12 }

function btnGhost(active) {
  return {
    display: 'inline-flex', alignItems: 'center', gap: 6, padding: '7px 12px', borderRadius: 8,
    border: `1px solid ${active ? '#116b70' : '#d1d5db'}`, background: active ? '#eef7f7' : '#fff',
    color: active ? '#116b70' : '#444', fontSize: 13, cursor: 'pointer',
  }
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

function btn(bg, disabled) {
  return {
    display: 'inline-flex', alignItems: 'center', gap: 6, padding: '7px 14px', borderRadius: 8,
    border: 'none', background: disabled ? '#e5e7eb' : bg, color: disabled ? '#9ca3af' : '#fff',
    fontSize: 13, fontWeight: 600, cursor: disabled ? 'default' : 'pointer',
  }
}
