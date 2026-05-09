import { useState, useRef, useEffect } from 'react'
import { Mic, Square, RotateCcw, CheckCircle2, AlertCircle } from 'lucide-react'

// ─── Phrase definitions ───────────────────────────────────────────────

const PHRASES = [
  {
    id:    'declarative',
    label: 'Declarativa',
    text:  'El plato está en la mesa.',
    hint:  'Entonación descendente — afirmación natural',
    color: '#116b70',
  },
  {
    id:    'interrogative',
    label: 'Interrogativa',
    text:  '¿El plato está en la mesa?',
    hint:  'El tono sube al final de la frase',
    color: '#b45309',
  },
  {
    id:    'exclamative',
    label: 'Exclamativa',
    text:  '¡El plato está en la mesa!',
    hint:  'Con energía y sorpresa — rango tonal amplio',
    color: '#b91c1c',
  },
]

// ─── F0 extraction (YIN algorithm) ───────────────────────────────────────────

function yinPitch(frame, sr) {
  const W  = frame.length
  const lo = Math.ceil(sr / 500)   // period for 500 Hz (highest)
  const hi = Math.floor(sr / 75)   // period for 75 Hz  (lowest)
  if (hi >= W / 2) return 0

  // Difference function
  const d = new Float32Array(hi + 1)
  const lim = W - hi
  for (let τ = 1; τ <= hi; τ++) {
    let s = 0
    for (let i = 0; i < lim; i++) { const x = frame[i] - frame[i + τ]; s += x * x }
    d[τ] = s
  }

  // Cumulative mean normalized difference
  const cmnd = new Float32Array(hi + 1); cmnd[0] = 1
  let rs = 0
  for (let τ = 1; τ <= hi; τ++) { rs += d[τ]; cmnd[τ] = rs > 0 ? d[τ] * τ / rs : 1 }

  // First valley below threshold (absolute threshold = 0.13)
  for (let τ = lo; τ < hi; τ++) {
    if (cmnd[τ] < 0.13) {
      while (τ + 1 < hi && cmnd[τ + 1] < cmnd[τ]) τ++
      return sr / τ
    }
  }
  return 0 // unvoiced
}

function extractF0(pcm, sr) {
  const FN = Math.round(sr * 0.025) // 25 ms frame
  const HN = Math.round(sr * 0.010) // 10 ms hop
  const times = [], raw = []

  for (let i = 0; i + FN <= pcm.length; i += HN) {
    const frame = pcm.subarray(i, i + FN)
    // Skip silence (RMS < 0.008)
    let rms = 0
    for (let k = 0; k < frame.length; k++) rms += frame[k] * frame[k]
    if (Math.sqrt(rms / frame.length) < 0.008) continue

    const f0 = yinPitch(frame, sr)
    if (f0 > 0) { times.push(+(i / sr).toFixed(3)); raw.push(+f0.toFixed(1)) }
  }

  // Median smooth (window = 5) to remove transient spikes
  const f0s = raw.map((_, i) => {
    const s = raw.slice(Math.max(0, i - 2), i + 3).sort((a, b) => a - b)
    return s[Math.floor(s.length / 2)]
  })

  return { times, f0s }
}

function computeMetrics(f0s) {
  if (!f0s.length) return { mean: 0, std: 0, range: 0 }
  const n    = f0s.length
  const mean = f0s.reduce((a, b) => a + b, 0) / n
  const std  = Math.sqrt(f0s.reduce((a, b) => a + (b - mean) ** 2, 0) / n)
  const range = Math.max(...f0s) - Math.min(...f0s)
  return { mean: +mean.toFixed(1), std: +std.toFixed(1), range: +range.toFixed(1) }
}

function getSupportedMimeType() {
  const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/ogg', 'audio/mp4']
  return types.find(t => MediaRecorder.isTypeSupported(t)) || ''
}

// ─── Waveform canvas ──────────────────────────────────────────────────

function WaveformCanvas({ analyserRef, active }) {
  const canvasRef = useRef(null)
  const rafRef    = useRef(null)

  useEffect(() => {
    if (!active || !analyserRef.current) return
    const canvas   = canvasRef.current; if (!canvas) return
    const ctx      = canvas.getContext('2d')
    const analyser = analyserRef.current
    const buf      = new Uint8Array(analyser.fftSize)

    const draw = () => {
      rafRef.current = requestAnimationFrame(draw)
      analyser.getByteTimeDomainData(buf)
      ctx.fillStyle = '#f0faf9'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.lineWidth   = 2
      ctx.strokeStyle = '#116b70'
      ctx.beginPath()
      const sw = canvas.width / buf.length
      let x = 0
      for (let i = 0; i < buf.length; i++) {
        const y = (buf[i] / 128.0) * (canvas.height / 2)
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
        x += sw
      }
      ctx.stroke()
    }
    draw()
    return () => cancelAnimationFrame(rafRef.current)
  }, [active, analyserRef])

  return (
    <canvas
      ref={canvasRef} width={460} height={58}
      style={{ width: '100%', height: 58, borderRadius: 8, background: '#f0faf9', display: 'block', border: '1px solid #b2dede' }}
    />
  )
}

// ─── F0 contour chart ──────────────────────────────────────────────

// Approximate reference contours (normalised 0–1 within the data's F0 range)
const REF_SHAPES = {
  declarative:   [[0, 0.75], [0.35, 0.6], [0.7, 0.35], [1, 0.2]],
  interrogative: [[0, 0.5],  [0.4, 0.45], [0.75, 0.4], [1, 0.8]],
  exclamative:   [[0, 0.55], [0.25, 0.85],[0.6, 0.6],  [1, 0.25]],
}

function F0Chart({ times, f0s, phraseId }) {
  if (!times?.length) return null
  const W = 460, H = 120, pl = 40, pt = 10, pr = 10, pb = 24
  const iW = W - pl - pr, iH = H - pt - pb

  const minF = Math.max(50,  Math.min(...f0s) - 15)
  const maxF = Math.min(550, Math.max(...f0s) + 15)
  const dur  = times[times.length - 1] || 1

  const tx = t  => pl + (t / dur) * iW
  const ty = f0 => pt + iH - ((f0 - minF) / (maxF - minF)) * iH

  const pts = times.map((t, i) => `${tx(t).toFixed(1)},${ty(f0s[i]).toFixed(1)}`).join(' ')

  const refPts = (REF_SHAPES[phraseId] || [])
    .map(([t, r]) => `${tx(t * dur).toFixed(1)},${ty(minF + r * (maxF - minF)).toFixed(1)}`)
    .join(' ')

  const gridFs = []; for (let f = Math.ceil(minF / 50) * 50; f <= maxF; f += 50) gridFs.push(f)

  const COLORS = { declarative: '#116b70', interrogative: '#b45309', exclamative: '#b91c1c' }
  const c = COLORS[phraseId] || '#116b70'

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 140 }}>
      {/* Grid */}
      {gridFs.map(f => (
        <g key={f}>
          <line x1={pl} y1={ty(f)} x2={W - pr} y2={ty(f)} stroke="#f0f0f0" strokeWidth="1"/>
          <text x={pl - 5} y={ty(f) + 4} fontSize="9" fill="#ccc" textAnchor="end">{f}</text>
        </g>
      ))}
      {/* Reference contour (dashed) */}
      {refPts && (
        <polyline points={refPts} fill="none" stroke={c}
          strokeWidth="1.5" strokeDasharray="5 4" opacity="0.3"/>
      )}
      {/* Measured F0 contour */}
      <polyline points={pts} fill="none" stroke={c}
        strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round"/>
      {/* Axis labels */}
      <text x={pl + iW / 2} y={H - 3} fontSize="9" fill="#bbb" textAnchor="middle">tiempo (s)</text>
      <text x={10} y={pt + iH / 2 + 4} fontSize="9" fill="#bbb" textAnchor="middle"
        transform={`rotate(-90 10 ${pt + iH / 2})`}>Hz</text>
    </svg>
  )
}

// ─── Metric card ───────────────────────────────────────────────────

function Card({ label, value, unit, accent }) {
  return (
    <div style={{
      background: accent ? '#eef7f7' : '#f9fafb',
      border: `1px solid ${accent ? '#b2dede' : '#e8ecf0'}`,
      borderRadius: 10, padding: '10px 14px', flex: 1,
    }}>
      <div style={{ fontSize: 10, color: '#999', marginBottom: 3, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        {label}
      </div>
      <div style={{ fontSize: 21, fontWeight: 700, color: accent ? '#116b70' : '#222', lineHeight: 1 }}>
        {value}
        <span style={{ fontSize: 12, fontWeight: 400, color: '#aaa', marginLeft: 3 }}>{unit}</span>
      </div>
    </div>
  )
}

// ─── Main component ─────────────────────────────────────────────────

export function Prosodia({ onResult }) {
  const [activeIdx, setActiveIdx] = useState(0)
  const [recording, setRecording] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [results,   setResults]   = useState({})
  const [error,     setError]     = useState(null)

  const mrRef       = useRef(null)
  const chunksRef   = useRef([])
  const analyserRef = useRef(null)
  const audioCtxRef = useRef(null)

  const phrase   = PHRASES[activeIdx]
  const result   = results[phrase.id]
  const doneIds  = PHRASES.filter(p => results[p.id])
  const allDone  = doneIds.length === 3

  // Report global result when all 3 phrases are recorded
  useEffect(() => {
    if (!allDone) return
    const all = PHRASES.map(p => results[p.id].metrics)
    onResult?.('prosody', {
      f0_mean:  +(all.reduce((a, m) => a + m.mean,  0) / 3).toFixed(1),
      f0_range: +(all.reduce((a, m) => a + m.range, 0) / 3).toFixed(1),
    })
  }, [allDone]) // eslint-disable-line

  const startRecording = async () => {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })

      const ctx      = new AudioContext()
      const source   = ctx.createMediaStreamSource(stream)
      const analyser = ctx.createAnalyser(); analyser.fftSize = 1024
      source.connect(analyser)
      audioCtxRef.current = ctx
      analyserRef.current = analyser

      const mr = new MediaRecorder(stream, { mimeType: getSupportedMimeType() })
      mrRef.current    = mr
      chunksRef.current = []
      mr.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data) }
      mr.start(100)
      setRecording(true)
    } catch {
      setError('No se pudo acceder al micrófono. Verifica los permisos.')
    }
  }

  const stopRecording = async () => {
    const mr = mrRef.current; if (!mr) return
    const phraseId = phrase.id
    setRecording(false)
    setAnalyzing(true)

    await new Promise(resolve => { mr.onstop = resolve; mr.stop() })
    mr.stream.getTracks().forEach(t => t.stop())

    try {
      const blob     = new Blob(chunksRef.current, { type: chunksRef.current[0]?.type || 'audio/webm' })
      const arrBuf   = await blob.arrayBuffer()
      const ctx      = audioCtxRef.current || new AudioContext()
      const audioBuf = await ctx.decodeAudioData(arrBuf)
      const pcm      = audioBuf.getChannelData(0)

      const { times, f0s } = extractF0(pcm, audioBuf.sampleRate)
      if (!f0s.length) throw new Error('Sin voz detectada. Habla más cerca del micrófono e intenta de nuevo.')

      setResults(prev => ({ ...prev, [phraseId]: { times, f0s, metrics: computeMetrics(f0s) } }))
    } catch (e) {
      setError(e.message || 'Error procesando el audio.')
    } finally {
      setAnalyzing(false)
      analyserRef.current = null
      try { await audioCtxRef.current?.close() } catch {}
      audioCtxRef.current = null
    }
  }

  const resetPhrase = id => setResults(prev => { const n = { ...prev }; delete n[id]; return n })

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

      {/* Mode tabs */}
      <div style={{ display: 'flex', gap: 6 }}>
        {PHRASES.map((p, i) => {
          const done   = !!results[p.id]
          const active = i === activeIdx
          return (
            <button
              key={p.id}
              onClick={() => { if (!recording) setActiveIdx(i) }}
              style={{
                flex: 1, padding: '8px 6px', borderRadius: 9,
                border: `1.5px solid ${active ? p.color : done ? '#d1d5db' : '#e5e7eb'}`,
                background: active ? p.color : done ? '#f9fafb' : '#fff',
                color: active ? '#fff' : done ? '#555' : '#888',
                fontSize: 12, fontWeight: 600,
                cursor: recording ? 'default' : 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5,
                transition: 'all 0.15s',
              }}
            >
              {done && <CheckCircle2 size={12} strokeWidth={2.5}/>}
              {p.label}
            </button>
          )
        })}
      </div>

      {/* Phrase display */}
      <div style={{
        background: '#f9fafb', borderRadius: 12,
        border: `1.5px solid ${phrase.color}33`,
        padding: '18px 22px', textAlign: 'center',
      }}>
        <div style={{ fontSize: 22, fontWeight: 700, color: '#111', letterSpacing: '-0.3px', marginBottom: 6 }}>
          {phrase.text}
        </div>
        <div style={{ fontSize: 13, color: '#888' }}>{phrase.hint}</div>
      </div>

      {/* Recorder (only shown when no result yet) */}
      {!result && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {recording && <WaveformCanvas analyserRef={analyserRef} active={recording}/>}

          <div style={{ display: 'flex', justifyContent: 'center' }}>
            {recording ? (
              <button onClick={stopRecording} style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '11px 32px', borderRadius: 10, border: 'none',
                background: '#dc2626', color: '#fff', fontSize: 14, fontWeight: 600, cursor: 'pointer',
              }}>
                <Square size={15} fill="#fff"/> Detener
              </button>
            ) : (
              <button onClick={startRecording} disabled={analyzing} style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '11px 32px', borderRadius: 10, border: 'none',
                background: analyzing ? '#9ca3af' : phrase.color,
                color: '#fff', fontSize: 14, fontWeight: 600,
                cursor: analyzing ? 'default' : 'pointer',
              }}>
                <Mic size={16}/> {analyzing ? 'Analizando…' : 'Grabar'}
              </button>
            )}
          </div>

          {analyzing && (
            <p style={{ textAlign: 'center', fontSize: 12, color: '#888', margin: 0 }}>
              Extrayendo contorno de F0…
            </p>
          )}
        </div>
      )}

      {/* Results for current phrase */}
      {result && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div style={{ display: 'flex', gap: 8 }}>
            <Card label="F0 Medio"       value={result.metrics.mean}  unit="Hz"/>
            <Card label="Desv. Estándar" value={result.metrics.std}   unit="Hz"/>
            <Card label="Rango F0"       value={result.metrics.range} unit="Hz" accent/>
          </div>

          {result.metrics.range < 40 && (
            <div style={{
              display: 'flex', gap: 8, alignItems: 'flex-start',
              background: '#fef3c7', border: '1px solid #fcd34d', borderRadius: 9, padding: '10px 14px',
            }}>
              <AlertCircle size={15} color="#b45309" style={{ flexShrink: 0, marginTop: 1 }}/>
              <span style={{ fontSize: 12, color: '#92400e', lineHeight: 1.5 }}>
                Rango F0 reducido (&lt; 40 Hz): posible voz monótona — indicativo de disartria hipocinética.
              </span>
            </div>
          )}

          <div style={{ background: '#fff', borderRadius: 10, border: '1px solid #e8ecf0', padding: '12px 8px 6px' }}>
            <div style={{ fontSize: 11, color: '#aaa', paddingLeft: 12, marginBottom: 2 }}>
              Contorno melódico F0
              <span style={{ fontSize: 10, color: '#d1d5db', marginLeft: 8 }}>— — referencia ideal</span>
            </div>
            <F0Chart times={result.times} f0s={result.f0s} phraseId={phrase.id}/>
          </div>

          <button onClick={() => resetPhrase(phrase.id)} style={{
            display: 'flex', alignItems: 'center', gap: 6, alignSelf: 'flex-start',
            color: '#aaa', fontSize: 12, background: 'none', border: 'none', cursor: 'pointer', padding: '4px 0',
          }}>
            <RotateCcw size={12}/> Repetir grabación
          </button>
        </div>
      )}

      {error && (
        <div style={{
          background: '#fef2f2', border: '1px solid #fca5a5',
          borderRadius: 9, padding: '10px 14px', fontSize: 13, color: '#dc2626',
        }}>
          {error}
        </div>
      )}

      {/* Global summary */}
      {allDone && (() => {
        const globalMean  = +(PHRASES.reduce((a, p) => a + results[p.id].metrics.mean,  0) / 3).toFixed(1)
        const globalRange = +(PHRASES.reduce((a, p) => a + results[p.id].metrics.range, 0) / 3).toFixed(1)
        return (
          <div style={{ background: '#eef7f7', border: '1px solid #b2dede', borderRadius: 12, padding: '16px 20px' }}>
            <div style={{ fontWeight: 700, color: '#073447', fontSize: 14, marginBottom: 12 }}>
              Resumen de prosodia
            </div>
            <div style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
              <Card label="F0 Medio global"   value={globalMean}  unit="Hz"/>
              <Card label="Rango F0 promedio" value={globalRange} unit="Hz" accent/>
            </div>
            <div style={{ display: 'flex', gap: 14 }}>
              {PHRASES.map(p => (
                <div key={p.id} style={{ flex: 1 }}>
                  <div style={{ fontSize: 10, color: p.color, fontWeight: 700, letterSpacing: '0.05em', textTransform: 'uppercase', marginBottom: 5 }}>
                    {p.label}
                  </div>
                  <div style={{ fontSize: 12, color: '#333' }}>
                    Rango: <strong>{results[p.id].metrics.range} Hz</strong>
                  </div>
                  <div style={{ fontSize: 12, color: '#777', marginTop: 1 }}>
                    Media: {results[p.id].metrics.mean} Hz
                  </div>
                </div>
              ))}
            </div>
          </div>
        )
      })()}

    </div>
  )
}
