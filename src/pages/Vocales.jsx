import { useState, useRef } from 'react'
import { Mic, Square, RotateCcw, CheckCircle2, AlertCircle } from 'lucide-react'
import { extractFormants } from '../utils/lpc'

// ─── Vocales a analizar ────────────────────────────────────────────────────────

const VOWELS = [
  { id: 'a', label: '/a/', hint: 'Vocal abierta central — boca bien abierta', color: '#116b70' },
  { id: 'i', label: '/i/', hint: 'Vocal cerrada anterior — "sonrisa"',         color: '#b45309' },
  { id: 'u', label: '/u/', hint: 'Vocal cerrada posterior — labios redondeados', color: '#7c3aed' },
]

// Referencias canónicas para adultos hablantes de español
// (Quilis & Esgueva 1983; Martínez Celdrán 2004)
const REF = {
  a: { F1: 700, F2: 1200 },
  i: { F1: 300, F2: 2200 },
  u: { F1: 350, F2:  800 },
}

function getSupportedMimeType() {
  const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/ogg', 'audio/mp4']
  return types.find(t => MediaRecorder.isTypeSupported(t)) || ''
}

// ─── Diagrama F1-F2 (plano vocálico canónico) ─────────────────────────────────

function VowelPlot({ formants }) {
  const W = 420, H = 240, pl = 40, pt = 10, pr = 20, pb = 30
  const iW = W - pl - pr, iH = H - pt - pb

  // Rango del plano: F2 eje X invertido (3000→500), F1 eje Y invertido (800→150)
  const F2min = 500, F2max = 3000
  const F1min = 150, F1max = 800

  const tx = f2 => pl + (1 - (f2 - F2min) / (F2max - F2min)) * iW  // F2 invertido
  const ty = f1 => pt + ((f1 - F1min) / (F1max - F1min)) * iH      // F1 hacia abajo

  const colors = { a: '#116b70', i: '#b45309', u: '#7c3aed' }
  const labels = { a: '/a/', i: '/i/', u: '/u/' }

  // Referencias grises
  const refPoints = VOWELS.map(v => ({ id: v.id, x: tx(REF[v.id].F2), y: ty(REF[v.id].F1) }))

  // Puntos medidos
  const measured = VOWELS
    .filter(v => formants[v.id])
    .map(v => ({ id: v.id, x: tx(formants[v.id].F2), y: ty(formants[v.id].F1), color: colors[v.id] }))

  const refPoly = refPoints.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')
  const meaPoly = measured.length === 3
    ? measured.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')
    : null

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 220 }}>
      {/* Ejes */}
      <line x1={pl} y1={pt} x2={pl} y2={H - pb} stroke="#e5e7eb" strokeWidth="1"/>
      <line x1={pl} y1={H - pb} x2={W - pr} y2={H - pb} stroke="#e5e7eb" strokeWidth="1"/>
      {/* Etiquetas ejes */}
      <text x={pl + iW / 2} y={H - 4} fontSize="9" fill="#bbb" textAnchor="middle">F2 (Hz) →</text>
      <text x={10} y={pt + iH / 2 + 4} fontSize="9" fill="#bbb" textAnchor="middle"
        transform={`rotate(-90 10 ${pt + iH / 2})`}>F1 (Hz) ↓</text>

      {/* Marcas F2 */}
      {[500, 1000, 1500, 2000, 2500, 3000].map(f2 => (
        <g key={f2}>
          <line x1={tx(f2)} y1={H - pb} x2={tx(f2)} y2={H - pb + 3} stroke="#ddd" strokeWidth="1"/>
          <text x={tx(f2)} y={H - pb + 11} fontSize="8" fill="#ccc" textAnchor="middle">{f2}</text>
        </g>
      ))}

      {/* Triángulo de referencia (gris claro) */}
      <polygon points={refPoly} fill="rgba(0,0,0,0.03)" stroke="#d1d5db" strokeWidth="1" strokeDasharray="4 3"/>
      {refPoints.map(p => (
        <g key={p.id}>
          <circle cx={p.x} cy={p.y} r={3} fill="#e5e7eb" stroke="#d1d5db" strokeWidth="1"/>
          <text x={p.x + 5} y={p.y + 4} fontSize="9" fill="#bbb">{labels[p.id]}</text>
        </g>
      ))}

      {/* Triángulo medido */}
      {meaPoly && (
        <polygon points={meaPoly} fill="rgba(17,107,112,0.08)" stroke="#116b70" strokeWidth="1.5"/>
      )}
      {measured.map(p => (
        <g key={p.id}>
          <circle cx={p.x} cy={p.y} r={5} fill={p.color} stroke="white" strokeWidth="1.5"/>
          <text x={p.x + 7} y={p.y + 4} fontSize="10" fontWeight="600" fill={p.color}>
            {labels[p.id]}
          </text>
        </g>
      ))}
    </svg>
  )
}

// ─── Componente principal ─────────────────────────────────────────────────────

export function Vocales({ onResult }) {
  const [activeIdx, setActiveIdx] = useState(0)
  const [recording, setRecording] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [formants,  setFormants]  = useState({})
  const [error,     setError]     = useState(null)

  const mrRef       = useRef(null)
  const chunksRef   = useRef([])
  const audioCtxRef = useRef(null)

  const vowel  = VOWELS[activeIdx]
  const allDone = VOWELS.every(v => formants[v.id])

  const startRecording = async () => {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
      const ctx    = new AudioContext()
      audioCtxRef.current = ctx
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
    const vowelId = vowel.id
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

      if (audioBuf.duration < 1.0)
        throw new Error('Grabación muy corta. Sostén la vocal al menos 2 segundos.')

      const result = extractFormants(pcm, audioBuf.sampleRate)
      if (!result)
        throw new Error('No se detectaron formantes. Habla más cerca del micrófono y sostén la vocal.')

      const updated = { ...formants, [vowelId]: result }
      setFormants(updated)

      // Reportar VSA y FCR cuando las tres vocales estén grabadas
      if (VOWELS.every(v => updated[v.id])) {
        const fa = updated.a, fi = updated.i, fu = updated.u
        const vsa = 0.5 * Math.abs(
          fa.F2 * (fi.F1 - fu.F1) + fi.F2 * (fu.F1 - fa.F1) + fu.F2 * (fa.F1 - fi.F1)
        )
        const fcr = fu.F2 + fa.F2 + fi.F1 + fu.F1 > 0
          ? (fu.F2 + fa.F2 + fi.F1 + fu.F1) / (fi.F2 + fa.F1)
          : null
        onResult?.('formants', {
          a: { F1: fa.F1, F2: fa.F2 },
          i: { F1: fi.F1, F2: fi.F2 },
          u: { F1: fu.F1, F2: fu.F2 },
          vsa: vsa ? +vsa.toFixed(0) : null,
          fcr: fcr ? +fcr.toFixed(3) : null,
        })
      }

      // Avanzar a la siguiente vocal sin grabar
      if (activeIdx < VOWELS.length - 1) setActiveIdx(activeIdx + 1)
    } catch (e) {
      setError(e.message || 'Error procesando el audio.')
    } finally {
      setAnalyzing(false)
      try { await audioCtxRef.current?.close() } catch {}
      audioCtxRef.current = null
    }
  }

  const resetVowel = id => setFormants(prev => { const n = { ...prev }; delete n[id]; return n })

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ background: '#e8f4f4', borderLeft: '4px solid #116b70', borderRadius: '0 12px 12px 0', padding: 16 }}>
        <p className="font-semibold text-sm" style={{ color: '#073447' }}>Instrucciones</p>
        <p className="text-sm mt-1" style={{ color: '#0e5a5e' }}>
          Tome aire y emita cada vocal de forma <strong>sostenida y estable</strong> durante al menos 2 segundos.
          Mantenga el volumen uniforme. Se analizarán las frecuencias de resonancia (formantes) de su voz.
        </p>
      </div>

      {/* Tabs vocales */}
      <div style={{ display: 'flex', gap: 6 }}>
        {VOWELS.map((v, i) => {
          const done   = !!formants[v.id]
          const active = i === activeIdx
          return (
            <button key={v.id} onClick={() => { if (!recording) setActiveIdx(i) }}
              style={{
                flex: 1, padding: '8px 6px', borderRadius: 9,
                border: `1.5px solid ${active ? v.color : done ? '#d1d5db' : '#e5e7eb'}`,
                background: active ? v.color : done ? '#f9fafb' : '#fff',
                color: active ? '#fff' : done ? '#6b7280' : '#aaa',
                fontWeight: active ? 700 : 400, fontSize: 13, cursor: recording ? 'not-allowed' : 'pointer',
                display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2,
              }}>
              <span style={{ fontSize: 18, fontWeight: 700 }}>{v.label}</span>
              {done && <CheckCircle2 size={12} color={active ? '#fff' : '#22c55e'}/>}
            </button>
          )
        })}
      </div>

      {/* Panel de grabación */}
      <div style={{ background: '#fff', border: '1px solid #f0f0f0', borderRadius: 14, padding: 20, display: 'flex', flexDirection: 'column', gap: 14 }}>
        <div>
          <p className="font-semibold" style={{ color: '#111', marginBottom: 2 }}>{vowel.label} — {vowel.hint}</p>
          {formants[vowel.id] && (
            <p style={{ fontSize: 12, color: '#116b70', fontFamily: 'monospace' }}>
              F1 = {formants[vowel.id].F1} Hz &nbsp;·&nbsp; F2 = {formants[vowel.id].F2} Hz
              {formants[vowel.id].F3 ? ` · F3 = ${formants[vowel.id].F3} Hz` : ''}
            </p>
          )}
        </div>

        <div style={{ display: 'flex', justifyContent: 'center', gap: 10 }}>
          {!recording ? (
            formants[vowel.id] ? (
              <button onClick={() => resetVowel(vowel.id)}
                style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '10px 20px', borderRadius: 10, border: '1.5px solid #e5e7eb', background: '#fff', color: '#666', cursor: 'pointer', fontSize: 13 }}>
                <RotateCcw size={14}/> Repetir
              </button>
            ) : (
              <button onClick={startRecording} disabled={analyzing}
                style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '10px 28px', borderRadius: 10, border: 'none', background: vowel.color, color: '#fff', cursor: 'pointer', fontSize: 14, fontWeight: 600 }}>
                <Mic size={16}/> Grabar
              </button>
            )
          ) : (
            <button onClick={stopRecording}
              style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '10px 28px', borderRadius: 10, border: 'none', background: '#dc2626', color: '#fff', cursor: 'pointer', fontSize: 14, fontWeight: 600, animation: 'pulse 1.2s infinite' }}>
              <Square size={16}/> Parar
            </button>
          )}
        </div>

        {analyzing && <p style={{ textAlign: 'center', color: '#888', fontSize: 13 }} className="animate-pulse">Analizando formantes…</p>}

        {error && (
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, padding: '10px 14px' }}>
            <AlertCircle size={15} color="#dc2626" style={{ flexShrink: 0, marginTop: 1 }}/>
            <span style={{ fontSize: 13, color: '#991b1b' }}>{error}</span>
          </div>
        )}
      </div>

      {/* Plano vocálico */}
      {Object.keys(formants).length > 0 && (
        <div style={{ background: '#fff', border: '1px solid #f0f0f0', borderRadius: 14, padding: 16 }}>
          <p style={{ fontSize: 13, fontWeight: 600, color: '#333', marginBottom: 8 }}>
            Espacio vocálico F1 / F2
          </p>
          <VowelPlot formants={formants}/>
          <p style={{ fontSize: 11, color: '#bbb', marginTop: 4, textAlign: 'center' }}>
            Puntos de color: paciente · Triángulo gris: referencia adulto español
          </p>

          {allDone && (() => {
            const fa = formants.a, fi = formants.i, fu = formants.u
            const vsa = 0.5 * Math.abs(
              fa.F2 * (fi.F1 - fu.F1) + fi.F2 * (fu.F1 - fa.F1) + fu.F2 * (fa.F1 - fi.F1)
            )
            const fcr = (fu.F2 + fa.F2 + fi.F1 + fu.F1) / (fi.F2 + fa.F1)
            return (
              <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
                {[
                  { label: 'VSA',  value: Math.round(vsa).toLocaleString() + ' Hz²', hint: 'Área triángulo vocálico' },
                  { label: 'FCR',  value: fcr.toFixed(3),                             hint: '< 1.17 normal' },
                ].map(m => (
                  <div key={m.label} style={{ flex: 1, background: '#f8fafc', border: '1px solid #e5e7eb', borderRadius: 8, padding: '8px 12px' }}>
                    <div style={{ fontSize: 10, color: '#999', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{m.label}</div>
                    <div style={{ fontSize: 18, fontWeight: 700, color: m.label === 'FCR' && fcr > 1.20 ? '#dc2626' : '#116b70', lineHeight: 1.2 }}>{m.value}</div>
                    <div style={{ fontSize: 10, color: '#bbb' }}>{m.hint}</div>
                  </div>
                ))}
              </div>
            )
          })()}
        </div>
      )}

      {/* Progreso */}
      {!allDone && (
        <p style={{ textAlign: 'center', fontSize: 12, color: '#bbb' }}>
          {VOWELS.filter(v => formants[v.id]).length} / 3 vocales grabadas
        </p>
      )}
    </div>
  )
}
