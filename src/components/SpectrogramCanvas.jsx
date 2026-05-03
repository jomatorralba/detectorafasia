import { useEffect, useRef, useState } from 'react'
import { computeSpectrogram, findOnsetFrame, renderPatientSpectrogram } from '../utils/spectrogram'

const MAX_FREQ = 8000

function addAxisLabels(ctx, w, h) {
  ctx.font      = '9px monospace'
  ;[8000, 4000, 2000, 1000, 500].forEach(hz => {
    if (hz > MAX_FREQ) return
    const y = h - Math.floor((hz / MAX_FREQ) * h)
    ctx.fillStyle = 'rgba(255,255,255,0.1)'
    ctx.fillRect(0, y, w, 1)
    ctx.fillStyle = 'rgba(255,255,255,0.4)'
    ctx.fillText(hz >= 1000 ? hz / 1000 + 'k' : String(hz), 3, y - 2)
  })
}

function SpectroPanel({ audioData, label, height }) {
  const canvasRef = useRef(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !audioData) return

    let cancelled = false
    setBusy(true)

    setTimeout(() => {
      try {
        const spec  = computeSpectrogram(audioData.channelData, audioData.sampleRate)
        const onset = findOnsetFrame(spec.frames)
        const img   = renderPatientSpectrogram(spec, onset, canvas.width, canvas.height, MAX_FREQ)
        if (cancelled) return
        const ctx = canvas.getContext('2d')
        ctx.putImageData(img, 0, 0)
        addAxisLabels(ctx, canvas.width, canvas.height)
      } catch (e) {
        console.error(e)
      } finally {
        if (!cancelled) setBusy(false)
      }
    }, 30)

    return () => { cancelled = true }
  }, [audioData])

  return (
    <div style={{ flex: 1, position: 'relative', minWidth: 0 }}>
      <div style={{
        position: 'absolute', top: 5, left: 5, zIndex: 1,
        fontSize: 10, color: 'rgba(255,255,255,0.5)',
        fontFamily: 'monospace', textTransform: 'uppercase', letterSpacing: '0.06em',
        background: 'rgba(0,0,0,0.45)', padding: '1px 6px', borderRadius: 3,
        pointerEvents: 'none',
      }}>{label}</div>
      <canvas
        ref={canvasRef}
        width={400}
        height={height}
        style={{ width: '100%', height, display: 'block' }}
      />
      {busy && (
        <div style={{
          position: 'absolute', inset: 0, display: 'flex',
          alignItems: 'center', justifyContent: 'center',
          background: 'rgba(0,0,0,0.65)',
        }}>
          <span style={{ color: 'rgba(255,255,255,0.7)', fontSize: 12, fontFamily: 'monospace' }}>
            calculando…
          </span>
        </div>
      )}
    </div>
  )
}

export function SpectrogramCanvas({ patientData, referenceData, height = 180 }) {
  return (
    <div style={{
      display: 'flex', gap: 1, background: '#000',
      borderRadius: 12, overflow: 'hidden', height,
    }}>
      {referenceData && (
        <>
          <SpectroPanel audioData={referenceData} label="Referencia" height={height} />
          <div style={{ width: 1, background: 'rgba(255,255,255,0.1)', flexShrink: 0 }} />
        </>
      )}
      <SpectroPanel audioData={patientData} label="Paciente" height={height} />
    </div>
  )
}
