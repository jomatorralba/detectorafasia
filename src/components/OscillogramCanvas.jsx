import { useEffect, useRef, useState } from 'react'
import { computeSpectrogram, findOnsetFrame } from '../utils/spectrogram'

const HOP_SIZE   = 128
const PREROLL_MS = 20

function getOnsetSample(channelData, sampleRate) {
  const spec    = computeSpectrogram(channelData, sampleRate, 512, HOP_SIZE)
  const frame   = findOnsetFrame(spec.frames)
  const preroll = Math.floor((PREROLL_MS / 1000) * sampleRate)
  return Math.max(0, frame * HOP_SIZE - preroll)
}

export function OscillogramCanvas({ patientData, referenceData, height = 100 }) {
  const canvasRef = useRef(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !patientData) return

    let cancelled = false
    setBusy(true)

    setTimeout(() => {
      try {
        const ctx = canvas.getContext('2d')
        const { width: w, height: h } = canvas

        ctx.clearRect(0, 0, w, h)
        ctx.fillStyle = '#070a0f'
        ctx.fillRect(0, 0, w, h)

        // Zero line
        ctx.strokeStyle = 'rgba(255,255,255,0.07)'
        ctx.lineWidth = 1
        ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke()

        const patOnset  = getOnsetSample(patientData.channelData, patientData.sampleRate)
        const patSlice  = patientData.channelData.slice(patOnset)
        let   maxLen    = patSlice.length
        let   refSlice  = null

        if (referenceData) {
          const refOnset = getOnsetSample(referenceData.channelData, referenceData.sampleRate)
          refSlice = referenceData.channelData.slice(refOnset)
          maxLen   = Math.max(maxLen, refSlice.length)

          // Reference — cyan
          ctx.strokeStyle = 'rgba(34,211,238,0.5)'
          ctx.lineWidth   = 1.5
          ctx.beginPath()
          for (let i = 0; i < refSlice.length; i++) {
            const x = (i / maxLen) * w
            const y = (0.5 - refSlice[i] * 0.44) * h
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
          }
          ctx.stroke()
        }

        // Patient — amber
        ctx.strokeStyle = 'rgba(251,146,60,0.85)'
        ctx.lineWidth   = 1.5
        ctx.beginPath()
        for (let i = 0; i < patSlice.length; i++) {
          const x = (i / maxLen) * w
          const y = (0.5 - patSlice[i] * 0.44) * h
          i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
        }
        ctx.stroke()
      } finally {
        if (!cancelled) setBusy(false)
      }
    }, 10)

    return () => { cancelled = true }
  }, [patientData, referenceData])

  return (
    <div style={{ position: 'relative', background: '#070a0f', borderRadius: 10, overflow: 'hidden', height }}>
      <canvas
        ref={canvasRef}
        width={800}
        height={height}
        style={{ width: '100%', height, display: 'block' }}
      />
      {busy && (
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <span style={{ color: 'rgba(255,255,255,0.35)', fontSize: 11, fontFamily: 'monospace' }}>…</span>
        </div>
      )}
      <div style={{
        position: 'absolute', bottom: 6, right: 8,
        display: 'flex', gap: 12, fontSize: 10,
        color: 'rgba(255,255,255,0.45)', fontFamily: 'monospace',
      }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ width: 14, height: 2, background: 'rgba(251,146,60,0.85)', display: 'inline-block', borderRadius: 1 }} />
          Paciente
        </span>
        {referenceData && (
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ width: 14, height: 2, background: 'rgba(34,211,238,0.5)', display: 'inline-block', borderRadius: 1 }} />
            Referencia
          </span>
        )}
      </div>
    </div>
  )
}
