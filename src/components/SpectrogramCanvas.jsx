import { useEffect, useRef, useState } from 'react'
import {
  computeSpectrogram,
  findOnsetFrame,
  blendSpectrograms,
  renderPatientSpectrogram,
} from '../utils/spectrogram'

const MAX_FREQ = 8000

export function SpectrogramCanvas({ patientData, referenceData, height = 200 }) {
  const canvasRef = useRef(null)
  const [busy, setBusy]   = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !patientData) return

    let cancelled = false
    setBusy(true)
    setError(null)

    // Run in a macro task so the spinner renders first
    setTimeout(() => {
      try {
        const { channelData, sampleRate } = patientData
        const patSpec   = computeSpectrogram(channelData, sampleRate)
        const patOnset  = findOnsetFrame(patSpec.frames)

        let imageData
        if (referenceData) {
          const refSpec  = computeSpectrogram(referenceData.channelData, referenceData.sampleRate)
          const refOnset = findOnsetFrame(refSpec.frames)
          imageData = blendSpectrograms(patSpec, patOnset, refSpec, refOnset, canvas.width, canvas.height, MAX_FREQ)
        } else {
          imageData = renderPatientSpectrogram(patSpec, patOnset, canvas.width, canvas.height, MAX_FREQ)
        }

        if (cancelled) return
        canvas.getContext('2d').putImageData(imageData, 0, 0)

        // Frequency axis labels
        const ctx = canvas.getContext('2d')
        ctx.fillStyle = 'rgba(255,255,255,0.55)'
        ctx.font = '10px monospace'
        ;[8000, 4000, 2000, 1000, 500].forEach(hz => {
          if (hz > MAX_FREQ) return
          const y = canvas.height - Math.floor((hz / MAX_FREQ) * canvas.height)
          ctx.fillText(`${hz >= 1000 ? hz/1000 + 'k' : hz}`, 4, y - 2)
          ctx.fillStyle = 'rgba(255,255,255,0.15)'
          ctx.fillRect(0, y, canvas.width, 1)
          ctx.fillStyle = 'rgba(255,255,255,0.55)'
        })
      } catch (e) {
        if (!cancelled) setError(e.message)
      } finally {
        if (!cancelled) setBusy(false)
      }
    }, 30)

    return () => { cancelled = true }
  }, [patientData, referenceData])

  return (
    <div className="relative rounded-xl overflow-hidden" style={{ background: '#000', height }}>
      <canvas
        ref={canvasRef}
        width={800}
        height={height}
        className="w-full"
        style={{ height, display: 'block' }}
      />

      {busy && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/70">
          <span className="text-white text-sm animate-pulse">Calculando espectrograma…</span>
        </div>
      )}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80">
          <span className="text-red-400 text-sm">Error: {error}</span>
        </div>
      )}

      {/* Legend */}
      <div className="absolute bottom-2 right-2 flex gap-3 text-[10px]">
        <span className="flex items-center gap-1 text-white/60">
          <span className="w-3 h-2 rounded-sm inline-block" style={{ background: 'linear-gradient(to right,#000,#f97316)' }} />
          Paciente
        </span>
        {referenceData && (
          <span className="flex items-center gap-1 text-white/60">
            <span className="w-3 h-2 rounded-sm inline-block" style={{ background: 'linear-gradient(to right,#000,#22d3ee)' }} />
            Referencia
          </span>
        )}
      </div>
    </div>
  )
}
