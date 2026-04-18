import { useEffect, useRef } from 'react'

/* Static waveform after recording */
export function Waveform({ points, duration, onsetTimes = [], color = '#1565C0', height = 90 }) {
  const ref = useRef(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas || !points?.length) return
    const ctx = canvas.getContext('2d')
    const { width: w, height: h } = canvas

    ctx.clearRect(0, 0, w, h)

    // Zero line
    ctx.strokeStyle = '#e5e7eb'
    ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke()

    // Waveform
    ctx.strokeStyle = color
    ctx.lineWidth = 1.5
    ctx.beginPath()
    points.forEach((p, i) => {
      const x = (p.t / duration) * w
      const y = (0.5 - p.v * 0.44) * h
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
    })
    ctx.stroke()

    // Onset markers
    ctx.strokeStyle = 'rgba(239,83,80,0.55)'
    ctx.lineWidth = 1.5
    onsetTimes.forEach(t => {
      const x = (t / duration) * w
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke()
    })
  }, [points, duration, onsetTimes, color])

  return (
    <canvas
      ref={ref} width={800} height={height}
      className="w-full rounded-lg bg-gray-50 border border-gray-100"
      style={{ height }}
    />
  )
}

/* Live waveform during recording — reads from analyserRef via RAF */
export function LiveWaveform({ analyserRef }) {
  const canvasRef = useRef(null)
  const rafRef    = useRef(null)

  useEffect(() => {
    const draw = () => {
      rafRef.current = requestAnimationFrame(draw)
      const analyser = analyserRef.current
      const canvas   = canvasRef.current
      if (!analyser || !canvas) return

      const data = new Float32Array(analyser.frequencyBinCount)
      analyser.getFloatTimeDomainData(data)

      const ctx = canvas.getContext('2d')
      const { width: w, height: h } = canvas
      ctx.clearRect(0, 0, w, h)

      ctx.strokeStyle = '#ef4444'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      const sw = w / data.length
      data.forEach((v, i) => {
        const x = i * sw
        const y = (0.5 - v * 0.44) * h
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      })
      ctx.stroke()
    }

    rafRef.current = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(rafRef.current)
  }, []) // accesses refs, no deps needed

  return (
    <canvas
      ref={canvasRef} width={800} height={70}
      className="w-full rounded-lg bg-gray-50 border border-gray-100"
      style={{ height: 70 }}
    />
  )
}
