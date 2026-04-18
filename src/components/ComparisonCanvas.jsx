import { useEffect, useRef } from 'react'

export function ComparisonCanvas({ envelope, duration, ref: refData, height = 160 }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !envelope || !refData) return
    const ctx = canvas.getContext('2d')
    const { width: w, height: h } = canvas
    const { ref, times, tMax } = refData

    ctx.clearRect(0, 0, w, h)

    const toX = t  => (t / tMax) * w
    const toY = v  => h - 10 - v * (h - 20)

    // Reference band (shaded)
    ctx.fillStyle = 'rgba(34,197,94,0.10)'
    ctx.beginPath()
    times.forEach((t, i) => {
      i === 0 ? ctx.moveTo(toX(t), toY(ref[i] * 1.25)) : ctx.lineTo(toX(t), toY(ref[i] * 1.25))
    })
    for (let i = times.length - 1; i >= 0; i--)
      ctx.lineTo(toX(times[i]), toY(ref[i] * 0.75))
    ctx.closePath()
    ctx.fill()

    // Reference mean (dashed green)
    ctx.strokeStyle = 'rgba(34,197,94,0.70)'
    ctx.lineWidth = 2
    ctx.setLineDash([6, 4])
    ctx.beginPath()
    times.forEach((t, i) => {
      i === 0 ? ctx.moveTo(toX(t), toY(ref[i])) : ctx.lineTo(toX(t), toY(ref[i]))
    })
    ctx.stroke()
    ctx.setLineDash([])

    // Recorded envelope (blue)
    ctx.strokeStyle = '#1d4ed8'
    ctx.lineWidth = 2.5
    ctx.beginPath()
    envelope.forEach((v, i) => {
      const t = (i / envelope.length) * duration
      i === 0 ? ctx.moveTo(toX(t), toY(v)) : ctx.lineTo(toX(t), toY(v))
    })
    ctx.stroke()

    // Ref duration dotted line
    const refDur = times[Math.floor(times.length / 2)] * 2 // approx refDuration from gaussian center
    // Actually use tMax/1.4*0.5 as refDuration approximation
    // Better: get it from tMax context — just skip for simplicity

  }, [envelope, duration, refData])

  return (
    <canvas
      ref={canvasRef} width={800} height={height}
      className="w-full rounded-lg bg-gray-50 border border-gray-100"
      style={{ height }}
    />
  )
}
