/* Semi-circle SVG gauge */
export function Gauge({ value, min, max, label, unit }) {
  const cx = 80, cy = 72, r = 60
  const pct    = Math.min(1, Math.max(0, value / max))
  const minPct = Math.min(1, min / max)

  const toXY = (p) => ({
    x: cx + r * Math.cos(Math.PI - p * Math.PI),
    y: cy - r * Math.sin(Math.PI - p * Math.PI) * -1,  // flip for SVG y
  })

  // Arc from left (180°) to right (0°) — d attribute
  const arcD = (from, to) => {
    const a1 = Math.PI - from * Math.PI
    const a2 = Math.PI - to   * Math.PI
    const x1 = cx + r * Math.cos(a1), y1 = cy + r * Math.sin(a1)
    const x2 = cx + r * Math.cos(a2), y2 = cy + r * Math.sin(a2)
    const large = to - from > 0.5 ? 1 : 0
    return `M ${x1.toFixed(1)} ${y1.toFixed(1)} A ${r} ${r} 0 ${large} 1 ${x2.toFixed(1)} ${y2.toFixed(1)}`
  }

  const minAngle = Math.PI - minPct * Math.PI
  const mx = cx + r * Math.cos(minAngle)
  const my = cy + r * Math.sin(minAngle)

  const color = pct * max >= min
    ? '#22c55e'
    : pct * max >= min * 0.7
    ? '#eab308'
    : '#ef4444'

  return (
    <div className="flex flex-col items-center">
      <svg width="160" height="92" viewBox="0 0 160 92">
        {/* Track */}
        <path d={arcD(0, 1)} fill="none" stroke="#e5e7eb" strokeWidth="10" strokeLinecap="round" />
        {/* Value arc */}
        {pct > 0 && (
          <path d={arcD(0, pct)} fill="none" stroke={color} strokeWidth="10" strokeLinecap="round" />
        )}
        {/* Min marker */}
        <circle cx={mx.toFixed(1)} cy={my.toFixed(1)} r="5" fill="#dc2626" />
        {/* Value text */}
        <text x={cx} y={cy + 8} textAnchor="middle" fontSize="17" fontWeight="700" fill="#111827">
          {typeof value === 'number' ? value.toFixed(1) : value}
        </text>
        <text x={cx} y={cy + 22} textAnchor="middle" fontSize="10" fill="#6b7280">{unit}</text>
      </svg>
      <p className="text-xs font-medium text-gray-500 -mt-1">{label}</p>
    </div>
  )
}
