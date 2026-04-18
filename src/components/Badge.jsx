import { STATUS } from '../utils/normative'

export function Badge({ label, value, status, refText }) {
  const s = STATUS[status]
  return (
    <div className={`border-2 rounded-xl p-4 text-center ${s.css}`}>
      <div className="text-2xl font-bold">{s.icon} {value}</div>
      <div className="text-sm mt-0.5">{label} · <strong>{s.label}</strong></div>
      <div className="text-xs opacity-60 mt-1">Referencia: {refText}</div>
    </div>
  )
}
