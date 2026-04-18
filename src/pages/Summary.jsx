import { PALABRAS, NORM, STATUS, classify } from '../utils/normative'

const ROWS = [
  { key: 'tmf_a',  label: 'TMF /A/',           mode: 'min',   fmt: v => `${v.toFixed(1)} s`,  ref: `≥ ${NORM.tmf_a.min} s` },
  { key: 'tmf_s',  label: 'TMF /S/',           mode: 'min',   fmt: v => `${v.toFixed(1)} s`,  ref: `≥ ${NORM.tmf_s.min} s` },
  { key: 'ratio',  label: 'Cociente S/A',      mode: 'range', fmt: v => v.toFixed(2),          ref: `${NORM.ratio.min} – ${NORM.ratio.max}` },
  { key: 'ddk',    label: 'Diadococinesias',   mode: 'min',   fmt: v => `${v.toFixed(1)} /5s`, ref: `≥ ${NORM.ddk.min} rep/5s` },
  { key: 'wpm',    label: 'Velocidad lectora', mode: 'min',   fmt: v => `${Math.round(v)} ppm`,ref: `≥ ${NORM.wpm.min} ppm` },
]

export function Summary({ results, onReset }) {
  const hasData = Object.keys(results).length > 0

  if (!hasData) {
    return (
      <div className="text-center py-16 text-gray-400">
        <div className="text-4xl mb-4">📋</div>
        <p className="text-lg font-medium">Aún no hay resultados</p>
        <p className="text-sm mt-1">Completa al menos una prueba para ver el resumen.</p>
      </div>
    )
  }

  const mainRows = ROWS.filter(r => results[r.key] != null)
  const palabrasEntries = Object.entries(results)
    .filter(([k]) => k.startsWith('palabra_'))
    .map(([k, v]) => [k.replace('palabra_', ''), v])

  return (
    <div className="space-y-6">
      {mainRows.length > 0 && (
        <div className="bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 text-xs text-gray-500 uppercase">
              <tr>
                {['Prueba', 'Resultado', 'Referencia', 'Estado'].map(h => (
                  <th key={h} className="px-5 py-3 text-left font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {mainRows.map(({ key, label, mode, fmt, ref }) => {
                const normKey = key === 'ratio' ? 'ratio' : key
                const s = STATUS[classify(results[key], normKey, mode)]
                return (
                  <tr key={key} className="border-t border-gray-50 hover:bg-gray-50/50">
                    <td className="px-5 py-3 font-semibold text-gray-800">{label}</td>
                    <td className="px-5 py-3 font-mono">{fmt(results[key])}</td>
                    <td className="px-5 py-3 text-gray-500">{ref}</td>
                    <td className="px-5 py-3">
                      <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5
                        rounded-full text-xs font-semibold border ${s.css}`}>
                        {s.icon} {s.label}
                      </span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {palabrasEntries.length > 0 && (
        <div className="bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden">
          <div className="px-5 py-3 border-b border-gray-100">
            <h3 className="font-semibold text-gray-800">Lectura de palabras</h3>
          </div>
          <table className="w-full text-sm">
            <thead className="bg-gray-50 text-xs text-gray-500 uppercase">
              <tr>
                {['Palabra', 'Duración', 'Referencia', 'Ratio', 'Estado'].map(h => (
                  <th key={h} className="px-5 py-2 text-left font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {palabrasEntries.map(([w, d]) => {
                const ratio = d / PALABRAS[w]?.ref
                return (
                  <tr key={w} className="border-t border-gray-50 hover:bg-gray-50/50">
                    <td className="px-5 py-2 font-semibold">{w}</td>
                    <td className="px-5 py-2 font-mono">{d.toFixed(2)} s</td>
                    <td className="px-5 py-2 text-gray-500">{PALABRAS[w]?.ref} s</td>
                    <td className="px-5 py-2 font-mono">{ratio?.toFixed(2)}×</td>
                    <td className="px-5 py-2">
                      {ratio <= 1.5 ? '✅ Normal' : '⚠️ Lento'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      <div className="flex justify-center pt-2">
        <button
          onClick={onReset}
          className="px-6 py-2.5 rounded-xl border-2 border-gray-200 text-gray-600
                     hover:border-red-300 hover:text-red-600 font-medium text-sm transition"
        >
          🔄 Nueva evaluación
        </button>
      </div>
    </div>
  )
}
