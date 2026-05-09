import { useState, useEffect } from 'react'
import { Save, CheckCircle } from 'lucide-react'
import { PALABRAS, NORM, STATUS, classify } from '../utils/normative'
import { isConfigured } from '../lib/supabase'
import { saveEvaluation, getEvaluations } from '../lib/db'

const ROWS = [
  { key: 'tmf_a',  label: 'TMF /A/',           mode: 'min',   fmt: v => `${v.toFixed(1)} s`,  ref: `≥ ${NORM.tmf_a.min} s` },
  { key: 'tmf_s',  label: 'TMF /S/',           mode: 'min',   fmt: v => `${v.toFixed(1)} s`,  ref: `≥ ${NORM.tmf_s.min} s` },
  { key: 'ratio',  label: 'Cociente S/A',      mode: 'range', fmt: v => v.toFixed(2),          ref: `${NORM.ratio.min} – ${NORM.ratio.max}` },
  { key: 'ddk',    label: 'Diadococinesias',   mode: 'min',   fmt: v => `${v.toFixed(1)} /5s`, ref: `≥ ${NORM.ddk.min} rep/5s` },
  { key: 'wpm',    label: 'Velocidad lectora', mode: 'min',   fmt: v => `${Math.round(v)} ppm`,ref: `≥ ${NORM.wpm.min} ppm` },
]

function SavePanel({ patient, results, onSaved }) {
  const [notes,   setNotes]   = useState('')
  const [saving,  setSaving]  = useState(false)
  const [saved,   setSaved]   = useState(false)
  const [saveErr, setSaveErr] = useState(null)

  const handleSave = async () => {
    setSaving(true); setSaveErr(null)
    try {
      await saveEvaluation(patient.id, results, notes)
      setSaved(true)
      onSaved()
    } catch (e) {
      setSaveErr(e.message)
    } finally {
      setSaving(false)
    }
  }

  if (saved) {
    return (
      <div style={{ background: '#f0fdf4', border: '1px solid #86efac', borderRadius: 12, padding: '14px 18px', display: 'flex', alignItems: 'center', gap: 10 }}>
        <CheckCircle size={18} color="#22c55e"/>
        <span style={{ fontSize: 14, color: '#15803d', fontWeight: 600 }}>
          Evaluación guardada para {patient.name}
        </span>
      </div>
    )
  }

  return (
    <div style={{ background: '#f8f9fb', border: '1px solid #e5e7eb', borderRadius: 12, padding: '16px 18px' }}>
      <h3 style={{ margin: '0 0 12px', fontSize: 14, fontWeight: 600, color: '#073447' }}>
        Guardar evaluación · {patient.name}
      </h3>
      <textarea
        value={notes} onChange={e => setNotes(e.target.value)}
        placeholder="Notas clínicas de la sesión (opcional)…"
        rows={3}
        style={{ width: '100%', padding: '10px 12px', borderRadius: 8, border: '1px solid #d1d5db', fontSize: 13, fontFamily: 'inherit', resize: 'vertical', boxSizing: 'border-box', background: '#fff', outline: 'none' }}
      />
      {saveErr && <p style={{ fontSize: 12, color: '#dc2626', margin: '6px 0 0' }}>{saveErr}</p>}
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 10 }}>
        <button onClick={handleSave} disabled={saving}
          style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '8px 18px', borderRadius: 8, border: 'none', background: '#073447', color: '#fff', cursor: 'pointer', fontSize: 13, fontWeight: 600, opacity: saving ? 0.7 : 1 }}>
          <Save size={14}/>
          {saving ? 'Guardando…' : 'Guardar evaluación'}
        </button>
      </div>
    </div>
  )
}

function EvaluationHistory({ patientId, refreshTrigger }) {
  const [evals,    setEvals]    = useState([])
  const [loading,  setLoading]  = useState(true)
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    setLoading(true)
    getEvaluations(patientId)
      .then(setEvals).catch(console.error).finally(() => setLoading(false))
  }, [patientId, refreshTrigger])

  if (loading || evals.length === 0) return null

  const shown = expanded ? evals : evals.slice(0, 3)

  return (
    <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 12, overflow: 'hidden' }}>
      <div style={{ padding: '11px 16px', background: '#f8f9fb', borderBottom: '1px solid #f0f0f0' }}>
        <span style={{ fontSize: 13, fontWeight: 600, color: '#555' }}>
          Historial · {evals.length} evaluaciones guardadas
        </span>
      </div>
      {shown.map((ev, i) => {
        const date = new Date(ev.date)
        const dateStr = date.toLocaleDateString('es-ES', { day: 'numeric', month: 'short', year: 'numeric' })
        const r = ev.results || {}
        const metrics = [
          r.tmf_a != null && `A: ${r.tmf_a.toFixed(1)}s`,
          r.tmf_s != null && `S: ${r.tmf_s.toFixed(1)}s`,
          r.ddk   != null && `DDK: ${r.ddk.toFixed(1)}`,
          r.wpm   != null && `${Math.round(r.wpm)} ppm`,
        ].filter(Boolean)
        return (
          <div key={ev.id} style={{ padding: '10px 16px', borderBottom: i < shown.length - 1 ? '1px solid #f5f5f5' : 'none', display: 'flex', gap: 12, alignItems: 'flex-start' }}>
            <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#116b70', flexShrink: 0, marginTop: 5 }}/>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: '#333' }}>{dateStr}</div>
              {ev.notes && <p style={{ fontSize: 11, color: '#888', margin: '2px 0 4px', fontStyle: 'italic' }}>"{ ev.notes}"</p>}
              {metrics.length > 0 && (
                <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
                  {metrics.map(m => (
                    <span key={m} style={{ fontSize: 10, background: '#eef7f7', color: '#116b70', padding: '1px 6px', borderRadius: 3, fontFamily: 'monospace', fontWeight: 600 }}>{m}</span>
                  ))}
                </div>
              )}
            </div>
          </div>
        )
      })}
      {evals.length > 3 && (
        <button onClick={() => setExpanded(e => !e)}
          style={{ width: '100%', padding: '8px', background: '#f8f9fb', border: 'none', borderTop: '1px solid #f0f0f0', cursor: 'pointer', fontSize: 12, color: '#888' }}>
          {expanded ? 'Ver menos' : `Ver ${evals.length - 3} más`}
        </button>
      )}
    </div>
  )
}

export function Summary({ results, onReset, patient, onNewPatient }) {
  const [savedFlag, setSavedFlag] = useState(false)
  const hasData = Object.keys(results).length > 0

  if (!hasData) {
    return (
      <div className="text-center py-16 text-gray-400">
        <div className="mb-4 flex justify-center opacity-30">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M9 5H7a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-2"/>
            <rect x="9" y="3" width="6" height="4" rx="1"/>
            <path d="M9 12h6M9 16h4"/>
          </svg>
        </div>
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
                const s = STATUS[classify(results[key], key === 'ratio' ? 'ratio' : key, mode)]
                return (
                  <tr key={key} className="border-t border-gray-50 hover:bg-gray-50/50">
                    <td className="px-5 py-3 font-semibold text-gray-800">{label}</td>
                    <td className="px-5 py-3 font-mono">{fmt(results[key])}</td>
                    <td className="px-5 py-3 text-gray-500">{ref}</td>
                    <td className="px-5 py-3">
                      <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold border ${s.css}`}>
                        <span className={`w-1.5 h-1.5 rounded-full ${s.dot} shrink-0`}/>
                        {s.label}
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
                      {ratio <= 1.5
                        ? <span className="inline-flex items-center gap-1 text-green-700 text-xs font-medium"><span className="w-1.5 h-1.5 rounded-full bg-green-400 shrink-0"/> Normal</span>
                        : <span className="inline-flex items-center gap-1 text-red-700 text-xs font-medium"><span className="w-1.5 h-1.5 rounded-full bg-red-400 shrink-0"/> Lento</span>}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {patient && isConfigured && (
        <SavePanel patient={patient} results={results} onSaved={() => setSavedFlag(true)}/>
      )}

      {patient && isConfigured && (
        <EvaluationHistory patientId={patient.id} refreshTrigger={savedFlag}/>
      )}

      <div style={{ display: 'flex', gap: 10, justifyContent: 'center', flexWrap: 'wrap', paddingTop: 4 }}>
        <button onClick={onReset}
          className="px-6 py-2.5 rounded-xl border-2 border-gray-200 text-gray-600 hover:border-gray-400 hover:text-gray-800 font-medium text-sm transition">
          Nueva evaluación
        </button>
        {onNewPatient && (
          <button onClick={onNewPatient}
            className="px-6 py-2.5 rounded-xl border-2 border-gray-200 text-gray-500 hover:border-blue-300 hover:text-blue-700 font-medium text-sm transition">
            Cambiar paciente
          </button>
        )}
      </div>
    </div>
  )
}
