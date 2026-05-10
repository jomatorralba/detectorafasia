import { useState, useEffect } from 'react'
import { Search, UserPlus, ChevronRight, Trash2, X, ClipboardList } from 'lucide-react'
import { getPatients, createPatient, deletePatient, getEvaluations, getEvaluationSummaries } from '../lib/db'
import { isConfigured } from '../lib/supabase'
import { NiLogo } from '../components/NiLogo'

function getInitials(name) {
  return name.split(' ').filter(Boolean).map(w => w[0]).join('').slice(0, 2).toUpperCase()
}

function getAge(birthDate) {
  if (!birthDate) return null
  const d = new Date(birthDate)
  const now = new Date()
  let age = now.getFullYear() - d.getFullYear()
  const m = now.getMonth() - d.getMonth()
  if (m < 0 || (m === 0 && now.getDate() < d.getDate())) age--
  return age
}

function timeAgo(dateStr) {
  const days = Math.floor((Date.now() - new Date(dateStr).getTime()) / 86400000)
  if (days === 0) return 'Hoy'
  if (days === 1) return 'Ayer'
  if (days < 7)  return `Hace ${days} días`
  if (days < 30) return `Hace ${Math.floor(days / 7)} semanas`
  if (days < 365) return `Hace ${Math.floor(days / 30)} meses`
  return `Hace ${Math.floor(days / 365)} años`
}

function NotConfiguredScreen({ onContinue }) {
  return (
    <div style={{
      minHeight: '100vh', background: '#f4f6f8', display: 'flex',
      flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      gap: 28, padding: 32, textAlign: 'center',
    }}>
      <NiLogo size={52} color="#116b70"/>
      <div style={{ maxWidth: 460 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: '#073447', margin: '0 0 10px' }}>
          Evaluación de Disartria
        </h1>
        <p style={{ color: '#666', fontSize: 14, lineHeight: 1.7, margin: 0 }}>
          Para gestionar pacientes y guardar evaluaciones conecta Supabase (gratuito).
          Sin base de datos la app funciona localmente sin guardar nada.
        </p>
      </div>

      <div style={{
        background: '#fff', borderRadius: 14, border: '1px solid #e5e7eb',
        padding: '20px 24px', maxWidth: 480, width: '100%', textAlign: 'left',
      }}>
        <p style={{ fontSize: 13, fontWeight: 600, color: '#073447', margin: '0 0 10px' }}>
          Configurar Supabase (gratis, 5 min)
        </p>
        <ol style={{ fontSize: 12, color: '#555', lineHeight: 2.1, paddingLeft: 18, margin: '0 0 10px' }}>
          <li>Crea un proyecto en <strong>supabase.com</strong></li>
          <li>Copia tu <strong>URL</strong> y <strong>anon key</strong> (Settings → API)</li>
          <li>Crea <code style={{ background: '#f0f0f0', padding: '1px 5px', borderRadius: 3 }}>.env.local</code> en la raíz con:</li>
        </ol>
        <pre style={{
          background: '#f8f9fb', borderRadius: 8, padding: '10px 14px',
          fontSize: 11, color: '#073447', margin: '0 0 10px', overflow: 'auto',
        }}>
{`VITE_SUPABASE_URL=https://xxxx.supabase.co
VITE_SUPABASE_ANON_KEY=eyJhbGc...`}
        </pre>
        <p style={{ fontSize: 11, color: '#888', margin: 0 }}>
          Luego ejecuta el SQL de <code style={{ background: '#f0f0f0', padding: '1px 5px', borderRadius: 3 }}>supabase/schema.sql</code> en el editor SQL de Supabase.
        </p>
      </div>

      <button
        onClick={onContinue}
        style={{
          padding: '10px 28px', borderRadius: 10, border: '1.5px solid #d1d5db',
          background: '#fff', color: '#555', cursor: 'pointer', fontSize: 14, fontWeight: 500,
        }}
      >
        Continuar sin base de datos
      </button>
    </div>
  )
}

function fmt1(v, unit = '') { return v != null ? `${v.toFixed(1)}${unit}` : '—' }
function fmt0(v, unit = '') { return v != null ? `${Math.round(v)}${unit}` : '—' }

function Chip({ label }) {
  return (
    <span style={{ fontSize: 10, background: '#eef7f7', color: '#116b70', padding: '1px 7px', borderRadius: 4, fontFamily: 'monospace', fontWeight: 600 }}>
      {label}
    </span>
  )
}

function MetricRow({ label, value, good, warn }) {
  const dot = value === '—' ? '#ddd' : good ? '#16a34a' : warn ? '#f59e0b' : '#ef4444'
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 11.5, padding: '2px 0', gap: 8 }}>
      <span style={{ color: '#777' }}>{label}</span>
      <span style={{ fontFamily: 'monospace', fontWeight: 600, color: '#222', display: 'flex', alignItems: 'center', gap: 4, flexShrink: 0 }}>
        <span style={{ width: 5, height: 5, borderRadius: '50%', background: dot, display: 'inline-block', flexShrink: 0 }}/>
        {value}
      </span>
    </div>
  )
}

function EvalDetail({ r }) {
  const vq = r.voice_quality || {}
  const ps = r.pause_stats   || {}
  const pr = r.prosody       || {}
  const fo = r.formants      || {}
  const hasAero    = r.tmf_a != null || r.tmf_s != null || r.ddk != null
  const hasVQ      = vq.jitter != null
  const hasWPM     = r.wpm != null
  const hasProsody = pr.f0_mean != null
  const hasFormants= fo.fcr != null

  const cols = []
  if (hasAero) cols.push(
    <div key="aero">
      <div style={{ fontSize: 9.5, fontWeight: 700, color: '#999', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>Aerodinámica</div>
      {r.tmf_a != null && <MetricRow label="TMF /a/" value={fmt1(r.tmf_a, 's')} good={r.tmf_a >= 15} warn={r.tmf_a >= 10}/>}
      {r.tmf_s != null && <MetricRow label="TMF /s/" value={fmt1(r.tmf_s, 's')} good={r.tmf_s >= 20} warn={r.tmf_s >= 14}/>}
      {r.tmf_a != null && r.tmf_s != null && (
        <MetricRow label="Cociente S/A" value={fmt1(r.tmf_s / r.tmf_a)} good={r.tmf_s/r.tmf_a >= 0.85} warn={r.tmf_s/r.tmf_a >= 0.7}/>
      )}
      {r.ddk != null && <MetricRow label="DDK (rep/s)" value={fmt1(r.ddk)} good={r.ddk >= 5} warn={r.ddk >= 4}/>}
    </div>
  )
  if (hasVQ) cols.push(
    <div key="vq">
      <div style={{ fontSize: 9.5, fontWeight: 700, color: '#999', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>Calidad vocal</div>
      <MetricRow label="Jitter (%)" value={fmt1(vq.jitter)} good={vq.jitter < 1.04} warn={vq.jitter < 1.5}/>
      <MetricRow label="Shimmer (%)" value={fmt1(vq.shimmer)} good={vq.shimmer < 3.81} warn={vq.shimmer < 5}/>
      <MetricRow label="HNR (dB)" value={fmt1(vq.hnr)} good={vq.hnr >= 20} warn={vq.hnr >= 15}/>
    </div>
  )
  if (hasWPM) cols.push(
    <div key="lectura">
      <div style={{ fontSize: 9.5, fontWeight: 700, color: '#999', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>Lectura</div>
      <MetricRow label="WPM" value={fmt0(r.wpm)} good={r.wpm >= 120} warn={r.wpm >= 80}/>
      {ps.silence_pct != null && <>
        <MetricRow label="Silencio" value={fmt0(ps.silence_pct * 100, '%')} good={ps.silence_pct >= 0.15 && ps.silence_pct <= 0.25} warn={ps.silence_pct <= 0.35}/>
        <MetricRow label="Art. (sil/s)" value={fmt1(ps.articulation_rate)} good={ps.articulation_rate >= 4.5} warn={ps.articulation_rate >= 3.5}/>
        <MetricRow label="Pausa media" value={fmt1(ps.mean_pause_dur, 's')} good={ps.mean_pause_dur <= 0.6} warn={ps.mean_pause_dur <= 1}/>
      </>}
    </div>
  )
  if (hasProsody) cols.push(
    <div key="prosody">
      <div style={{ fontSize: 9.5, fontWeight: 700, color: '#999', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>Prosodia</div>
      <MetricRow label="F0 medio (Hz)" value={fmt0(pr.f0_mean)} good warn/>
      <MetricRow label="Rango F0 (Hz)" value={fmt0(pr.f0_range)} good={pr.f0_range >= 50} warn={pr.f0_range >= 25}/>
      {pr.decl_falls != null && <MetricRow label="Decl. desciende" value={pr.decl_falls ? 'Sí' : 'No'} good={pr.decl_falls}/>}
      {pr.int_rises  != null && <MetricRow label="Int. asciende"   value={pr.int_rises  ? 'Sí' : 'No'} good={pr.int_rises}/>}
    </div>
  )
  if (hasFormants) cols.push(
    <div key="formants">
      <div style={{ fontSize: 9.5, fontWeight: 700, color: '#999', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>Espacio vocálico</div>
      {fo.a && <MetricRow label="/a/ F1·F2" value={`${Math.round(fo.a.F1)}·${Math.round(fo.a.F2)}`} good warn/>}
      {fo.i && <MetricRow label="/i/ F1·F2" value={`${Math.round(fo.i.F1)}·${Math.round(fo.i.F2)}`} good warn/>}
      {fo.u && <MetricRow label="/u/ F1·F2" value={`${Math.round(fo.u.F1)}·${Math.round(fo.u.F2)}`} good warn/>}
      <MetricRow label="VSA (Hz²)" value={fo.vsa != null ? Math.round(fo.vsa).toLocaleString('es') : '—'} good warn/>
      <MetricRow label="FCR" value={fmt1(fo.fcr)} good={fo.fcr <= 1.17} warn={fo.fcr <= 1.25}/>
    </div>
  )

  if (cols.length === 0) return <p style={{ fontSize: 12, color: '#aaa', margin: 0 }}>Sin datos detallados.</p>

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: '10px 20px' }}>
      {cols}
    </div>
  )
}

function EvolutionChart({ evals }) {
  const VW = 600, VH = 110, pad = { t: 10, r: 12, b: 28, l: 12 }
  const cW = VW - pad.l - pad.r, cH = VH - pad.t - pad.b
  const n = evals.length

  const LINES = [
    { label: 'TMF/A',  color: '#116b70', get: r => r.tmf_a },
    { label: 'WPM',    color: '#6366f1', get: r => r.wpm },
    { label: 'HNR',    color: '#f59e0b', get: r => r.voice_quality?.hnr },
    { label: 'Art.',   color: '#10b981', get: r => r.pause_stats?.articulation_rate },
    { label: 'FCR↓',   color: '#ef4444', get: r => r.formants ? 2 - r.formants.fcr : null },
  ]

  const lines = LINES.map(m => {
    const vals = evals.map(ev => m.get(ev.results || {}))
    const valid = vals.filter(v => v != null && isFinite(v))
    if (valid.length < 1) return null
    const mn = Math.min(...valid), mx = Math.max(...valid), range = mx - mn || 1
    const pts = vals.map((v, i) => {
      if (v == null || !isFinite(v)) return null
      return {
        x: +(pad.l + (n > 1 ? (i / (n - 1)) * cW : cW / 2)).toFixed(1),
        y: +(pad.t + cH - ((v - mn) / range) * cH).toFixed(1),
      }
    })
    const d = pts.reduce((acc, p) => {
      if (!p) return acc
      return acc === '' ? `M${p.x},${p.y}` : `${acc} L${p.x},${p.y}`
    }, '')
    return { ...m, pts, d }
  }).filter(Boolean)

  if (lines.length === 0) return null

  const dates = evals.map(ev => {
    const d = new Date(ev.date)
    return `${d.getDate()}/${d.getMonth() + 1}`
  })

  return (
    <div style={{ padding: '14px 20px 0', borderBottom: '1px solid #f0f0f0' }}>
      <div style={{ fontSize: 10, fontWeight: 700, color: '#aaa', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 6 }}>
        Evolución temporal
      </div>
      <svg viewBox={`0 0 ${VW} ${VH}`} width="100%" style={{ display: 'block' }} preserveAspectRatio="none">
        {[0, 0.5, 1].map(t => (
          <line key={t} x1={pad.l} x2={pad.l + cW} y1={pad.t + t * cH} y2={pad.t + t * cH}
            stroke="#f0f0f0" strokeWidth="1"/>
        ))}
        {lines.map(l => (
          <path key={l.label} d={l.d} fill="none" stroke={l.color} strokeWidth="1.8" strokeLinejoin="round" strokeLinecap="round"/>
        ))}
        {lines.map(l => l.pts.map((p, i) => p && (
          <circle key={`${l.label}-${i}`} cx={p.x} cy={p.y} r="3" fill={l.color}/>
        )))}
        {dates.map((d, i) => (
          <text key={i}
            x={pad.l + (n > 1 ? (i / (n - 1)) * cW : cW / 2)}
            y={VH - 6} textAnchor="middle" fontSize="9" fill="#bbb">
            {d}
          </text>
        ))}
      </svg>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', padding: '6px 0 12px', fontSize: 10, color: '#aaa', alignItems: 'center' }}>
        {lines.map(l => (
          <span key={l.label} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ display: 'inline-block', width: 14, height: 2, background: l.color, borderRadius: 1 }}/>
            {l.label}
          </span>
        ))}
        <span style={{ marginLeft: 'auto', fontSize: 9 }}>Valores normalizados · FCR↓ = mejor al bajar</span>
      </div>
    </div>
  )
}

function ReferencesFooter() {
  return (
    <div style={{ padding: '10px 20px 14px', borderTop: '1px solid #f5f5f5' }}>
      <div style={{ fontSize: 9, color: '#c0c0c0', lineHeight: 1.9, letterSpacing: '0.01em' }}>
        <span style={{ fontWeight: 700, color: '#bbb' }}>Referencias · </span>
        Titze IR (1995) Jitter & Shimmer ·{' '}
        Boersma P (1993) HNR por autocorrelación ·{' '}
        Tjaden K & Wilding GE (2004) Pausas en disartria ·{' '}
        Tsao Y & Weismer G (1997) Tasa de articulación ·{' '}
        Patel R (2002) Prosodia clínica ·{' '}
        Sapir S et al. (2010) VSA & FCR en Parkinson ·{' '}
        Hillenbrand J et al. (1995) Formantes vocálicos
      </div>
    </div>
  )
}

function HistoryModal({ patient, onClose, onEvaluate }) {
  const [evals, setEvals] = useState([])
  const [loading, setLoading] = useState(true)
  const [expandedId, setExpandedId] = useState(null)

  useEffect(() => {
    getEvaluations(patient.id).then(setEvals).catch(console.error).finally(() => setLoading(false))
  }, [patient.id])

  const chronological = [...evals].reverse()

  return (
    <div
      style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.45)', zIndex: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20 }}
      onClick={onClose}
    >
      <div
        style={{ background: '#fff', borderRadius: 16, width: '100%', maxWidth: 680, maxHeight: '88vh', display: 'flex', flexDirection: 'column', overflow: 'hidden', boxShadow: '0 20px 60px rgba(0,0,0,0.2)' }}
        onClick={e => e.stopPropagation()}
      >
        <div style={{ padding: '16px 20px', borderBottom: '1px solid #f0f0f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexShrink: 0 }}>
          <div>
            <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: '#111' }}>{patient.name}</h3>
            <p style={{ margin: 0, fontSize: 12, color: '#888' }}>{evals.length} evaluaciones guardadas</p>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              onClick={() => { onClose(); onEvaluate(patient) }}
              style={{ padding: '7px 14px', borderRadius: 8, border: 'none', background: '#073447', color: '#fff', cursor: 'pointer', fontSize: 13, fontWeight: 600 }}
            >
              Nueva evaluación
            </button>
            <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#aaa', padding: 4 }}>
              <X size={18}/>
            </button>
          </div>
        </div>

        <div style={{ flex: 1, overflowY: 'auto' }}>
          {loading ? (
            <p style={{ textAlign: 'center', padding: 32, color: '#aaa' }}>Cargando…</p>
          ) : evals.length === 0 ? (
            <p style={{ textAlign: 'center', padding: 32, color: '#aaa' }}>Sin evaluaciones guardadas todavía.</p>
          ) : (
            <>
              {chronological.length > 1 && <EvolutionChart evals={chronological}/>}

              {evals.map((ev, i) => {
                const date = new Date(ev.date)
                const dateStr = date.toLocaleDateString('es-ES', { day: 'numeric', month: 'short', year: 'numeric' })
                const timeStr = date.toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit' })
                const r = ev.results || {}
                const isExp = expandedId === ev.id
                return (
                  <div key={ev.id} style={{ borderBottom: i < evals.length - 1 ? '1px solid #f5f5f5' : 'none' }}>
                    <button
                      onClick={() => setExpandedId(isExp ? null : ev.id)}
                      style={{ width: '100%', display: 'flex', gap: 12, alignItems: 'flex-start', padding: '11px 20px', background: 'none', border: 'none', cursor: 'pointer', textAlign: 'left' }}
                    >
                      <div style={{ width: 7, height: 7, borderRadius: '50%', background: '#116b70', flexShrink: 0, marginTop: 5 }}/>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: 13, fontWeight: 600, color: '#222' }}>{dateStr} · {timeStr}</div>
                        {ev.notes && <p style={{ fontSize: 12, color: '#777', margin: '2px 0 5px', fontStyle: 'italic' }}>"{ev.notes}"</p>}
                        <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap', marginTop: 4 }}>
                          {r.tmf_a            != null && <Chip label={`TMF ${r.tmf_a.toFixed(1)}s`}/>}
                          {r.ddk              != null && <Chip label={`DDK ${r.ddk.toFixed(1)}`}/>}
                          {r.wpm              != null && <Chip label={`${Math.round(r.wpm)} ppm`}/>}
                          {r.voice_quality?.hnr != null && <Chip label={`HNR ${r.voice_quality.hnr.toFixed(1)}dB`}/>}
                          {r.prosody?.f0_mean  != null && <Chip label={`F0 ${Math.round(r.prosody.f0_mean)}Hz`}/>}
                          {r.formants?.fcr     != null && <Chip label={`FCR ${r.formants.fcr.toFixed(2)}`}/>}
                        </div>
                      </div>
                      <span style={{ fontSize: 10, color: '#ccc', flexShrink: 0, marginTop: 3 }}>{isExp ? '▲' : '▼'}</span>
                    </button>

                    {isExp && (
                      <div style={{ padding: '4px 20px 14px 39px' }}>
                        <EvalDetail r={r}/>
                      </div>
                    )}
                  </div>
                )
              })}

              <ReferencesFooter/>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

const AVATAR_HUES = [210, 160, 280, 30, 340, 190, 60]

function PatientCard({ patient, evalCount, lastEval, onEvaluate, onHistory, onDelete }) {
  const age      = getAge(patient.birth_date)
  const initials = getInitials(patient.name)
  const hue      = AVATAR_HUES[patient.name.charCodeAt(0) % AVATAR_HUES.length]

  return (
    <div
      style={{ background: '#fff', borderRadius: 12, border: '1px solid #e8ecf0', padding: '13px 16px', display: 'flex', gap: 14, alignItems: 'center', transition: 'box-shadow 0.15s' }}
      onMouseEnter={e => e.currentTarget.style.boxShadow = '0 2px 14px rgba(0,0,0,0.08)'}
      onMouseLeave={e => e.currentTarget.style.boxShadow = 'none'}
    >
      <div style={{ width: 44, height: 44, borderRadius: '50%', flexShrink: 0, background: `hsl(${hue},35%,88%)`, color: `hsl(${hue},40%,32%)`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 15, fontWeight: 700 }}>
        {initials}
      </div>

      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontWeight: 600, fontSize: 14, color: '#111', lineHeight: 1.3 }}>
          {patient.name}
          {age != null && <span style={{ fontWeight: 400, color: '#888', marginLeft: 6 }}>{age} años</span>}
        </div>
        <div style={{ fontSize: 12, color: '#999', marginTop: 2 }}>
          {patient.diagnosis || 'Sin diagnóstico'}
          {lastEval  && <span> · {timeAgo(lastEval)}</span>}
          {evalCount > 0 && <span> · {evalCount} eval.</span>}
        </div>
      </div>

      <div style={{ display: 'flex', gap: 6, flexShrink: 0, alignItems: 'center' }}>
        {evalCount > 0 && (
          <button onClick={onHistory} title="Ver historial"
            style={{ padding: '6px 10px', borderRadius: 7, border: '1px solid #e8ecf0', background: '#fff', color: '#555', cursor: 'pointer', fontSize: 12, display: 'flex', alignItems: 'center', gap: 4 }}>
            <ClipboardList size={13}/> {evalCount}
          </button>
        )}
        <button onClick={onDelete} title="Eliminar paciente"
          style={{ padding: '6px 8px', borderRadius: 7, border: '1px solid #fecaca', background: '#fff', color: '#dc2626', cursor: 'pointer', display: 'flex', alignItems: 'center' }}>
          <Trash2 size={13}/>
        </button>
        <button onClick={onEvaluate}
          style={{ padding: '7px 14px', borderRadius: 8, border: 'none', background: '#073447', color: '#fff', cursor: 'pointer', fontSize: 13, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 5 }}>
          Evaluar <ChevronRight size={14}/>
        </button>
      </div>
    </div>
  )
}

const input = {
  padding: '8px 12px', borderRadius: 8, border: '1px solid #d1d5db',
  background: '#fff', fontSize: 13, outline: 'none', width: '100%', boxSizing: 'border-box',
}

function CreateForm({ onSubmit, onCancel }) {
  const [form, setForm]   = useState({ name: '', birthDate: '', diagnosis: '', notes: '' })
  const [saving, setSaving] = useState(false)
  const [error, setError]  = useState(null)

  const set = k => e => setForm(f => ({ ...f, [k]: e.target.value }))

  const handleSubmit = async () => {
    if (!form.name.trim()) return
    setSaving(true); setError(null)
    try { await onSubmit(form) }
    catch (e) { setError(e.message); setSaving(false) }
  }

  return (
    <div style={{ background: '#eef7f7', borderRadius: 12, border: '1px solid #b2dede', padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: '#073447' }}>Nuevo paciente</div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        <input
          placeholder="Nombre completo *" value={form.name} onChange={set('name')}
          style={{ ...input, gridColumn: '1 / -1' }} autoFocus
          onKeyDown={e => e.key === 'Enter' && handleSubmit()}
        />
        <input type="date" value={form.birthDate} onChange={set('birthDate')} style={input} title="Fecha de nacimiento"/>
        <input placeholder="Diagnóstico" value={form.diagnosis} onChange={set('diagnosis')} style={input}/>
      </div>
      <textarea
        placeholder="Notas iniciales (opcional)" value={form.notes} onChange={set('notes')}
        rows={2} style={{ ...input, resize: 'vertical', fontFamily: 'inherit' }}
      />
      {error && <p style={{ fontSize: 12, color: '#dc2626', margin: 0 }}>{error}</p>}
      <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
        <button onClick={onCancel}
          style={{ padding: '7px 14px', borderRadius: 8, border: '1px solid #d1d5db', background: '#fff', color: '#555', cursor: 'pointer', fontSize: 13 }}>
          Cancelar
        </button>
        <button onClick={handleSubmit} disabled={!form.name.trim() || saving}
          style={{ padding: '7px 16px', borderRadius: 8, border: 'none', background: form.name.trim() ? '#073447' : '#9ca3af', color: '#fff', cursor: form.name.trim() ? 'pointer' : 'default', fontSize: 13, fontWeight: 600 }}>
          {saving ? 'Creando…' : 'Crear y evaluar →'}
        </button>
      </div>
    </div>
  )
}

export function PatientsPage({ onSelectPatient, embedded = false }) {
  const [patients,    setPatients]    = useState([])
  const [summaries,   setSummaries]   = useState({})
  const [loading,     setLoading]     = useState(true)
  const [error,       setError]       = useState(null)
  const [search,      setSearch]      = useState('')
  const [showCreate,  setShowCreate]  = useState(false)
  const [historyFor,  setHistoryFor]  = useState(null)

  useEffect(() => { if (isConfigured) load() }, [])

  const load = async () => {
    setLoading(true); setError(null)
    try {
      const [ps, sums] = await Promise.all([getPatients(), getEvaluationSummaries()])
      setPatients(ps); setSummaries(sums)
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  const handleCreate = async (data) => {
    const p = await createPatient(data)
    onSelectPatient(p)
  }

  const handleDelete = async (p) => {
    if (!confirm(`¿Eliminar a "${p.name}" y todas sus evaluaciones?`)) return
    await deletePatient(p.id).catch(console.error)
    setPatients(ps => ps.filter(x => x.id !== p.id))
  }

  if (!isConfigured) return <NotConfiguredScreen onContinue={() => onSelectPatient(null)}/>

  const filtered = patients.filter(p =>
    p.name.toLowerCase().includes(search.toLowerCase()) ||
    (p.diagnosis || '').toLowerCase().includes(search.toLowerCase())
  )

  return (
    <>
      <div style={{ maxWidth: 680, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div style={{ display: 'flex', gap: 8 }}>
          <div style={{ flex: 1, position: 'relative' }}>
            <Search size={14} style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: '#aaa', pointerEvents: 'none' }}/>
            <input
              value={search} onChange={e => setSearch(e.target.value)}
              placeholder="Buscar paciente o diagnóstico…"
              style={{ width: '100%', padding: '10px 12px 10px 34px', borderRadius: 10, border: '1px solid #d1d5db', background: '#fff', fontSize: 14, outline: 'none', boxSizing: 'border-box' }}
            />
          </div>
          <button onClick={() => setShowCreate(true)}
            style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '9px 14px', borderRadius: 10, border: 'none', background: '#073447', color: '#fff', cursor: 'pointer', fontSize: 13, fontWeight: 600, flexShrink: 0 }}>
            <UserPlus size={14}/> Nuevo paciente
          </button>
        </div>

        {showCreate && <CreateForm onSubmit={handleCreate} onCancel={() => setShowCreate(false)}/>}

        {loading ? (
          <p style={{ textAlign: 'center', color: '#aaa', padding: 40 }}>Cargando pacientes…</p>
        ) : error ? (
          <div style={{ background: '#fef2f2', border: '1px solid #fca5a5', borderRadius: 10, padding: '12px 16px' }}>
            <p style={{ color: '#dc2626', fontSize: 13, margin: '0 0 6px' }}>Error: {error}</p>
            <button onClick={load} style={{ color: '#dc2626', fontSize: 12, textDecoration: 'underline', cursor: 'pointer', background: 'none', border: 'none', padding: 0 }}>
              Reintentar
            </button>
          </div>
        ) : filtered.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '40px 20px' }}>
            <p style={{ color: '#aaa', marginBottom: 12 }}>
              {search ? 'Sin resultados.' : 'Aún no hay pacientes.'}
            </p>
            {!search && (
              <button onClick={() => setShowCreate(true)}
                style={{ padding: '8px 18px', borderRadius: 8, border: '1.5px dashed #d1d5db', background: 'none', color: '#999', cursor: 'pointer', fontSize: 13 }}>
                + Crear primer paciente
              </button>
            )}
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {filtered.map(p => (
              <PatientCard
                key={p.id} patient={p}
                evalCount={summaries[p.id]?.count  ?? 0}
                lastEval= {summaries[p.id]?.lastDate ?? null}
                onEvaluate={() => onSelectPatient(p)}
                onHistory ={() => setHistoryFor(p)}
                onDelete  ={() => handleDelete(p)}
              />
            ))}
          </div>
        )}

        <button onClick={() => onSelectPatient(null)}
          style={{ background: 'none', border: '1.5px solid #d1d5db', borderRadius: 9, color: '#666', cursor: 'pointer', fontSize: 13, padding: '9px 20px', fontWeight: 500 }}>
          Continuar sin seleccionar paciente →
        </button>
      </div>

      {historyFor && (
        <HistoryModal
          patient={historyFor}
          onClose={() => setHistoryFor(null)}
          onEvaluate={p => { setHistoryFor(null); onSelectPatient(p) }}
        />
      )}
    </>
  )
}
