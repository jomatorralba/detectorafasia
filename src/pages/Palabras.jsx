import { useState, useEffect } from 'react'
import { useRecorder } from '../hooks/useRecorder'
import { RecordButton } from '../components/RecordButton'
import { LiveWaveform } from '../components/Waveform'
import { SpectrogramCanvas } from '../components/SpectrogramCanvas'
import { OscillogramCanvas } from '../components/OscillogramCanvas'
import { decodeAudio } from '../utils/audio'
import { analyzeAudio, acousticDistance } from '../utils/acoustics'
import {
  loadWords, addWord, deleteWord,
  loadReference, saveReference, deleteReference,
  blobToBase64, base64ToBlob,
} from '../utils/storage'

function MetricCard({ label, value, refValue, prevValues = [], accent }) {
  return (
    <div style={{
      background: '#f9fafb', borderRadius: 10, padding: '10px 14px',
      borderLeft: accent ? `3px solid ${accent}` : '3px solid #e5e7eb',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 6 }}>
        {/* Left: current value + label */}
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 18, fontWeight: 700, fontFamily: 'monospace', color: '#111', lineHeight: 1.2 }}>
            {value ?? <span style={{ color: '#ccc', fontSize: 14 }}>—</span>}
          </div>
          <div style={{ fontSize: 11, color: '#888', marginTop: 2 }}>{label}</div>
        </div>
        {/* Right: ref in red + history in gray */}
        {(refValue || prevValues.length > 0) && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 3, flexShrink: 0, maxWidth: 96 }}>
            {refValue && (
              <span style={{ fontSize: 10, color: '#ef4444', fontFamily: 'monospace', fontWeight: 600, lineHeight: 1 }}>
                {refValue}
              </span>
            )}
            {prevValues.length > 0 && (
              <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                {prevValues.map((v, i) => (
                  <span key={i} style={{
                    fontSize: 9, fontFamily: 'monospace', lineHeight: 1.4,
                    color: `rgba(40,40,40,${0.3 + (i / prevValues.length) * 0.55})`,
                  }}>{v}</span>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function DistanceBadge({ dist }) {
  if (dist == null) return null
  const pct   = Math.round(dist * 100)
  const color = pct < 20 ? '#22c55e' : pct < 40 ? '#f59e0b' : '#ef4444'
  const label = pct < 20 ? 'Similitud alta' : pct < 40 ? 'Similitud moderada' : 'Diferencia significativa'
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 10,
      background: '#f9fafb', borderRadius: 10, padding: '10px 14px',
      borderLeft: `3px solid ${color}`,
    }}>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 18, fontWeight: 700, fontFamily: 'monospace', color: '#111' }}>
          {pct}%
        </div>
        <div style={{ fontSize: 11, color: '#888', marginTop: 2 }}>Distancia espectral</div>
      </div>
      <span style={{
        fontSize: 10, fontWeight: 600, color, background: color + '18',
        padding: '2px 7px', borderRadius: 4, border: `1px solid ${color}40`,
      }}>
        {label}
      </span>
    </div>
  )
}

export function Palabras({ onResult }) {
  const [words, setWords]               = useState([])
  const [selected, setSelected]         = useState(null)
  const [patientData, setPatient]       = useState(null)
  const [refData, setRefData]           = useState(null)
  const [patAnalysis, setPatAnalysis]   = useState(null)
  const [refAnalysis, setRefAnalysis]   = useState(null)
  const [refMode, setRefMode]           = useState(false)
  const [addMode, setAddMode]           = useState(false)
  const [newWord, setNewWord]           = useState('')
  const [recorded, setRecorded]         = useState({})
  const [history, setHistory]           = useState({})
  const [loadingRef, setLoadingRef]     = useState(false)

  const recorder = useRecorder()

  useEffect(() => {
    const ws = loadWords()
    setWords(ws)
    if (ws.length > 0) setSelected(ws[0])
  }, [])

  useEffect(() => {
    if (!selected) return
    setPatient(null)
    setRefData(null)
    setPatAnalysis(null)
    setRefAnalysis(null)
    setRefMode(false)

    const stored = loadReference(selected)
    if (!stored) return

    setLoadingRef(true)
    decodeAudio(base64ToBlob(stored.audio, stored.mimeType))
      .then(data => {
        setRefData(data)
        setRefAnalysis(analyzeAudio(data))
      })
      .catch(console.error)
      .finally(() => setLoadingRef(false))
  }, [selected])

  useEffect(() => {
    if (!recorder.blob || !selected) return
    let cancelled = false
    ;(async () => {
      try {
        const data = await decodeAudio(recorder.blob)
        if (cancelled) return

        if (refMode) {
          const b64 = await blobToBase64(recorder.blob)
          saveReference(selected, b64, recorder.blob.type)
          setRefData(data)
          setRefAnalysis(analyzeAudio(data))
          setRefMode(false)
        } else {
          setPatient(data)
          const analysis = analyzeAudio(data)
          setPatAnalysis(analysis)
          setRecorded(r => ({ ...r, [selected]: data.duration }))
          onResult('palabra_' + selected, data.duration)
          setHistory(h => {
            const prev = h[selected] || []
            const entry = {
              duration:    data.duration,
              onsetSec:    analysis.onsetSec,
              rmsDb:       analysis.rmsDb,
              peakDb:      analysis.peakDb,
              cvIntensity: analysis.cvIntensity,
              f0:          analysis.f0,
              centroid:    analysis.centroid,
            }
            return { ...h, [selected]: [...prev, entry].slice(-8) }
          })
        }
      } catch (e) { console.error(e) }
    })()
    return () => { cancelled = true }
  }, [recorder.blob])

  const dist = patAnalysis && refAnalysis
    ? acousticDistance(patAnalysis.spec, patAnalysis.onsetFrame, refAnalysis.spec, refAnalysis.onsetFrame)
    : null

  const selectWord = w => {
    setSelected(w); setPatient(null); setPatAnalysis(null); setRefMode(false)
  }

  const handleAddWord = () => {
    const w = newWord.trim().toUpperCase()
    if (!w) return
    addWord(w)
    setWords(loadWords())
    setSelected(w)
    setAddMode(false)
    setNewWord('')
  }

  const handleDeleteWord = w => {
    if (!confirm(`¿Eliminar la palabra "${w}" y su referencia?`)) return
    deleteWord(w)
    const ws = loadWords()
    setWords(ws)
    setSelected(ws[0] || null)
    setRecorded(r => { const n = { ...r }; delete n[w]; return n })
    setHistory(h => { const n = { ...h }; delete n[w]; return n })
  }

  const handleDeleteRef = () => {
    deleteReference(selected)
    setRefData(null)
    setRefAnalysis(null)
  }

  return (
    <div className="space-y-5">
      <div style={{ background: "#e8f4f4", borderLeft: "4px solid #116b70", borderRadius: "0 12px 12px 0", padding: 16 }}>
        <p className="font-semibold text-sm" style={{ color: "#073447" }}>Instrucciones para el paciente</p>
        <p className="text-sm mt-1" style={{ color: "#0e5a5e" }}>
          Lea la palabra que aparece en pantalla de forma <strong>clara y natural</strong>.
          Grabamos una palabra a la vez.
        </p>
      </div>

      {/* Word selector */}
      <div className="flex items-center gap-2 flex-wrap">
        {words.map(w => (
          <button
            key={w}
            onClick={() => selectWord(w)}
            className={`group flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm
              font-medium border transition
              ${selected === w
                ? 'bg-gray-900 border-gray-900 text-white'
                : recorded[w]
                ? 'bg-green-50 border-green-300 text-green-800'
                : 'bg-white border-gray-200 text-gray-700 hover:border-gray-400'}`}
          >
            {recorded[w] && (
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#22c55e', display: 'inline-block', flexShrink: 0 }} />
            )}
            {w}
            {words.length > 1 && selected === w && (
              <span
                onClick={e => { e.stopPropagation(); handleDeleteWord(w) }}
                className="ml-0.5 opacity-40 hover:opacity-100 text-xs"
                title="Eliminar palabra"
              >×</span>
            )}
          </button>
        ))}
        <button
          onClick={() => setAddMode(a => !a)}
          className="px-3 py-1.5 rounded-lg text-sm font-medium border border-dashed
            border-gray-300 text-gray-400 hover:border-gray-500 hover:text-gray-600 transition"
        >
          + Añadir
        </button>
      </div>

      {/* Add word form */}
      {addMode && (
        <div className="flex gap-2 p-4 bg-gray-50 rounded-xl border border-gray-200">
          <input
            value={newWord}
            onChange={e => setNewWord(e.target.value.toUpperCase())}
            onKeyDown={e => e.key === 'Enter' && handleAddWord()}
            placeholder="NUEVA PALABRA"
            className="flex-1 px-3 py-2 rounded-lg border border-gray-200 text-sm
              font-medium bg-white focus:outline-none focus:ring-2 focus:ring-gray-900"
            autoFocus
          />
          <button onClick={handleAddWord}
            className="px-4 py-2 bg-gray-900 text-white rounded-lg text-sm font-medium hover:bg-gray-700 transition">
            Añadir
          </button>
          <button onClick={() => setAddMode(false)}
            className="px-3 py-2 border border-gray-200 rounded-lg text-sm text-gray-500 hover:bg-gray-100 transition">
            ✕
          </button>
        </div>
      )}

      {selected && (
        <>
          {/* Big word card */}
          <div className="rounded-2xl py-10 text-center"
            style={{ background: 'linear-gradient(135deg,#0d1117 0%,#1c2333 100%)' }}>
            <div className="text-5xl font-black text-white tracking-[8px]">{selected}</div>
          </div>

          {/* Reference banner */}
          <div className={`rounded-xl p-4 border flex items-start justify-between gap-3
            ${refData ? 'bg-emerald-50 border-emerald-200' : 'bg-amber-50 border-amber-200'}`}>
            <div>
              <p className="text-sm font-semibold text-gray-800 flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full shrink-0 ${refData ? 'bg-emerald-400' : 'bg-amber-400'}`} />
                {loadingRef ? 'Cargando referencia…' : refData ? 'Referencia grabada' : 'Sin referencia — graba una voz normal'}
              </p>
              <p className="text-xs text-gray-500 mt-0.5 ml-4">
                {refData
                  ? 'Espectrogramas en paralelo · oscilograma superpuesto alineado por onset'
                  : 'Sin referencia solo se muestra el espectrograma del paciente'}
              </p>
            </div>
            <div className="flex gap-2 shrink-0">
              {refData && (
                <button onClick={handleDeleteRef}
                  className="px-2.5 py-1.5 text-xs font-medium rounded-lg border border-red-200
                    bg-white text-red-600 hover:bg-red-50 transition">
                  Borrar
                </button>
              )}
              <button
                onClick={() => { setRefMode(true); setPatient(null); setPatAnalysis(null) }}
                className="px-2.5 py-1.5 text-xs font-medium rounded-lg border bg-white hover:bg-gray-50 transition"
              >
                {refData ? 'Re-grabar' : 'Grabar referencia'}
              </button>
            </div>
          </div>

          {/* Recording section */}
          {refMode ? (
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-5 space-y-4">
              <p className="text-sm font-semibold text-amber-900 text-center">
                Grabando referencia normal de <strong>{selected}</strong>
              </p>
              <p className="text-xs text-center text-amber-700">
                Pronuncia la palabra con voz natural y clara. Esta grabación servirá de referencia.
              </p>
              <div className="flex justify-center">
                <RecordButton
                  isRecording={recorder.isRecording}
                  elapsed={recorder.elapsed}
                  onStart={recorder.start}
                  onStop={recorder.stop}
                />
              </div>
              {recorder.isRecording && <LiveWaveform analyserRef={recorder.analyserRef} />}
              <button onClick={() => setRefMode(false)}
                className="w-full text-xs text-amber-600 hover:text-amber-900 transition">
                Cancelar
              </button>
            </div>
          ) : (
            <div className="bg-white border border-gray-100 rounded-xl p-5 space-y-4 shadow-sm">
              <p className="text-sm text-gray-500 text-center font-medium">
                Grabar al paciente pronunciando <strong>{selected}</strong>
              </p>
              <div className="flex justify-center">
                <RecordButton
                  isRecording={recorder.isRecording}
                  elapsed={recorder.elapsed}
                  onStart={recorder.start}
                  onStop={recorder.stop}
                />
              </div>
              {recorder.isRecording && <LiveWaveform analyserRef={recorder.analyserRef} />}
            </div>
          )}

          {/* Analysis panels */}
          {patientData && (
            <div className="space-y-3">
              {/* Section label */}
              <div className="flex items-center gap-3">
                <span className="text-sm font-semibold text-gray-700">Análisis acústico</span>
                <div style={{ flex: 1, height: 1, background: '#f0f0f0' }} />
              </div>

              {/* Oscillogram — overlaid, onset-aligned */}
              <div>
                <p style={{ fontSize: 11, color: '#aaa', marginBottom: 4, fontFamily: 'monospace', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Oscilograma{refData ? ' (onset-aligned)' : ''}
                </p>
                <OscillogramCanvas
                  patientData={patientData}
                  referenceData={refData}
                  height={90}
                />
              </div>

              {/* Spectrograms — side by side */}
              <div>
                <p style={{ fontSize: 11, color: '#aaa', marginBottom: 4, fontFamily: 'monospace', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Espectrograma{refData ? ' (referencia | paciente)' : ''}
                </p>
                <SpectrogramCanvas
                  patientData={patientData}
                  referenceData={refData}
                  height={180}
                />
              </div>

              {/* Acoustic metrics grid */}
              {patAnalysis && (() => {
                const wh  = history[selected] || []
                const prev = wh.slice(0, -1)  // all entries before current
                const p = k => prev.map(e => e[k] != null ? String(e[k]) : '—')
                const pFmt = (k, fn) => prev.map(e => e[k] != null ? fn(e[k]) : '—')
                return (
                <div>
                  <p style={{ fontSize: 11, color: '#aaa', marginBottom: 6, fontFamily: 'monospace', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Métricas
                  </p>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
                    <MetricCard
                      label="Duración"
                      value={`${patientData.duration.toFixed(2)} s`}
                      refValue={refData ? `${refData.duration.toFixed(2)}s` : null}
                      prevValues={pFmt('duration', v => v.toFixed(2))}
                      accent="#6366f1"
                    />
                    <MetricCard
                      label="Latencia de inicio"
                      value={`${Math.round(patAnalysis.onsetSec * 1000)} ms`}
                      refValue={refAnalysis ? `${Math.round(refAnalysis.onsetSec * 1000)}ms` : null}
                      prevValues={pFmt('onsetSec', v => Math.round(v * 1000))}
                      accent="#6366f1"
                    />
                    <MetricCard
                      label="Intensidad media"
                      value={`${patAnalysis.rmsDb.toFixed(1)} dB`}
                      refValue={refAnalysis ? `${refAnalysis.rmsDb.toFixed(1)}dB` : null}
                      prevValues={pFmt('rmsDb', v => v.toFixed(1))}
                      accent="#0ea5e9"
                    />
                    <MetricCard
                      label="Intensidad pico"
                      value={`${patAnalysis.peakDb.toFixed(1)} dB`}
                      refValue={refAnalysis ? `${refAnalysis.peakDb.toFixed(1)}dB` : null}
                      prevValues={pFmt('peakDb', v => v.toFixed(1))}
                      accent="#0ea5e9"
                    />
                    <MetricCard
                      label="Variabilidad intens."
                      value={`${Math.round(patAnalysis.cvIntensity * 100)} %`}
                      refValue={refAnalysis ? `${Math.round(refAnalysis.cvIntensity * 100)}%` : null}
                      prevValues={pFmt('cvIntensity', v => Math.round(v * 100))}
                      accent="#f59e0b"
                    />
                    <MetricCard
                      label="F0 fundamental"
                      value={patAnalysis.f0 ? `${patAnalysis.f0} Hz` : null}
                      refValue={refAnalysis?.f0 ? `${refAnalysis.f0}Hz` : null}
                      prevValues={p('f0')}
                      accent="#ec4899"
                    />
                    <MetricCard
                      label="Centroide espectral"
                      value={`${patAnalysis.centroid} Hz`}
                      refValue={refAnalysis ? `${refAnalysis.centroid}Hz` : null}
                      prevValues={p('centroid')}
                      accent="#8b5cf6"
                    />
                    {dist != null && (
                      <div style={{ gridColumn: 'span 2' }}>
                        <DistanceBadge dist={dist} />
                      </div>
                    )}
                  </div>
                </div>
                )
              })()}
            </div>
          )}
        </>
      )}

      {/* Table of recorded words */}
      {Object.keys(recorded).length > 0 && (
        <div className="border border-gray-100 rounded-xl overflow-hidden mt-2">
          <div className="px-4 py-2.5 bg-gray-50 border-b border-gray-100">
            <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Palabras grabadas
            </span>
          </div>
          <table className="w-full text-sm">
            <tbody>
              {Object.entries(recorded).map(([w, d]) => (
                <tr key={w} className="border-t border-gray-50 hover:bg-gray-50">
                  <td className="px-4 py-2.5 font-semibold text-gray-800">{w}</td>
                  <td className="px-4 py-2.5 font-mono text-gray-600">{d.toFixed(2)} s</td>
                  <td className="px-4 py-2.5">
                    <span className="inline-flex items-center gap-1.5 text-xs text-green-700 font-medium">
                      <span className="w-1.5 h-1.5 rounded-full bg-green-400 shrink-0" />
                      Grabada
                    </span>
                  </td>
                  <td className="px-4 py-2.5 text-right">
                    <button onClick={() => selectWord(w)}
                      className="text-xs text-blue-600 hover:underline">
                      Ver
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
