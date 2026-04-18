import { useState, useEffect } from 'react'
import { useRecorder } from '../hooks/useRecorder'
import { RecordButton } from '../components/RecordButton'
import { LiveWaveform } from '../components/Waveform'
import { SpectrogramCanvas } from '../components/SpectrogramCanvas'
import { decodeAudio } from '../utils/audio'
import {
  loadWords, addWord, deleteWord,
  loadReference, saveReference, deleteReference,
  blobToBase64, base64ToBlob,
} from '../utils/storage'

export function Palabras({ onResult }) {
  const [words, setWords]           = useState([])
  const [selected, setSelected]     = useState(null)
  const [patientData, setPatient]   = useState(null)
  const [refData, setRefData]       = useState(null)
  const [refMode, setRefMode]       = useState(false) // recording reference vs patient
  const [addMode, setAddMode]       = useState(false)
  const [newWord, setNewWord]       = useState('')
  const [recorded, setRecorded]     = useState({}) // word → duration
  const [loadingRef, setLoadingRef] = useState(false)

  const recorder = useRecorder()

  // Load words on mount
  useEffect(() => {
    const ws = loadWords()
    setWords(ws)
    if (ws.length > 0) setSelected(ws[0])
  }, [])

  // Load reference when word changes
  useEffect(() => {
    if (!selected) return
    setPatient(null)
    setRefData(null)
    setRefMode(false)

    const stored = loadReference(selected)
    if (!stored) return

    setLoadingRef(true)
    decodeAudio(base64ToBlob(stored.audio, stored.mimeType))
      .then(data => setRefData(data))
      .catch(console.error)
      .finally(() => setLoadingRef(false))
  }, [selected])

  // Handle recording blob
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
          setRefMode(false)
        } else {
          setPatient(data)
          setRecorded(r => ({ ...r, [selected]: data.duration }))
          onResult('palabra_' + selected, data.duration)
        }
      } catch (e) { console.error(e) }
    })()
    return () => { cancelled = true }
  }, [recorder.blob])

  const selectWord = w => {
    setSelected(w); setPatient(null); setRefMode(false)
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
  }

  const handleDeleteRef = () => {
    deleteReference(selected)
    setRefData(null)
  }

  return (
    <div className="space-y-5">
      <div className="bg-blue-50 border-l-4 border-blue-500 rounded-r-xl p-4">
        <p className="font-semibold text-blue-900 text-sm">Instrucciones para el paciente</p>
        <p className="text-blue-800 text-sm mt-1">
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
            {recorded[w] && '✓ '}{w}
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
              <p className="text-sm font-semibold text-gray-800">
                {loadingRef ? 'Cargando referencia…' : refData ? '✅ Referencia grabada' : '⚠️ Sin referencia — graba una voz normal'}
              </p>
              <p className="text-xs text-gray-500 mt-0.5">
                {refData
                  ? 'El espectrograma se superpondrá sobre la referencia (azul = paciente, verde = normal)'
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
                onClick={() => { setRefMode(true); setPatient(null) }}
                className="px-2.5 py-1.5 text-xs font-medium rounded-lg border bg-white hover:bg-gray-50 transition"
              >
                {refData ? '↺ Re-grabar' : 'Grabar referencia'}
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

          {/* Spectrogram */}
          {patientData && (
            <div className="space-y-3">
              <h3 className="text-sm font-semibold text-gray-800">Comparativa espectral (alineada por onset)</h3>
              <SpectrogramCanvas
                patientData={patientData}
                referenceData={refData}
                height={200}
              />
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-gray-50 rounded-xl p-3 text-center">
                  <div className="text-xl font-bold font-mono text-gray-900">
                    {patientData.duration.toFixed(2)} s
                  </div>
                  <div className="text-xs text-gray-500 mt-0.5">Duración paciente</div>
                </div>
                {refData && (
                  <div className="bg-gray-50 rounded-xl p-3 text-center">
                    <div className="text-xl font-bold font-mono text-gray-900">
                      {refData.duration.toFixed(2)} s
                    </div>
                    <div className="text-xs text-gray-500 mt-0.5">Duración referencia</div>
                  </div>
                )}
              </div>
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
                    <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full font-medium">
                      ✓ Grabada
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
