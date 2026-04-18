import { useState, useEffect } from 'react'
import { useRecorder } from '../hooks/useRecorder'
import { RecordButton } from '../components/RecordButton'
import { LiveWaveform } from '../components/Waveform'
import { ComparisonCanvas } from '../components/ComparisonCanvas'
import { decodeAudio, getEnvelope, gaussianRef } from '../utils/audio'
import { PALABRAS } from '../utils/normative'

const WORD_LIST = Object.keys(PALABRAS)

export function Palabras({ onResult }) {
  const [wordIdx, setWordIdx]   = useState(0)
  const [recorded, setRecorded] = useState({}) // word → duration

  const word = WORD_LIST[wordIdx]

  return (
    <div className="space-y-6">
      <div className="bg-blue-50 border-l-4 border-blue-500 rounded-r-xl p-4">
        <p className="font-semibold text-blue-900 text-sm">Instrucciones para el paciente</p>
        <p className="text-blue-800 text-sm mt-1">
          Lea la palabra que aparece en pantalla en voz alta, de forma <strong>clara y natural</strong>.
          Grabamos una palabra a la vez. Pulse stop al terminar.
        </p>
      </div>

      {/* Word selector */}
      <div className="flex gap-2 flex-wrap">
        {WORD_LIST.map((w, i) => (
          <button
            key={w}
            onClick={() => setWordIdx(i)}
            className={`px-4 py-1.5 rounded-full text-sm font-semibold border-2 transition
              ${wordIdx === i
                ? 'bg-blue-600 border-blue-600 text-white'
                : recorded[w]
                ? 'bg-green-50 border-green-400 text-green-700'
                : 'bg-white border-gray-200 text-gray-600 hover:border-blue-300'}`}
          >
            {recorded[w] ? '✓ ' : ''}{w}
          </button>
        ))}
      </div>

      <WordCard
        key={word}
        word={word}
        onRecorded={(dur) => {
          setRecorded(r => ({ ...r, [word]: dur }))
          onResult('palabra_' + word, dur)
        }}
      />

      {/* Progress & table */}
      <div className="space-y-2">
        <div className="flex items-center gap-3">
          <div className="flex-1 bg-gray-100 rounded-full h-2">
            <div
              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${(Object.keys(recorded).length / WORD_LIST.length) * 100}%` }}
            />
          </div>
          <span className="text-sm text-gray-500 whitespace-nowrap">
            {Object.keys(recorded).length} / {WORD_LIST.length}
          </span>
        </div>

        {Object.keys(recorded).length > 0 && (
          <div className="bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-xs text-gray-500 uppercase">
                <tr>
                  {['Palabra', 'Duración', 'Referencia', 'Ratio', 'Estado'].map(h => (
                    <th key={h} className="px-4 py-2 text-left font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(recorded).map(([w, d]) => {
                  const ratio = d / PALABRAS[w].ref
                  return (
                    <tr key={w} className="border-t border-gray-50">
                      <td className="px-4 py-2 font-semibold">{w}</td>
                      <td className="px-4 py-2">{d.toFixed(2)} s</td>
                      <td className="px-4 py-2">{PALABRAS[w].ref} s</td>
                      <td className="px-4 py-2">{ratio.toFixed(2)}×</td>
                      <td className="px-4 py-2">
                        {ratio <= 1.5 ? '✅ Normal' : '⚠️ Lento'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

function WordCard({ word, onRecorded }) {
  const recorder              = useRecorder()
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)

  const ref = PALABRAS[word]

  useEffect(() => {
    if (!recorder.blob) return
    let cancelled = false
    ;(async () => {
      setLoading(true)
      try {
        const { channelData, sampleRate, duration } = await decodeAudio(recorder.blob)
        if (cancelled) return
        const envelope = getEnvelope(channelData)
        const refData  = gaussianRef(ref.ref, duration)
        setResult({ envelope, duration, refData })
        onRecorded(duration)
      } catch (e) { console.error(e) }
      if (!cancelled) setLoading(false)
    })()
    return () => { cancelled = true }
  }, [recorder.blob])

  const ratio = result ? result.duration / ref.ref : null

  return (
    <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6 space-y-5">
      {/* Big word display */}
      <div className="rounded-2xl text-center py-8 px-4"
        style={{ background: 'linear-gradient(135deg,#1a237e,#3949ab)' }}>
        <div className="text-4xl font-black text-white tracking-[6px]">{word}</div>
        <div className="text-blue-200 text-sm mt-2">
          {ref.syl} sílabas · duración esperada ≈ {ref.ref} s
        </div>
      </div>

      <div className="flex justify-center py-2">
        <RecordButton
          isRecording={recorder.isRecording}
          elapsed={recorder.elapsed}
          onStart={recorder.start}
          onStop={recorder.stop}
        />
      </div>

      {recorder.isRecording && <LiveWaveform analyserRef={recorder.analyserRef} />}

      {loading && (
        <p className="text-center text-gray-400 text-sm animate-pulse">Analizando…</p>
      )}

      {result && !loading && (
        <>
          <div>
            <ComparisonCanvas
              envelope={result.envelope}
              duration={result.duration}
              ref={result.refData}
              height={150}
            />
            <div className="flex gap-4 mt-2 text-xs text-gray-400 justify-center">
              <span className="flex items-center gap-1">
                <span className="inline-block w-6 h-0.5 bg-blue-700 rounded" /> Grabación
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block w-6 h-0.5 border-t-2 border-dashed border-green-500" /> Referencia normal
              </span>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3 text-center">
            <div className="bg-gray-50 rounded-xl p-3">
              <div className="text-xl font-bold text-gray-800">{result.duration.toFixed(2)} s</div>
              <div className="text-xs text-gray-400 mt-0.5">Duración grabada</div>
            </div>
            <div className="bg-gray-50 rounded-xl p-3">
              <div className="text-xl font-bold text-gray-800">{ref.ref} s</div>
              <div className="text-xs text-gray-400 mt-0.5">Referencia normal</div>
            </div>
            <div className={`rounded-xl p-3 ${ratio <= 1.5
              ? 'bg-green-50 text-green-800' : 'bg-yellow-50 text-yellow-800'}`}>
              <div className="text-xl font-bold">{ratio.toFixed(2)}×</div>
              <div className="text-xs mt-0.5">{ratio <= 1.5 ? '✅ Normal' : '⚠️ Lento'}</div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
