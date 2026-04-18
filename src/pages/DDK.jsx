import { useState, useEffect } from 'react'
import { useRecorder } from '../hooks/useRecorder'
import { RecordButton } from '../components/RecordButton'
import { Waveform, LiveWaveform } from '../components/Waveform'
import { Badge } from '../components/Badge'
import { decodeAudio, calcDDK, getWaveformPoints } from '../utils/audio'
import { classify } from '../utils/normative'

export function DDK({ onResult }) {
  const recorder              = useRecorder()
  const [result, setResult]   = useState(null)
  const [waveData, setWave]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [manual, setManual]   = useState(null)

  useEffect(() => {
    if (!recorder.blob) return
    let cancelled = false
    ;(async () => {
      setLoading(true)
      setManual(null)
      try {
        const { channelData, sampleRate, duration } = await decodeAudio(recorder.blob)
        if (cancelled) return
        const { nPataka, onsetTimes } = calcDDK(channelData, sampleRate)
        const points   = getWaveformPoints(channelData, sampleRate)
        const durEval  = Math.min(duration, 5)
        setResult({ nPataka, onsetTimes, duration, durEval })
        setWave({ points, duration })
        onResult('ddk', nPataka)
      } catch (e) { console.error(e) }
      if (!cancelled) setLoading(false)
    })()
    return () => { cancelled = true }
  }, [recorder.blob])

  const displayed = manual !== null ? manual : result?.nPataka
  const status    = displayed != null ? classify(displayed, 'ddk') : null

  const adjust = delta => {
    const base = manual !== null ? manual : (result?.nPataka ?? 0)
    const next = Math.max(0, Math.round(base) + delta)
    setManual(next)
    onResult('ddk', next)
  }

  return (
    <div className="space-y-6">
      <div className="bg-blue-50 border-l-4 border-blue-500 rounded-r-xl p-4">
        <p className="font-semibold text-blue-900 text-sm">Instrucciones para el paciente</p>
        <p className="text-blue-800 text-sm mt-1">
          Tome aire y repita <strong>PA · TA · KA</strong> tan rápido y claro como pueda
          durante <strong>5 segundos</strong>. Pulse stop al terminar.
        </p>
      </div>

      <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6 space-y-5">
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
          <p className="text-center text-gray-400 text-sm animate-pulse">Contando sílabas…</p>
        )}

        {result && !loading && (
          <>
            <div>
              <Waveform
                points={waveData.points}
                duration={waveData.duration}
                onsetTimes={result.onsetTimes}
              />
              <p className="text-xs text-gray-400 mt-1 text-center">
                Las líneas rojas indican las sílabas detectadas
              </p>
            </div>

            <div className="grid grid-cols-3 gap-3 text-center">
              {[
                ['Duración evaluada', `${result.durEval.toFixed(1)} s`],
                ['PATAKA detectados', (displayed ?? 0).toFixed(1)],
                ['Tasa', `${(displayed / result.durEval).toFixed(2)} /s`],
              ].map(([lbl, val]) => (
                <div key={lbl} className="bg-gray-50 rounded-xl p-3">
                  <div className="text-xl font-bold text-gray-800">{val}</div>
                  <div className="text-xs text-gray-400 mt-0.5">{lbl}</div>
                </div>
              ))}
            </div>

            {status && (
              <Badge
                label="Diadococinesias"
                value={`${(displayed ?? 0).toFixed(1)} rep / 5 s`}
                status={status}
                refText="≥ 5 rep / 5 s"
              />
            )}

            {/* Manual correction */}
            <div className="pt-3 border-t border-gray-100">
              <p className="text-xs text-gray-400 mb-3 text-center">
                ¿El conteo automático no es preciso? Corrígelo:
              </p>
              <div className="flex items-center justify-center gap-4">
                <button
                  onClick={() => adjust(-1)}
                  className="w-10 h-10 rounded-full bg-gray-100 hover:bg-gray-200
                             text-xl font-bold text-gray-600 transition"
                >−</button>
                <span className="text-2xl font-bold tabular-nums w-10 text-center">
                  {(manual !== null ? manual : Math.round(result.nPataka))}
                </span>
                <button
                  onClick={() => adjust(+1)}
                  className="w-10 h-10 rounded-full bg-gray-100 hover:bg-gray-200
                             text-xl font-bold text-gray-600 transition"
                >+</button>
                <span className="text-sm text-gray-400">repeticiones</span>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
