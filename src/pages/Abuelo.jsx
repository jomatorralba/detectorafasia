import { useState, useEffect } from 'react'
import { useRecorder } from '../hooks/useRecorder'
import { RecordButton } from '../components/RecordButton'
import { Waveform, LiveWaveform } from '../components/Waveform'
import { Badge } from '../components/Badge'
import { decodeAudio, getWaveformPoints } from '../utils/audio'
import { ABUELO_TEXT, ABUELO_WORDS, classify } from '../utils/normative'

export function Abuelo({ onResult }) {
  const recorder              = useRecorder()
  const [result, setResult]   = useState(null)
  const [waveData, setWave]   = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!recorder.blob) return
    let cancelled = false
    ;(async () => {
      setLoading(true)
      try {
        const { channelData, sampleRate, duration } = await decodeAudio(recorder.blob)
        if (cancelled) return
        const wpm    = duration > 0 ? (ABUELO_WORDS / duration) * 60 : 0
        const points = getWaveformPoints(channelData, sampleRate)
        setResult({ wpm, duration })
        setWave({ points, duration })
        onResult('wpm', wpm)
      } catch (e) { console.error(e) }
      if (!cancelled) setLoading(false)
    })()
    return () => { cancelled = true }
  }, [recorder.blob])

  const status = result ? classify(result.wpm, 'wpm') : null

  return (
    <div className="space-y-6">
      <div className="bg-blue-50 border-l-4 border-blue-500 rounded-r-xl p-4">
        <p className="font-semibold text-blue-900 text-sm">Instrucciones para el paciente</p>
        <p className="text-blue-800 text-sm mt-1">
          Lea el texto en voz alta a su ritmo habitual, de forma{' '}
          <strong>fluida y natural</strong>. Pulse stop cuando haya terminado.
        </p>
      </div>

      {/* Text */}
      <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
        <p className="font-serif text-[1.15rem] leading-[2.1] text-gray-800 select-none">
          {ABUELO_TEXT}
        </p>
        <p className="text-xs text-gray-400 mt-3">{ABUELO_WORDS} palabras</p>
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
          <p className="text-center text-gray-400 text-sm animate-pulse">Calculando velocidad…</p>
        )}

        {result && !loading && (
          <>
            <Waveform
              points={waveData.points} duration={waveData.duration}
              color="#15803d"
            />

            <div className="grid grid-cols-3 gap-3 text-center">
              {[
                ['Duración total', `${result.duration.toFixed(1)} s`],
                ['Palabras / min', `${Math.round(result.wpm)}`],
                ['Mínimo normal', '≥ 100 ppm'],
              ].map(([lbl, val]) => (
                <div key={lbl} className="bg-gray-50 rounded-xl p-3">
                  <div className="text-xl font-bold text-gray-800">{val}</div>
                  <div className="text-xs text-gray-400 mt-0.5">{lbl}</div>
                </div>
              ))}
            </div>

            <Badge
              label="Velocidad lectora"
              value={`${Math.round(result.wpm)} ppm`}
              status={status}
              refText="≥ 100 palabras/min"
            />

            <p className={`text-sm rounded-lg px-4 py-2 ${
              result.wpm < 70  ? 'bg-red-50 text-red-800' :
              result.wpm < 100 ? 'bg-yellow-50 text-yellow-800' :
              result.wpm < 130 ? 'bg-yellow-50 text-yellow-800' :
              'bg-green-50 text-green-800'
            }`}>
              {result.wpm < 70  ? '🔴 Velocidad muy reducida · posible disartria moderada-severa.' :
               result.wpm < 100 ? '🟡 Velocidad por debajo del rango normal.' :
               result.wpm < 130 ? '🟡 Velocidad ligeramente reducida o en límite inferior.' :
               '🟢 Velocidad dentro del rango normal.'}
            </p>
          </>
        )}
      </div>
    </div>
  )
}
