import { useState, useEffect } from 'react'
import { useRecorder } from '../hooks/useRecorder'
import { RecordButton } from '../components/RecordButton'
import { Waveform, LiveWaveform } from '../components/Waveform'
import { Badge } from '../components/Badge'
import { Gauge } from '../components/Gauge'
import { decodeAudio, calcTMF, getWaveformPoints } from '../utils/audio'
import { classify } from '../utils/normative'

function RecordBlock({ label, caption, color, normKey, normMin, onResult }) {
  const recorder = useRecorder()
  const [result, setResult]     = useState(null)
  const [waveData, setWaveData] = useState(null)
  const [loading, setLoading]   = useState(false)

  useEffect(() => {
    if (!recorder.blob) return
    let cancelled = false
    ;(async () => {
      setLoading(true)
      try {
        const { channelData, sampleRate, duration } = await decodeAudio(recorder.blob)
        if (cancelled) return
        const tmf    = calcTMF(channelData, sampleRate)
        const points = getWaveformPoints(channelData, sampleRate)
        setResult({ tmf, duration })
        setWaveData({ points, duration })
        onResult(tmf)
      } catch (e) { console.error(e) }
      if (!cancelled) setLoading(false)
    })()
    return () => { cancelled = true }
  }, [recorder.blob])

  const status = result ? classify(result.tmf, normKey) : null

  return (
    <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6 flex flex-col gap-4">
      <div>
        <h3 className="text-lg font-bold text-gray-800">{label}</h3>
        <p className="text-sm text-gray-400 mt-0.5">{caption}</p>
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
        <p className="text-center text-gray-400 text-sm animate-pulse">Procesando…</p>
      )}

      {result && !loading && (
        <>
          <Waveform points={waveData.points} duration={waveData.duration} color={color} />
          <div className="grid grid-cols-2 gap-3 items-center">
            <Gauge
              value={result.tmf} min={normMin} max={35}
              label={`TMF ${label}`} unit="s"
            />
            <Badge
              label={label} value={`${result.tmf.toFixed(1)} s`}
              status={status} refText={`≥ ${normMin} s`}
            />
          </div>
        </>
      )}
    </div>
  )
}

export function TMF({ onResult }) {
  const [tmfA, setTmfA] = useState(null)
  const [tmfS, setTmfS] = useState(null)

  const handleA = v => { setTmfA(v); onResult('tmf_a', v) }
  const handleS = v => { setTmfS(v); onResult('tmf_s', v) }

  const ratio = tmfA && tmfS && tmfA > 0 ? tmfS / tmfA : null

  return (
    <div className="space-y-6">
      <div className="bg-blue-50 border-l-4 border-blue-500 rounded-r-xl p-4">
        <p className="font-semibold text-blue-900 text-sm">Instrucciones para el paciente</p>
        <p className="text-blue-800 text-sm mt-1">
          Tome aire profundamente. Emita el sonido indicado de forma <strong>sostenida y uniforme</strong>{' '}
          lo más tiempo posible. Pare cuando se le acabe el aire.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-5">
        <RecordBlock
          label="Vocal /A/" caption="Emitir A sostenida lo máximo posible"
          color="#1565C0" normKey="tmf_a" normMin={10} onResult={handleA}
        />
        <RecordBlock
          label="Fricativa /S/" caption="Emitir S sostenida lo máximo posible"
          color="#6a1b9a" normKey="tmf_s" normMin={8} onResult={handleS}
        />
      </div>

      {ratio !== null && (
        <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
          <h3 className="text-lg font-bold text-gray-800 mb-4">Cociente S/A</h3>
          <div className="grid grid-cols-3 gap-4 items-center">
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-700">{tmfA.toFixed(1)}<span className="text-base font-normal ml-1">s</span></div>
              <div className="text-xs text-gray-400 mt-1">TMF /A/</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-700">{tmfS.toFixed(1)}<span className="text-base font-normal ml-1">s</span></div>
              <div className="text-xs text-gray-400 mt-1">TMF /S/</div>
            </div>
            <Badge
              label="Cociente S/A" value={ratio.toFixed(2)}
              status={classify(ratio, 'ratio', 'range')} refText="0.7 – 1.4"
            />
          </div>
          {ratio > 1.4 && (
            <p className="mt-4 text-sm bg-blue-50 text-blue-800 rounded-lg px-4 py-2">
Cociente &gt; 1.4 → posible patología glótica.
            </p>
          )}
          {ratio < 0.7 && (
            <p className="mt-4 text-sm bg-blue-50 text-blue-800 rounded-lg px-4 py-2">
Cociente &lt; 0.7 → posible patología respiratoria o insuficiencia velofaríngea.
            </p>
          )}
        </div>
      )}
    </div>
  )
}
