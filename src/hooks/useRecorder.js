import { useState, useRef, useCallback, useEffect } from 'react'

export function useRecorder() {
  const [isRecording, setIsRecording] = useState(false)
  const [elapsed, setElapsed]         = useState(0)
  const [blob, setBlob]               = useState(null)

  const mrRef       = useRef(null)
  const chunksRef   = useRef([])
  const streamRef   = useRef(null)
  const acRef       = useRef(null)
  const timerRef    = useRef(null)
  const analyserRef = useRef(null) // live waveform — not state, no re-renders

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      const ac = new AudioContext()
      acRef.current = ac
      const src     = ac.createMediaStreamSource(stream)
      const analyser = ac.createAnalyser()
      analyser.fftSize = 1024
      src.connect(analyser)
      analyserRef.current = analyser

      const mimeType =
        MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' :
        MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')  ? 'audio/ogg;codecs=opus'  :
        ''

      const mr = new MediaRecorder(stream, mimeType ? { mimeType } : {})
      mrRef.current   = mr
      chunksRef.current = []

      mr.ondataavailable = e => e.data.size > 0 && chunksRef.current.push(e.data)
      mr.onstop = () => {
        setBlob(new Blob(chunksRef.current, { type: mr.mimeType }))
        setIsRecording(false)
        analyserRef.current = null
        stream.getTracks().forEach(t => t.stop())
        ac.close()
        clearInterval(timerRef.current)
      }

      mr.start(100)
      setIsRecording(true)
      setElapsed(0)
      setBlob(null)

      const t0 = Date.now()
      timerRef.current = setInterval(
        () => setElapsed(Math.floor((Date.now() - t0) / 1000)),
        500
      )
    } catch {
      alert('No se puede acceder al micrófono.\nComprueba los permisos del navegador.')
    }
  }, [])

  const stop = useCallback(() => {
    if (mrRef.current?.state === 'recording') mrRef.current.stop()
  }, [])

  useEffect(() => () => {
    clearInterval(timerRef.current)
    streamRef.current?.getTracks().forEach(t => t.stop())
    acRef.current?.close()
  }, [])

  return { isRecording, elapsed, blob, analyserRef, start, stop }
}
