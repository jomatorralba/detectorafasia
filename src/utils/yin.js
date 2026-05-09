// YIN pitch estimation — extraído de Prosodia.jsx para reutilización
// de Cheveigné & Kawahara (2002) "YIN, a fundamental frequency estimator
// for speech and music", JASA 111(4).

export function yinPitch(frame, sr) {
  const W  = frame.length
  const lo = Math.ceil(sr / 500)   // periodo para 500 Hz (tono más agudo)
  const hi = Math.floor(sr / 75)   // periodo para 75 Hz  (tono más grave)
  if (hi >= W / 2) return 0

  const d = new Float32Array(hi + 1)
  const lim = W - hi
  for (let τ = 1; τ <= hi; τ++) {
    let s = 0
    for (let i = 0; i < lim; i++) { const x = frame[i] - frame[i + τ]; s += x * x }
    d[τ] = s
  }

  const cmnd = new Float32Array(hi + 1); cmnd[0] = 1
  let rs = 0
  for (let τ = 1; τ <= hi; τ++) { rs += d[τ]; cmnd[τ] = rs > 0 ? d[τ] * τ / rs : 1 }

  for (let τ = lo; τ < hi; τ++) {
    if (cmnd[τ] < 0.15) {
      while (τ + 1 < hi && cmnd[τ + 1] < cmnd[τ]) τ++
      return sr / τ
    }
  }
  return 0
}
