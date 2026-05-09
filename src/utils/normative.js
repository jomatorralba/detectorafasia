export const NORM = {
  // Aerodinámica fonatoria
  tmf_a: { min: 10 },
  tmf_s: { min: 8  },
  ratio: { min: 0.7, max: 1.4 },
  // Diadococinesias
  ddk:   { min: 5  },
  // Velocidad lectora
  wpm:   { min: 100 },
  // Calidad vocal — Titze (1995); Boersma (1993)
  jitter:  { max: 1.04 },   // % local — < 1.04 % normal
  shimmer: { max: 3.81 },   // % local — < 3.81 % normal
  hnr:     { min: 20   },   // dB      — > 20 dB normal
  // Pausas en lectura — Tjaden & Wilding (2004); Duffy (2013)
  silence_pct:       { min: 0.15, max: 0.25 },  // 15–25 % normal
  mean_pause_dur:    { max: 0.6  },              // < 0.6 s normal
  articulation_rate: { min: 4.5  },              // ≥ 4.5 síl/s normal
  // Espacio vocálico — Sapir et al. (2010)
  fcr: { max: 1.17 },  // < 1.17 normal; > 1.20 indica centralización vocálica
}

export const PALABRAS = {
  PIANO:         { syl: 2, ref: 0.65 },
  LAICO:         { syl: 3, ref: 0.75 },
  CALENDARIO:    { syl: 5, ref: 1.10 },
  AUTOBIOGRAFÍA: { syl: 7, ref: 1.55 },
}

export const ABUELO_TEXT =
  'Usted quiere saber sobre mi abuelo. Bueno, él tiene cerca de noventa y tres años de edad y aún piensa tan lúcidamente como siempre. ' +
  'Se viste solo y se pone su vieja chaqueta negra, que comúnmente tiene varios botones menos. ' +
  'Una barba larga cuelga de su cara, inspirando a aquellos que lo observan, un profundo sentimiento de respeto. ' +
  'Cuando habla, su voz parece un poco quebrada y temblorosa. ' +
  'Dos veces al día, él disfruta tocando hábilmente un pequeño órgano. ' +
  'Todos los días, el abuelo da un corto paseo, excepto en el invierno, cuando la lluvia o el frío se lo impiden.'

export const ABUELO_WORDS = ABUELO_TEXT.split(' ').length

export function classify(value, key, mode = 'min') {
  const n = NORM[key]
  if (!n) return 'warn'
  if (mode === 'min') {
    if (value >= n.min)        return 'good'
    if (value >= n.min * 0.7)  return 'warn'
    return 'bad'
  }
  if (mode === 'max') {
    if (value <= n.max)        return 'good'
    if (value <= n.max * 1.4)  return 'warn'
    return 'bad'
  }
  // range
  if (value >= n.min && value <= n.max)               return 'good'
  if (value >= n.min * 0.7 && value <= n.max * 1.3)  return 'warn'
  return 'bad'
}

export const STATUS = {
  good: { dot: 'bg-green-400',  label: 'Normal',   css: 'border-green-300 bg-green-50 text-green-800' },
  warn: { dot: 'bg-yellow-400', label: 'Límite',   css: 'border-yellow-300 bg-yellow-50 text-yellow-800' },
  bad:  { dot: 'bg-red-400',    label: 'Alterado', css: 'border-red-300 bg-red-50 text-red-800' },
}
