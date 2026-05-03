export const NORM = {
  tmf_a: { min: 10 },
  tmf_s: { min: 8  },
  ratio: { min: 0.7, max: 1.4 },
  ddk:   { min: 5  },
  wpm:   { min: 100 },
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
  if (mode === 'min') {
    if (value >= n.min)         return 'good'
    if (value >= n.min * 0.7)   return 'warn'
    return 'bad'
  }
  if (value >= n.min && value <= n.max)                     return 'good'
  if (value >= n.min * 0.7 && value <= n.max * 1.3)        return 'warn'
  return 'bad'
}

export const STATUS = {
  good: { dot: 'bg-green-400',  label: 'Normal',   css: 'border-green-300 bg-green-50 text-green-800' },
  warn: { dot: 'bg-yellow-400', label: 'Límite',   css: 'border-yellow-300 bg-yellow-50 text-yellow-800' },
  bad:  { dot: 'bg-red-400',    label: 'Alterado', css: 'border-red-300 bg-red-50 text-red-800' },
}
