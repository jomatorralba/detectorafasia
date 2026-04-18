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
  'Abuelo viene a visitarnos todos los domingos. ' +
  'Siempre trae caramelos para los niños y flores para la abuela. ' +
  'Le gusta sentarse en el jardín y contarnos historias de cuando era joven. ' +
  'Dice que antes la vida era más sencilla pero también más dura. ' +
  'Trabajaba en el campo desde muy temprano hasta que se ponía el sol. ' +
  'Ahora descansa y cuida su huerto con mucho cariño. ' +
  'Los tomates de su huerto son los más ricos que he probado nunca. ' +
  'Cuando nos vamos a casa siempre nos da un abrazo muy fuerte y cariñoso.'

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
  good: { icon: '🟢', label: 'Normal',   css: 'border-green-400 bg-green-50 text-green-900' },
  warn: { icon: '🟡', label: 'Límite',   css: 'border-yellow-400 bg-yellow-50 text-yellow-900' },
  bad:  { icon: '🔴', label: 'Alterado', css: 'border-red-400 bg-red-50 text-red-900' },
}
