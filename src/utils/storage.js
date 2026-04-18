const REFS_KEY  = 'disartria_refs'
const WORDS_KEY = 'disartria_words'
const DEFAULT_WORDS = ['PIANO', 'LAICO', 'CALENDARIO', 'AUTOBIOGRAFÍA']

export function loadWords() {
  try { return JSON.parse(localStorage.getItem(WORDS_KEY)) || DEFAULT_WORDS }
  catch { return DEFAULT_WORDS }
}

export function saveWords(words) {
  localStorage.setItem(WORDS_KEY, JSON.stringify(words))
}

export function addWord(word) {
  const words = loadWords()
  if (!words.includes(word)) saveWords([...words, word])
}

export function deleteWord(word) {
  saveWords(loadWords().filter(w => w !== word))
  const refs = loadAllRefs()
  delete refs[word]
  localStorage.setItem(REFS_KEY, JSON.stringify(refs))
}

function loadAllRefs() {
  try { return JSON.parse(localStorage.getItem(REFS_KEY)) || {} }
  catch { return {} }
}

export function loadReference(word) {
  return loadAllRefs()[word] || null
}

export function saveReference(word, audioBase64, mimeType) {
  const refs = loadAllRefs()
  refs[word] = { audio: audioBase64, mimeType, savedAt: Date.now() }
  localStorage.setItem(REFS_KEY, JSON.stringify(refs))
}

export function deleteReference(word) {
  const refs = loadAllRefs()
  delete refs[word]
  localStorage.setItem(REFS_KEY, JSON.stringify(refs))
}

export function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload  = () => resolve(reader.result.split(',')[1])
    reader.onerror = reject
    reader.readAsDataURL(blob)
  })
}

export function base64ToBlob(base64, mimeType) {
  const bytes = atob(base64)
  const arr   = new Uint8Array(bytes.length)
  for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i)
  return new Blob([arr], { type: mimeType || 'audio/webm' })
}
