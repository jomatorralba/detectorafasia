import { useState, useCallback } from 'react'
import { TMF }     from './pages/TMF'
import { DDK }     from './pages/DDK'
import { Abuelo }  from './pages/Abuelo'
import { Palabras } from './pages/Palabras'
import { Summary } from './pages/Summary'

const TABS = [
  { id: 0, icon: '⏱️', label: 'TMF' },
  { id: 1, icon: '🗣️', label: 'Diadococinesias' },
  { id: 2, icon: '📖', label: 'Lectura del Abuelo' },
  { id: 3, icon: '🔤', label: 'Palabras' },
  { id: 4, icon: '📊', label: 'Resumen' },
]

export default function App() {
  const [tab, setTab]         = useState(0)
  const [results, setResults] = useState({})

  const onResult = useCallback((key, value) => {
    setResults(r => ({ ...r, [key]: value }))
  }, [])

  const onReset = () => {
    setResults({})
    setTab(0)
  }

  const pages = [
    <TMF      key="tmf"     onResult={onResult} />,
    <DDK      key="ddk"     onResult={onResult} />,
    <Abuelo   key="abuelo"  onResult={onResult} />,
    <Palabras key="palabras" onResult={onResult} />,
    <Summary  key="summary" results={results} onReset={onReset} />,
  ]

  const doneCount = [
    results.tmf_a != null || results.tmf_s != null,
    results.ddk   != null,
    results.wpm   != null,
    Object.keys(results).some(k => k.startsWith('palabra_')),
  ].filter(Boolean).length

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-100 shadow-sm sticky top-0 z-10">
        <div className="max-w-3xl mx-auto px-4 py-3 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-gray-900">🎙️ Evaluación de Disartria</h1>
            <p className="text-xs text-gray-400">Herramienta clínica para logopedas</p>
          </div>
          {doneCount > 0 && (
            <span className="text-xs bg-blue-100 text-blue-700 font-semibold px-2.5 py-1 rounded-full">
              {doneCount}/4 pruebas
            </span>
          )}
        </div>

        {/* Tab bar */}
        <div className="max-w-3xl mx-auto px-4 flex gap-1 overflow-x-auto pb-0 scrollbar-hide">
          {TABS.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`flex items-center gap-1.5 px-3 py-2.5 text-sm font-medium
                whitespace-nowrap border-b-2 transition-colors duration-150
                ${tab === t.id
                  ? 'border-blue-600 text-blue-700'
                  : 'border-transparent text-gray-500 hover:text-gray-700'}`}
            >
              <span>{t.icon}</span>
              <span className="hidden sm:inline">{t.label}</span>
            </button>
          ))}
        </div>
      </header>

      {/* Content */}
      <main className="max-w-3xl mx-auto px-4 py-6">
        {pages[tab]}
      </main>

      <footer className="max-w-3xl mx-auto px-4 py-6 text-center text-xs text-gray-300">
        Esta herramienta no sustituye la valoración clínica profesional
      </footer>
    </div>
  )
}
