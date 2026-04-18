import { useState, useCallback } from 'react'
import { Sidebar }  from './components/Sidebar'
import { TMF }      from './pages/TMF'
import { DDK }      from './pages/DDK'
import { Abuelo }   from './pages/Abuelo'
import { Palabras } from './pages/Palabras'
import { Summary }  from './pages/Summary'

const PAGE_META = [
  { title: 'Tiempo Máximo de Fonación',  desc: 'Duración de /A/ y /S/ sostenidas · cociente S/A' },
  { title: 'Diadococinesias',            desc: 'Repetición de PA·TA·KA durante 5 segundos' },
  { title: 'Lectura del Abuelo',         desc: 'Velocidad lectora en palabras por minuto' },
  { title: 'Lectura de Palabras',        desc: 'Comparativa espectral con voz de referencia' },
  { title: 'Resumen',                    desc: 'Resultados de todas las pruebas completadas' },
]

export default function App() {
  const [tab, setTab]         = useState(0)
  const [results, setResults] = useState({})

  const onResult = useCallback((key, value) => {
    setResults(r => ({ ...r, [key]: value }))
  }, [])

  const onReset = () => { setResults({}); setTab(0) }

  const pages = [
    <TMF      key="tmf"      onResult={onResult} />,
    <DDK      key="ddk"      onResult={onResult} />,
    <Abuelo   key="abuelo"   onResult={onResult} />,
    <Palabras key="palabras" onResult={onResult} />,
    <Summary  key="summary"  results={results} onReset={onReset} />,
  ]

  const doneCount = [
    results.tmf_a != null || results.tmf_s != null,
    results.ddk   != null,
    results.wpm   != null,
    Object.keys(results).some(k => k.startsWith('palabra_')),
  ].filter(Boolean).length

  const meta = PAGE_META[tab]

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden', background: '#fff' }}>
      <Sidebar active={tab} onChange={setTab} doneCount={doneCount} />

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Page header */}
        <header style={{
          padding: '16px 32px',
          borderBottom: '1px solid #f0f0f0',
          flexShrink: 0,
          background: '#fff',
        }}>
          <h1 style={{ margin: 0, fontSize: 17, fontWeight: 600, color: '#111', lineHeight: 1.3 }}>
            {meta.title}
          </h1>
          <p style={{ margin: '2px 0 0', fontSize: 13, color: '#888' }}>{meta.desc}</p>
        </header>

        {/* Scrollable content */}
        <main style={{ flex: 1, overflowY: 'auto', padding: '24px 32px' }}>
          <div style={{ maxWidth: 680, margin: '0 auto' }}>
            {pages[tab]}
          </div>
        </main>
      </div>
    </div>
  )
}
