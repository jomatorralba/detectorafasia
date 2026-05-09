import { useState, useCallback } from 'react'
import { Analytics } from '@vercel/analytics/react'
import { User } from 'lucide-react'
import { Sidebar }      from './components/Sidebar'
import { TMF }          from './pages/TMF'
import { DDK }          from './pages/DDK'
import { Abuelo }       from './pages/Abuelo'
import { Palabras }     from './pages/Palabras'
import { Prosodia }     from './pages/Prosodia'
import { Summary }      from './pages/Summary'
import { PatientsPage } from './pages/PatientsPage'
import { isConfigured } from './lib/supabase'

const PAGE_META = [
  { title: 'Tiempo Máximo de Fonación',  desc: 'Duración de /A/ y /S/ sostenidas · cociente S/A' },
  { title: 'Diadococinesias',            desc: 'Repetición de PA·TA·KA durante 5 segundos' },
  { title: 'Lectura del Abuelo',         desc: 'Velocidad lectora en palabras por minuto' },
  { title: 'Lectura de Palabras',        desc: 'Comparativa espectral con voz de referencia' },
  { title: 'Prosodia',                   desc: 'Variación de tono en tres modalidades de entonación' },
  { title: 'Resumen',                    desc: 'Resultados de todas las pruebas completadas' },
]

export default function App() {
  const [tab,              setTab]              = useState(0)
  const [results,          setResults]          = useState({})
  const [patient,          setPatient]          = useState(null)
  const [sessionKey,       setSessionKey]       = useState(0)
  const [patientPanelOpen, setPatientPanelOpen] = useState(false)

  const onResult = useCallback((key, value) => {
    setResults(r => ({ ...r, [key]: value }))
  }, [])

  const onReset = () => { setResults({}); setTab(0); setSessionKey(k => k + 1) }

  const handleSelectPatient = (p) => {
    setResults({}); setTab(0); setSessionKey(k => k + 1)
    setPatient(p)
    setPatientPanelOpen(false)
  }

  const openPatientPanel = isConfigured ? () => setPatientPanelOpen(true) : undefined

  const handleTabChange = (t) => { setTab(t); setPatientPanelOpen(false) }

  const meta = patientPanelOpen
    ? { title: 'Pacientes', desc: 'Seleccionar, crear o ver historial de pacientes' }
    : PAGE_META[tab]

  const doneCount = [
    results.tmf_a != null || results.tmf_s != null,
    results.ddk   != null,
    results.wpm   != null,
    Object.keys(results).some(k => k.startsWith('palabra_')),
    results.prosody != null,
  ].filter(Boolean).length

  return (
    <>
      <div style={{ display: 'flex', height: '100vh', overflow: 'hidden', background: '#fff' }}>
        <Sidebar
          active={tab}
          onChange={handleTabChange}
          doneCount={doneCount}
          onPatientsClick={openPatientPanel}
          patientsActive={patientPanelOpen}
        />

        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {/* cabecera */}
          <header style={{ padding: '14px 32px', borderBottom: '1px solid #f0f0f0', flexShrink: 0, background: '#fff' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div>
                <h1 style={{ margin: 0, fontSize: 17, fontWeight: 600, color: '#111', lineHeight: 1.3 }}>
                  {meta.title}
                </h1>
                <p style={{ margin: '2px 0 0', fontSize: 13, color: '#888' }}>{meta.desc}</p>
              </div>
              {patient ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginTop: 2, flexShrink: 0 }}>
                  <User size={13} color="#116b70"/>
                  <span style={{ fontSize: 13, fontWeight: 600, color: '#116b70' }}>{patient.name}</span>
                  {openPatientPanel && (
                    <button onClick={openPatientPanel}
                      style={{ fontSize: 11, color: '#bbb', textDecoration: 'underline', background: 'none', border: 'none', cursor: 'pointer', padding: 0, marginLeft: 2 }}>
                      cambiar
                    </button>
                  )}
                </div>
              ) : isConfigured ? (
                <button onClick={openPatientPanel}
                  style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 2, padding: '5px 12px', borderRadius: 8, border: '1px solid #d1d5db', background: '#fff', color: '#555', cursor: 'pointer', fontSize: 12, flexShrink: 0 }}>
                  <User size={12}/> Seleccionar paciente
                </button>
              ) : null}
            </div>
          </header>

          {/* contenido scrollable */}
          <main style={{ flex: 1, overflowY: 'auto', padding: '24px 32px' }}>
            {patientPanelOpen ? (
              <PatientsPage onSelectPatient={handleSelectPatient} embedded/>
            ) : (
              <div style={{ maxWidth: 680, margin: '0 auto' }}>
                {tab === 0 && <TMF      onResult={onResult}/>}
                {tab === 1 && <DDK      onResult={onResult}/>}
                {tab === 2 && <Abuelo   onResult={onResult}/>}
                {tab === 3 && <Palabras  onResult={onResult}/>}
                {tab === 4 && <Prosodia  onResult={onResult}/>}
                {tab === 5 && (
                  <Summary
                    key={`summary-${sessionKey}`}
                    results={results}
                    onReset={onReset}
                    patient={patient}
                    onNewPatient={openPatientPanel}
                  />
                )}
              </div>
            )}
          </main>
        </div>
      </div>

      <Analytics/>
    </>
  )
}
