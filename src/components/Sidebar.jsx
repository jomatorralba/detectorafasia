import { useState } from 'react'

const IconTimer = () => (
  <svg width="15" height="15" viewBox="0 0 15 15" fill="none" stroke="currentColor"
    strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="7.5" cy="8.5" r="5.5"/>
    <path d="M7.5 5.5v3l2 1.3"/>
    <path d="M5.5 1h4M7.5 1v2"/>
  </svg>
)

const IconWave = () => (
  <svg width="15" height="15" viewBox="0 0 15 15" fill="none" stroke="currentColor"
    strokeWidth="1.5" strokeLinecap="round">
    <path d="M1 7.5Q3.75 2.5 5 7.5Q6.25 12.5 8.75 7.5Q10 2.5 11.25 7.5Q12.5 12.5 14 7.5"/>
  </svg>
)

const IconBook = () => (
  <svg width="15" height="15" viewBox="0 0 15 15" fill="none" stroke="currentColor"
    strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M2 3a1 1 0 0 1 1-1h9v11H3a1 1 0 0 1-1-1V3z"/>
    <path d="M2 11a1 1 0 0 0 1 1h9"/>
    <path d="M7.5 2v11"/>
  </svg>
)

const IconType = () => (
  <svg width="15" height="15" viewBox="0 0 15 15" fill="none" stroke="currentColor"
    strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M2 4h11M7.5 4v8M5 12h5"/>
  </svg>
)

const IconBars = () => (
  <svg width="15" height="15" viewBox="0 0 15 15" fill="none" stroke="currentColor"
    strokeWidth="1.5" strokeLinecap="round">
    <path d="M3 12V7M7.5 12V3M12 12V8.5"/>
    <path d="M1 12h13"/>
  </svg>
)

const TABS = [
  { id: 0, icon: <IconTimer />, label: 'TMF',                sub: 'Fonación máxima' },
  { id: 1, icon: <IconWave />,  label: 'Diadococinesias',   sub: 'PA·TA·KA' },
  { id: 2, icon: <IconBook />,  label: 'Lectura del Abuelo', sub: 'Velocidad lectora' },
  { id: 3, icon: <IconType />,  label: 'Palabras',           sub: 'Espectrograma' },
  { id: 4, icon: <IconBars />,  label: 'Resumen',            sub: 'Resultados' },
]

export function Sidebar({ active, onChange, doneCount }) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <aside
      className="flex flex-col shrink-0 h-screen transition-all duration-200 ease-in-out"
      style={{
        width: collapsed ? 52 : 216,
        background: '#0d1117',
        borderRight: '1px solid #21262d',
      }}
    >
      {/* Logo */}
      <div
        className="flex items-center gap-2.5 border-b"
        style={{
          padding: collapsed ? '14px 12px' : '14px 16px',
          borderColor: '#21262d',
          justifyContent: collapsed ? 'center' : 'flex-start',
        }}
      >
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="#388bfd"
          strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
          <ellipse cx="9" cy="9" rx="3" ry="6"/>
          <path d="M3 9a6 6 0 0 0 12 0"/>
          <path d="M9 15v2M6 17h6"/>
        </svg>
        {!collapsed && (
          <div style={{ lineHeight: 1.2 }}>
            <div style={{ color: '#e6edf3', fontWeight: 600, fontSize: 13 }}>Disartria</div>
            <div style={{ color: '#8b949e', fontSize: 11 }}>Evaluación clínica</div>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-1 overflow-y-auto overflow-x-hidden">
        {TABS.map(tab => {
          const isActive = active === tab.id
          return (
            <button
              key={tab.id}
              onClick={() => onChange(tab.id)}
              title={collapsed ? tab.label : undefined}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                padding: collapsed ? '9px 0' : '9px 16px',
                justifyContent: collapsed ? 'center' : 'flex-start',
                background: isActive ? 'rgba(31,111,235,0.12)' : 'transparent',
                borderRight: isActive ? '2px solid #388bfd' : '2px solid transparent',
                color: isActive ? '#e6edf3' : '#8b949e',
                fontSize: 13,
                cursor: 'pointer',
                textAlign: 'left',
                transition: 'background 0.1s, color 0.1s',
              }}
              onMouseEnter={e => {
                if (!isActive) {
                  e.currentTarget.style.background = '#161b22'
                  e.currentTarget.style.color = '#e6edf3'
                }
              }}
              onMouseLeave={e => {
                if (!isActive) {
                  e.currentTarget.style.background = 'transparent'
                  e.currentTarget.style.color = '#8b949e'
                }
              }}
            >
              <span style={{ flexShrink: 0, lineHeight: 1, display: 'flex', alignItems: 'center' }}>
                {tab.icon}
              </span>
              {!collapsed && (
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontWeight: isActive ? 600 : 400, lineHeight: 1.3, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {tab.label}
                  </div>
                  <div style={{ fontSize: 11, color: '#8b949e', lineHeight: 1.2 }}>{tab.sub}</div>
                </div>
              )}
            </button>
          )
        })}
      </nav>

      {/* Progress badge */}
      {!collapsed && doneCount > 0 && (
        <div style={{ padding: '8px 16px', borderTop: '1px solid #21262d' }}>
          <div style={{ background: '#161b22', borderRadius: 8, padding: '6px 10px', display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ flex: 1, background: '#21262d', borderRadius: 4, height: 4 }}>
              <div style={{ width: `${(doneCount / 4) * 100}%`, background: '#388bfd', height: 4, borderRadius: 4, transition: 'width 0.3s' }} />
            </div>
            <span style={{ color: '#8b949e', fontSize: 11, whiteSpace: 'nowrap' }}>{doneCount}/4</span>
          </div>
        </div>
      )}

      {/* Collapse toggle */}
      <button
        onClick={() => setCollapsed(c => !c)}
        style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: collapsed ? '12px 0' : '12px 16px',
          justifyContent: collapsed ? 'center' : 'flex-start',
          borderTop: '1px solid #21262d',
          color: '#8b949e', fontSize: 12, cursor: 'pointer',
          background: 'transparent', width: '100%',
          transition: 'color 0.1s, background 0.1s',
        }}
        onMouseEnter={e => { e.currentTarget.style.color = '#e6edf3'; e.currentTarget.style.background = '#161b22' }}
        onMouseLeave={e => { e.currentTarget.style.color = '#8b949e'; e.currentTarget.style.background = 'transparent' }}
      >
        <svg width="13" height="13" viewBox="0 0 13 13" fill="none" stroke="currentColor"
          strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          {collapsed
            ? <path d="M4 6.5h6M7 3.5l3 3-3 3"/>
            : <path d="M9 6.5H3M6 3.5l-3 3 3 3"/>}
        </svg>
        {!collapsed && 'Contraer'}
      </button>
    </aside>
  )
}
