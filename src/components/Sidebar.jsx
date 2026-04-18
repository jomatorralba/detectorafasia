import { useState } from 'react'

const TABS = [
  { id: 0, icon: '⏱', label: 'TMF',               sub: 'Fonación máxima' },
  { id: 1, icon: '🗣', label: 'Diadococinesias',   sub: 'PA·TA·KA' },
  { id: 2, icon: '📖', label: 'Lectura del Abuelo', sub: 'Velocidad lectora' },
  { id: 3, icon: '🔤', label: 'Palabras',           sub: 'Espectrograma' },
  { id: 4, icon: '📊', label: 'Resumen',            sub: 'Resultados' },
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
        <span style={{ fontSize: 20 }}>🎙️</span>
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
              <span style={{ fontSize: 16, flexShrink: 0, lineHeight: 1 }}>{tab.icon}</span>
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
        <span style={{ fontSize: 14 }}>{collapsed ? '›' : '‹'}</span>
        {!collapsed && 'Contraer'}
      </button>
    </aside>
  )
}
