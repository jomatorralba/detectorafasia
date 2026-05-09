import { useState } from 'react'
import { Timer, AudioLines, BookOpen, AudioWaveform, BarChart2, Users } from 'lucide-react'
import { NiLogo } from './NiLogo'

const C = {
  bg:     '#073447',
  border: 'rgba(255,255,255,0.09)',
  hover:  'rgba(255,255,255,0.05)',
  active: 'rgba(255,255,255,0.10)',
  text:   'rgba(255,255,255,0.92)',
  muted:  'rgba(255,255,255,0.42)',
}

const TABS = [
  { id: 0, icon: <Timer size={15} strokeWidth={1.6}/>,         label: 'TMF',                sub: 'Fonación máxima' },
  { id: 1, icon: <AudioLines size={15} strokeWidth={1.6}/>,    label: 'Diadococinesias',   sub: 'PA·TA·KA' },
  { id: 2, icon: <BookOpen size={15} strokeWidth={1.6}/>,      label: 'Lectura del Abuelo', sub: 'Velocidad lectora' },
  { id: 3, icon: <AudioWaveform size={15} strokeWidth={1.6}/>, label: 'Palabras',           sub: 'Espectrograma' },
  { id: 4, icon: <BarChart2 size={15} strokeWidth={1.6}/>,     label: 'Resumen',            sub: 'Resultados' },
]

export function Sidebar({ active, onChange, doneCount, onPatientsClick }) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <aside
      className="flex flex-col shrink-0 h-screen transition-all duration-200 ease-in-out"
      style={{ width: collapsed ? 52 : 220, background: C.bg, borderRight: `1px solid ${C.border}` }}
    >
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10,
        padding: collapsed ? '13px 12px' : '13px 16px',
        borderBottom: `1px solid ${C.border}`,
        justifyContent: collapsed ? 'center' : 'flex-start',
      }}>
        <NiLogo size={26} color="white"/>
        {!collapsed && (
          <div style={{ lineHeight: 1.2 }}>
            <div style={{ color: C.text, fontWeight: 700, fontSize: 14, letterSpacing: '-0.2px' }}>
              Disartria
            </div>
            <div style={{ color: C.muted, fontSize: 10, letterSpacing: '0.04em', marginTop: 1 }}>
              neuroinn
            </div>
          </div>
        )}
      </div>

      <nav style={{ flex: 1, overflowY: 'auto', overflowX: 'hidden', padding: '4px 0' }}>
        {TABS.map(tab => {
          const isActive = active === tab.id
          return (
            <button
              key={tab.id}
              onClick={() => onChange(tab.id)}
              title={collapsed ? tab.label : undefined}
              style={{
                width: '100%', display: 'flex', alignItems: 'center', gap: 10,
                padding: collapsed ? '9px 0' : '9px 16px',
                justifyContent: collapsed ? 'center' : 'flex-start',
                background: isActive ? C.active : 'transparent',
                borderRight: isActive ? '2px solid rgba(255,255,255,0.38)' : '2px solid transparent',
                color: isActive ? C.text : C.muted,
                fontSize: 13, cursor: 'pointer', textAlign: 'left',
                transition: 'background 0.12s, color 0.12s',
              }}
              onMouseEnter={e => {
                if (!isActive) { e.currentTarget.style.background = C.hover; e.currentTarget.style.color = C.text }
              }}
              onMouseLeave={e => {
                if (!isActive) { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = C.muted }
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
                  <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.2 }}>{tab.sub}</div>
                </div>
              )}
            </button>
          )
        })}
      </nav>

      {/* Botón pacientes — anclado abajo */}
      {onPatientsClick && (
        <div style={{ borderTop: `1px solid ${C.border}` }}>
          <button
            onClick={onPatientsClick}
            title={collapsed ? 'Pacientes' : undefined}
            style={{
              width: '100%', display: 'flex', alignItems: 'center', gap: 10,
              padding: collapsed ? '9px 0' : '9px 16px',
              justifyContent: collapsed ? 'center' : 'flex-start',
              background: 'transparent', border: 'none',
              color: C.muted, fontSize: 13, cursor: 'pointer',
              transition: 'background 0.12s, color 0.12s',
            }}
            onMouseEnter={e => { e.currentTarget.style.background = C.hover; e.currentTarget.style.color = C.text }}
            onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = C.muted }}
          >
            <span style={{ flexShrink: 0, display: 'flex', alignItems: 'center' }}>
              <Users size={15} strokeWidth={1.6}/>
            </span>
            {!collapsed && (
              <div style={{ minWidth: 0 }}>
                <div style={{ lineHeight: 1.3 }}>Pacientes</div>
                <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.2 }}>Gestión</div>
              </div>
            )}
          </button>
        </div>
      )}

      {!collapsed && doneCount > 0 && (
        <div style={{ padding: '8px 16px', borderTop: `1px solid ${C.border}` }}>
          <div style={{
            background: 'rgba(255,255,255,0.06)', borderRadius: 8, padding: '6px 10px',
            display: 'flex', alignItems: 'center', gap: 8,
          }}>
            <div style={{ flex: 1, background: 'rgba(255,255,255,0.1)', borderRadius: 4, height: 3 }}>
              <div style={{
                width: `${(doneCount / 4) * 100}%`,
                background: 'rgba(255,255,255,0.55)', height: 3, borderRadius: 4, transition: 'width 0.3s',
              }} />
            </div>
            <span style={{ color: C.muted, fontSize: 11, whiteSpace: 'nowrap' }}>{doneCount}/4</span>
          </div>
        </div>
      )}

      <button
        onClick={() => setCollapsed(c => !c)}
        style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: collapsed ? '10px 0' : '10px 16px',
          justifyContent: collapsed ? 'center' : 'flex-start',
          borderTop: `1px solid ${C.border}`,
          color: C.muted, fontSize: 12, cursor: 'pointer',
          background: 'transparent', width: '100%',
          transition: 'color 0.1s, background 0.1s',
        }}
        onMouseEnter={e => { e.currentTarget.style.color = C.text; e.currentTarget.style.background = C.hover }}
        onMouseLeave={e => { e.currentTarget.style.color = C.muted; e.currentTarget.style.background = 'transparent' }}
      >
        <svg width="13" height="13" viewBox="0 0 13 13" fill="none" stroke="currentColor"
          strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          {collapsed
            ? <path d="M4 6.5h6M7 3.5l3 3-3 3"/>
            : <path d="M9 6.5H3M6 3.5l-3 3 3 3"/>}
        </svg>
        {!collapsed && 'Contraer'}
      </button>

      {!collapsed && (
        <div style={{ padding: '8px 16px 10px', borderTop: `1px solid ${C.border}` }}>
          <div style={{ fontSize: 9, color: 'rgba(255,255,255,0.22)', lineHeight: 1.6, letterSpacing: '0.01em' }}>
            <span style={{ color: 'rgba(255,255,255,0.4)', fontWeight: 600, fontSize: 9, letterSpacing: '0.06em' }}>
              NEUROINN
            </span>
            {' '}Rehabilitación Neurológica
            <br />
            J.M. Torralba-Muñoz · M. Zapata-Soria
          </div>
        </div>
      )}
    </aside>
  )
}
