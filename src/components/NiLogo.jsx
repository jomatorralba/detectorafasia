export function NiLogo({ size = 28, color = 'white' }) {
  return (
    <svg width={size} height={Math.round(size * 118 / 100)} viewBox="0 0 100 118" fill="none">
      {/* "o" donut — círculo exterior + oval interior, evenodd crea el hueco */}
      <path
        fillRule="evenodd" clipRule="evenodd"
        d="M36,46 A32,32 0 1 0 36,110 A32,32 0 1 0 36,46 Z
           M36,60 A14,18 0 1 0 36,96  A14,18 0 1 0 36,60  Z"
        fill={color}
      />
      {/* brazo curvo del anillo al punto */}
      <path d="M60,55 C68,38 76,24 82,18" stroke={color} strokeWidth="10" strokeLinecap="round" fill="none"/>
      {/* punto de la i */}
      <circle cx="82" cy="13" r="11" fill={color}/>
      {/* barra de la i */}
      <rect x="75.5" y="34" width="13" height="62" rx="6.5" fill={color}/>
    </svg>
  )
}
