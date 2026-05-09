// Neuroinn isotipo: la "d" con brazo curvo y punto + barra "i"
export function NiLogo({ size = 28, color = 'white' }) {
  return (
    <svg width={size} height={Math.round(size * 48 / 50)} viewBox="0 0 50 48" fill="none">
      {/* anillo 'o' con agujero (evenodd) */}
      <path
        fillRule="evenodd" clipRule="evenodd"
        d="M16,19 A12,12 0,0,1 16,43 A12,12 0,0,1 16,19 Z M16,24.5 A6.5,6.5 0,0,1 16,37.5 A6.5,6.5 0,0,1 16,24.5 Z"
        fill={color}
      />
      {/* brazo curvo del anillo al punto */}
      <path d="M24,23 C29,14 35,8 40,5.5" stroke={color} strokeWidth="4.5" strokeLinecap="round" fill="none"/>
      {/* punto superior (cabeza de la 'i') */}
      <circle cx="40" cy="5.5" r="4.5" fill={color}/>
      {/* barra de la 'i' */}
      <rect x="36.25" y="13" width="7.5" height="30" rx="3.75" fill={color}/>
    </svg>
  )
}
