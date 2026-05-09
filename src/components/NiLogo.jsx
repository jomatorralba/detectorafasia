export function NiLogo({ size = 28, color = 'white' }) {
  return (
    <svg width={size} height={Math.round(size * 118 / 100)} viewBox="0 0 100 118" fill="none">
      <path
        fillRule="evenodd" clipRule="evenodd"
        d="M36,46 A32,32 0 1 0 36,110 A32,32 0 1 0 36,46 Z
           M36,60 A14,18 0 1 0 36,96  A14,18 0 1 0 36,60  Z"
        fill={color}
      />
      <path d="M60,55 C68,38 76,24 82,18" stroke={color} strokeWidth="10" strokeLinecap="round" fill="none"/>
      <circle cx="82" cy="13" r="11" fill={color}/>
      <rect x="75.5" y="34" width="13" height="62" rx="6.5" fill={color}/>
    </svg>
  )
}

export function NeuroinnWordmark({ height = 34, color = 'white', subtitleColor = 'rgba(255,255,255,0.55)' }) {
  const scale = height / 34
  const w = Math.round(168 * scale)
  return (
    <svg width={w} height={height} viewBox="0 0 168 34" fill="none">
      <text
        x="0" y="23"
        fontFamily="'Nunito', system-ui, sans-serif"
        fontWeight="800"
        fontSize="23"
        letterSpacing="-0.4"
        fill={color}
      >neuroinn</text>

      <path d="M74,8 C79,3 85,1 91,1" stroke={color} strokeWidth="2.8" strokeLinecap="round"/>
      <circle cx="92" cy="1" r="4" fill={color}/>

      <text
        x="1" y="34"
        fontFamily="'Nunito', system-ui, sans-serif"
        fontWeight="400"
        fontSize="9"
        letterSpacing="0.8"
        fill={subtitleColor}
      >voz y lenguaje</text>
    </svg>
  )
}
