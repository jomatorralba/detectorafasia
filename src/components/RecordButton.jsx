import { Mic, Square } from 'lucide-react'

export function RecordButton({ isRecording, elapsed, onStart, onStop }) {
  const mm = String(Math.floor(elapsed / 60)).padStart(2, '0')
  const ss = String(elapsed % 60).padStart(2, '0')

  return (
    <div className="flex flex-col items-center gap-3">
      <button
        onClick={isRecording ? onStop : onStart}
        className={`relative w-20 h-20 rounded-full flex flex-col items-center justify-center
          text-white font-bold shadow-lg transition-all duration-150 active:scale-95`}
        style={{ background: isRecording ? '#b91c1c' : '#dc2626', boxShadow: isRecording ? 'none' : '0 0 0 4px rgba(220,38,38,0.15)' }}
      >
        {isRecording && (
          <span className="absolute inset-0 rounded-full animate-ping opacity-25" style={{ background: '#dc2626' }} />
        )}
        <span className="relative z-10">
          {isRecording
            ? <Square size={24} strokeWidth={2} fill="white"/>
            : <Mic size={24} strokeWidth={2}/>}
        </span>
        <span className="relative z-10 text-[10px] mt-0.5 tracking-wide">
          {isRecording ? 'STOP' : 'GRABAR'}
        </span>
      </button>

      {isRecording && (
        <span className="text-2xl font-mono font-bold text-red-500 tabular-nums">
          {mm}:{ss}
        </span>
      )}
    </div>
  )
}
