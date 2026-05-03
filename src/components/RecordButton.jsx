import { Mic, Square } from 'lucide-react'

export function RecordButton({ isRecording, elapsed, onStart, onStop }) {
  const mm = String(Math.floor(elapsed / 60)).padStart(2, '0')
  const ss = String(elapsed % 60).padStart(2, '0')

  return (
    <div className="flex flex-col items-center gap-3">
      <button
        onClick={isRecording ? onStop : onStart}
        className={`relative w-20 h-20 rounded-full flex flex-col items-center justify-center
          text-white font-bold shadow-lg transition-all duration-150 active:scale-95
          ${isRecording
            ? 'bg-red-500 hover:bg-red-600'
            : 'bg-blue-600 hover:bg-blue-700'}`}
      >
        {isRecording && (
          <span className="absolute inset-0 rounded-full bg-red-400 animate-ping opacity-30" />
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
