import { useState, useRef, useCallback, useEffect } from 'react'
import init, { analyze_audio, compute_spectrogram, spectrogram_num_bins } from './wasm/analysis'

const NOTES = ['C6', 'D6', 'E6', 'F6', 'G6', 'A6', 'B6', 'C7'] as const
type NoteIndex = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7

const NOTE_FREQS: Record<string, number> = {
  C6: 1046.50, D6: 1174.66, E6: 1318.51, F6: 1396.91,
  G6: 1567.98, A6: 1760.00, B6: 1975.53, C7: 2093.00,
}

const NOTE_COLORS: Record<NoteIndex, string> = {
  0: '#FF6B6B', 1: '#FF9F40', 2: '#FFD93D', 3: '#6BCB77',
  4: '#4ECDC4', 5: '#45B7D1', 6: '#A855F7', 7: '#FF6B9D',
}

const SPEC_FREQ_MIN = 500
const SPEC_FREQ_MAX = 2500
const SPEC_FREQ_TICKS = [500, 1000, 1500, 2000, 2500] as const

type AnalysisState =
  | { type: 'idle' }
  | { type: 'analyzing'; filename: string }
  | {
      type: 'done'
      filename: string
      audioSrc: string
      frames: Uint8Array
      spectrogram: Float32Array
      spectrogramBins: number
      durationSec: number
    }
  | { type: 'error'; message: string }

const NO_PREDICTION = 255 // sentinel from WASM for silent frames
const FRAME_WIDTH_PX = 4
const ROW_HEIGHT_PX = 28
const CANVAS_HEIGHT = NOTES.length * ROW_HEIGHT_PX
const SPEC_HEIGHT_PX = 160

function infernoColor(t: number): readonly [number, number, number] {
  const stops = [
    [0.00,   0,   0,   4],
    [0.25,  58,   9,  99],
    [0.50, 188,  55,  84],
    [0.75, 252, 137,  97],
    [1.00, 252, 255, 164],
  ] as const
  if (t <= 0) return [0, 0, 4]
  if (t >= 1) return [252, 255, 164]
  let i = 1
  while (i < stops.length - 1 && (stops[i]?.[0] ?? 0) < t) i++
  const s0 = stops[i - 1]
  const s1 = stops[i]
  if (!s0 || !s1) return [0, 0, 0]
  const u = (t - s0[0]) / (s1[0] - s0[0])
  return [
    Math.round(s0[1] + u * (s1[1] - s0[1])),
    Math.round(s0[2] + u * (s1[2] - s0[2])),
    Math.round(s0[3] + u * (s1[3] - s0[3])),
  ]
}

export default function App() {
  const [wasmReady, setWasmReady] = useState(false)
  const [state, setState] = useState<AnalysisState>({ type: 'idle' })
  const [dragOver, setDragOver] = useState(false)
  const pianoRollRef = useRef<HTMLCanvasElement>(null)
  const spectrogramRef = useRef<HTMLCanvasElement>(null)
  const cursorCanvasRef = useRef<HTMLCanvasElement>(null)
  const audioRef = useRef<HTMLAudioElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const audioUrlRef = useRef<string | null>(null)
  const specPanelRef = useRef<HTMLDivElement>(null)
  const rollPanelRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    init().then(() => setWasmReady(true)).catch((e: unknown) => {
      setState({ type: 'error', message: `Failed to load WASM: ${String(e)}` })
    })
    return () => {
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current)
    }
  }, [])

  const processFile = useCallback(async (file: File) => {
    if (!wasmReady) return
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current)
      audioUrlRef.current = null
    }
    setState({ type: 'analyzing', filename: file.name })
    try {
      const arrayBuffer = await file.arrayBuffer()
      const audioCtx = new AudioContext()
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer)
      await audioCtx.close()

      const numChannels = audioBuffer.numberOfChannels
      const length = audioBuffer.length
      const mono = new Float32Array(length)
      for (let c = 0; c < numChannels; c++) {
        const channel = audioBuffer.getChannelData(c)
        for (let i = 0; i < length; i++) {
          mono[i] = (mono[i] ?? 0) + (channel[i] ?? 0)
        }
      }
      for (let i = 0; i < length; i++) {
        mono[i] = (mono[i] ?? 0) / numChannels
      }

      const sr = audioBuffer.sampleRate
      const frames = analyze_audio(mono, sr)
      const spectrogram = compute_spectrogram(mono, sr)
      const spectrogramBins = spectrogram_num_bins(sr)
      const audioSrc = URL.createObjectURL(file)
      audioUrlRef.current = audioSrc

      setState({
        type: 'done',
        filename: file.name,
        audioSrc,
        frames,
        spectrogram,
        spectrogramBins,
        durationSec: audioBuffer.duration,
      })
    } catch (e: unknown) {
      setState({ type: 'error', message: `Analysis failed: ${String(e)}` })
    }
  }, [wasmReady])

  // Draw piano roll
  useEffect(() => {
    if (state.type !== 'done') return
    const canvas = pianoRollRef.current
    if (!canvas) return
    const { frames } = state
    const numFrames = frames.length
    const width = Math.max(numFrames * FRAME_WIDTH_PX, 600)
    canvas.width = width
    canvas.height = CANVAS_HEIGHT
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.fillStyle = '#111'
    ctx.fillRect(0, 0, width, CANVAS_HEIGHT)
    for (let n = 0; n < NOTES.length; n++) {
      ctx.fillStyle = n % 2 === 0 ? '#161616' : '#111'
      ctx.fillRect(0, n * ROW_HEIGHT_PX, width, ROW_HEIGHT_PX)
    }
    for (let f = 0; f < numFrames; f++) {
      const raw = frames[f] ?? NO_PREDICTION
      if (raw === NO_PREDICTION) continue
      const noteIdx = raw as NoteIndex
      const row = NOTES.length - 1 - noteIdx // C7 at top, C6 at bottom
      ctx.fillStyle = NOTE_COLORS[noteIdx]
      ctx.fillRect(f * FRAME_WIDTH_PX, row * ROW_HEIGHT_PX + 2, FRAME_WIDTH_PX - 1, ROW_HEIGHT_PX - 4)
    }
  }, [state])

  // Draw spectrogram
  useEffect(() => {
    if (state.type !== 'done') return
    const canvas = spectrogramRef.current
    if (!canvas) return
    const { spectrogram, spectrogramBins: numBins, frames } = state
    const numFrames = frames.length
    const width = Math.max(numFrames * FRAME_WIDTH_PX, 600)
    canvas.width = width
    canvas.height = SPEC_HEIGHT_PX
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const imageData = ctx.createImageData(width, SPEC_HEIGHT_PX)
    const px = imageData.data
    for (let f = 0; f < numFrames; f++) {
      const xStart = f * FRAME_WIDTH_PX
      for (let y = 0; y < SPEC_HEIGHT_PX; y++) {
        const binIdx = Math.round((1 - y / (SPEC_HEIGHT_PX - 1)) * (numBins - 1))
        const val = spectrogram[f * numBins + binIdx] ?? 0
        const [r, g, b] = infernoColor(val)
        for (let dx = 0; dx < FRAME_WIDTH_PX; dx++) {
          const idx = (y * width + xStart + dx) * 4
          px[idx] = r
          px[idx + 1] = g
          px[idx + 2] = b
          px[idx + 3] = 255
        }
      }
    }
    ctx.putImageData(imageData, 0, 0)

    ctx.setLineDash([3, 4])
    ctx.lineWidth = 1
    for (const [note, freq] of Object.entries(NOTE_FREQS)) {
      const yFrac = 1 - (freq - SPEC_FREQ_MIN) / (SPEC_FREQ_MAX - SPEC_FREQ_MIN)
      const y = yFrac * SPEC_HEIGHT_PX
      const noteIdx = NOTES.indexOf(note as typeof NOTES[number]) as NoteIndex
      ctx.strokeStyle = NOTE_COLORS[noteIdx]
      ctx.globalAlpha = 0.6
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }
    ctx.globalAlpha = 1
    ctx.setLineDash([])
  }, [state])

  // Playback cursor animation
  useEffect(() => {
    if (state.type !== 'done') return
    const audio = audioRef.current
    const canvas = cursorCanvasRef.current
    if (!audio || !canvas) return

    const canvasWidth = Math.max(state.frames.length * FRAME_WIDTH_PX, 600)
    canvas.width = canvasWidth
    canvas.height = SPEC_HEIGHT_PX

    let rafId: number | null = null

    const cursorX = () =>
      Math.round((audio.currentTime / state.durationSec) * canvasWidth)

    const drawCursor = () => {
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.clearRect(0, 0, canvasWidth, SPEC_HEIGHT_PX)
      const x = cursorX()
      ctx.save()
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)'
      ctx.lineWidth = 2
      ctx.shadowColor = 'rgba(255, 255, 255, 0.5)'
      ctx.shadowBlur = 6
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, SPEC_HEIGHT_PX)
      ctx.stroke()
      ctx.restore()
    }

    const scrollToCursor = () => {
      const x = cursorX()
      for (const el of [specPanelRef.current, rollPanelRef.current]) {
        if (!el) continue
        const target = Math.max(0, x - el.clientWidth * 0.3)
        el.scrollLeft = target
      }
    }

    const loop = () => {
      drawCursor()
      scrollToCursor()
      rafId = requestAnimationFrame(loop)
    }

    const handlePlay = () => {
      if (rafId !== null) cancelAnimationFrame(rafId)
      loop()
    }

    const handlePauseOrEnd = () => {
      if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null }
      drawCursor()
    }

    const handleSeeked = () => {
      // Redraw immediately when scrubbing (RAF loop handles it during playback)
      if (rafId === null) drawCursor()
    }

    audio.addEventListener('play', handlePlay)
    audio.addEventListener('pause', handlePauseOrEnd)
    audio.addEventListener('ended', handlePauseOrEnd)
    audio.addEventListener('seeked', handleSeeked)

    drawCursor() // initial position at t=0

    return () => {
      if (rafId !== null) cancelAnimationFrame(rafId)
      audio.removeEventListener('play', handlePlay)
      audio.removeEventListener('pause', handlePauseOrEnd)
      audio.removeEventListener('ended', handlePauseOrEnd)
      audio.removeEventListener('seeked', handleSeeked)
    }
  }, [state])

  // Sync scroll position between the two canvas panels
  useEffect(() => {
    if (state.type !== 'done') return
    const specEl = specPanelRef.current
    const rollEl = rollPanelRef.current
    if (!specEl || !rollEl) return

    const onSpecScroll = () => {
      if (rollEl.scrollLeft !== specEl.scrollLeft)
        rollEl.scrollLeft = specEl.scrollLeft
    }
    const onRollScroll = () => {
      if (specEl.scrollLeft !== rollEl.scrollLeft)
        specEl.scrollLeft = rollEl.scrollLeft
    }

    specEl.addEventListener('scroll', onSpecScroll, { passive: true })
    rollEl.addEventListener('scroll', onRollScroll, { passive: true })
    return () => {
      specEl.removeEventListener('scroll', onSpecScroll)
      rollEl.removeEventListener('scroll', onRollScroll)
    }
  }, [state])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) void processFile(file)
  }, [processFile])

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) void processFile(file)
  }, [processFile])

  return (
    <div>
      <h1>Xylo Detect</h1>
      <p className="subtitle">Xylophone note detector — C6 to C7</p>

      <div
        className={`upload-zone${dragOver ? ' drag-over' : ''}`}
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <input ref={fileInputRef} type="file" accept="audio/*" onChange={handleFileChange} />
        <div className="icon">🎵</div>
        <strong>Drop an audio file or click to browse</strong>
        <p>Supports WAV, MP3, OGG, FLAC, and more</p>
        {!wasmReady && (
          <p style={{ color: '#ff9f40', marginTop: '0.5rem' }}>Loading analysis engine...</p>
        )}
      </div>

      {state.type === 'analyzing' && (
        <p className="status">Analyzing {state.filename}...</p>
      )}
      {state.type === 'error' && (
        <p className="status error">{state.message}</p>
      )}

      {state.type === 'done' && (
        <>
          <p className="file-info">
            {state.filename} — {state.frames.length} frames, {state.durationSec.toFixed(2)}s
          </p>

          {/* Audio player */}
          <div className="panel">
            <audio
              ref={audioRef}
              src={state.audioSrc}
              controls
              className="audio-player"
            />
          </div>

          {/* Spectrogram with cursor overlay */}
          <div className="panel">
            <h2 className="panel-title">
              Spectrogram — {SPEC_FREQ_MIN}–{SPEC_FREQ_MAX} Hz
            </h2>
            <div className="axis-layout">
              <div className="freq-labels" style={{ height: SPEC_HEIGHT_PX }}>
                {SPEC_FREQ_TICKS.map((hz) => {
                  const yPct =
                    (1 - (hz - SPEC_FREQ_MIN) / (SPEC_FREQ_MAX - SPEC_FREQ_MIN)) * 100
                  return (
                    <div key={hz} className="freq-label" style={{ top: `${yPct}%` }}>
                      {hz}
                    </div>
                  )
                })}
              </div>
              <div className="canvas-scroll-wrapper">
                <div ref={specPanelRef} className="canvas-scroll">
                  <div className="canvas-stack">
                    <canvas ref={spectrogramRef} style={{ height: `${SPEC_HEIGHT_PX}px` }} />
                    <canvas
                      ref={cursorCanvasRef}
                      style={{ height: `${SPEC_HEIGHT_PX}px` }}
                      className="cursor-canvas"
                    />
                  </div>
                </div>
                <div className="note-freq-labels" style={{ height: SPEC_HEIGHT_PX }}>
                {Object.entries(NOTE_FREQS).map(([note, freq]) => {
                  const yPct =
                    (1 - (freq - SPEC_FREQ_MIN) / (SPEC_FREQ_MAX - SPEC_FREQ_MIN)) * 100
                  const noteIdx = NOTES.indexOf(note as typeof NOTES[number]) as NoteIndex
                  return (
                    <div
                      key={note}
                      className="note-freq-label"
                      style={{ top: `${yPct}%`, color: NOTE_COLORS[noteIdx] }}
                    >
                      {note}
                    </div>
                  )
                })}
                </div>
              </div>
            </div>
          </div>

          {/* Piano roll */}
          <div className="panel">
            <h2 className="panel-title">
              Note predictions — each column = 25 ms frame (10 ms stride)
            </h2>
            <div className="axis-layout">
              <div className="note-labels" style={{ height: CANVAS_HEIGHT }}>
                {[...NOTES].reverse().map((n) => (
                  <div key={n} className="note-label">{n}</div>
                ))}
              </div>
              <div ref={rollPanelRef} className="canvas-scroll">
                <canvas ref={pianoRollRef} style={{ height: `${CANVAS_HEIGHT}px` }} />
              </div>
            </div>
            <div className="legend">
              {NOTES.map((n, i) => (
                <div key={n} className="legend-item">
                  <div className="legend-swatch" style={{ background: NOTE_COLORS[i as NoteIndex] }} />
                  {n}
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
