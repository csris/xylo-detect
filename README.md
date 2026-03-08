# xylo-detect

A browser-based xylophone note detector. Upload an audio recording and the app
identifies which of the eight notes **C6–C7** are being played, frame by frame.

## Features

- Drag-and-drop audio upload (WAV, MP3, OGG, FLAC, …)
- Analysis engine written in **Rust**, compiled to **WebAssembly** — runs entirely in the browser, no server required
- **Spectrogram** view (500–2500 Hz, inferno colormap) with note-frequency overlays
- **Piano-roll** predictions view (C7 at top → C6 at bottom)
- Synchronized scrolling between the two views
- Native **audio player** with a playback cursor that tracks across the spectrogram
- Auto-scroll during playback

## Algorithm

See [`docs/algorithm.md`](docs/algorithm.md) for a detailed description of the
signal-processing pipeline.

## Getting started

### Prerequisites

| Tool | Version tested |
|---|---|
| Node.js | 22 |
| Rust + Cargo | 1.93 |
| `wasm-pack` | 0.10 |

### Install dependencies

```bash
npm install
```

### Build the WASM analysis engine

```bash
npm run build:wasm
```

This compiles the Rust crate in `analysis/` and writes the generated JS/WASM
bindings to `src/wasm/`.  Re-run whenever you change Rust source.

### Run the dev server

```bash
npm run dev
```

### Production build

```bash
npm run build
```

## Project structure

```
xylo-detect/
├── analysis/               # Rust WASM crate
│   ├── Cargo.toml
│   └── src/lib.rs          # Analysis engine
├── src/
│   ├── App.tsx             # React application
│   ├── App.css
│   ├── main.tsx
│   └── wasm/               # Generated — do not edit (gitignored)
├── docs/
│   └── algorithm.md        # Signal-processing documentation
├── index.html
├── vite.config.ts
└── package.json
```

## Tunable parameters

All analysis parameters live as named constants at the top of
`analysis/src/lib.rs` and are documented in [`docs/algorithm.md`](docs/algorithm.md).

| Constant | Default | Description |
|---|---|---|
| `ENERGY_THRESHOLD` | `0.01` | Global RMS silence gate (~−40 dB) |
| `NOTE_DOMINANCE_THRESHOLD` | `0.45` | Min fraction of note-band energy the winning filter must hold |
| `NOTE_BAND_MIN_FRACTION` | `0.30` | Min fraction of total FFT power that must fall within the note band |
| `ONSET_FLUX_RATIO` | `3.0` | Energy ratio required to declare an onset |
| `ONSET_CONFIRM_FRAMES` | `2` | Consecutive frames with same winner required to confirm an onset |
| `ONSET_HOLD_FRAMES` | `120` | Frames to hold a prediction after an onset (~1200 ms) |
| `SMOOTH_HALF_WIN` | `2` | Half-window for the temporal mode filter (±2 frames = 50 ms) |
