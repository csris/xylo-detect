# Analysis algorithm

The analysis engine is implemented in Rust (`analysis/src/lib.rs`) and compiled
to WebAssembly.  It exposes two public functions to the browser:

- `analyze_audio(samples, sample_rate) → Uint8Array` — note predictions, one byte per frame
- `compute_spectrogram(samples, sample_rate) → Float32Array` — normalised log-magnitude spectrogram

Both functions operate on the same frame grid: **25 ms frames with a 10 ms stride**
(a sliding window with 60% overlap).

---

## Framing

```
|←——— 25 ms ———→|
|←——— 25 ms ———→|
        |←——— 25 ms ———→|
                |←——— 25 ms ———→|
        ↑10 ms↑
```

At 44 100 Hz this gives frames of 1 103 samples and a stride of 441 samples.
The frame count for a clip of *N* samples is:

```
num_frames = (N − frame_size) / stride + 1
```

---

## Spectrogram (`compute_spectrogram`)

For each frame:

1. **Hann window** — taper the frame to zero at both edges to suppress spectral
   leakage from the sharp frame boundaries.

   ```
   w(i) = 0.5 × (1 − cos(2π i / (N−1)))
   ```

2. **FFT** — compute the complex spectrum via a forward real FFT (rustfft).
   Only the positive-frequency half is kept (bins 0 … N/2).

3. **Log magnitude** — take `ln(|X[k]|)` for each bin in the display frequency
   range (currently 500–2 500 Hz).

4. **Global normalisation** — shift and scale all values to \[0, 1\] across the
   entire clip so that the dynamic range fills the colour map.

The result is a flat `Float32Array` of shape `[num_frames × num_bins]`
(row-major, bin 0 = lowest frequency) rendered in the browser using an inferno
colour map (black → purple → red → orange → yellow).

---

## Note prediction (`analyze_audio`)

Processing happens in three sequential passes.

### Pass 1 — Raw prediction with per-note dominance threshold

For each frame:

#### 1a. Global silence gate

Compute the RMS amplitude of the raw frame:

```
RMS = sqrt( mean(x²) )
```

If `RMS < ENERGY_THRESHOLD` (default 0.01, ≈ −40 dB relative to full scale),
the frame is marked `NO_PREDICTION` immediately and skipped.  This avoids
wasting computation on silent regions.

#### 1b. Hann window → FFT → magnitude spectrum

Same windowing and FFT as the spectrogram path.

#### 1c. Triangular filterbank

Eight overlapping triangular filters are placed in the frequency domain, one
centred on each target note:

| Index | Note | Frequency (Hz) |
|---|---|---|
| 0 | C6 | 1 046.50 |
| 1 | D6 | 1 174.66 |
| 2 | E6 | 1 318.51 |
| 3 | F6 | 1 396.91 |
| 4 | G6 | 1 567.98 |
| 5 | A6 | 1 760.00 |
| 6 | B6 | 1 975.53 |
| 7 | C7 | 2 093.00 |

Filter edges are placed at the midpoint between adjacent centre frequencies,
extrapolated symmetrically at the two ends.  Each filter rises linearly from
its left edge to its centre, then falls linearly to its right edge.

The energy captured by filter *i* is:

```
E_i = Σ_k  filter_i[k] × |X[k]|²
```

(a weighted sum of the power spectral density over the filter's support).

#### 1d. Per-note dominance check

Sum the energies across all eight filters:

```
E_total = Σ_i E_i
```

The winning filter is `argmax_i(E_i)`.  A prediction is made only if the
winner's share of total note-band energy exceeds the dominance threshold:

```
E_winner / E_total  ≥  NOTE_DOMINANCE_THRESHOLD  (default 0.25)
```

This check is **scale-invariant** (independent of recording level).  When
energy is spread roughly equally across filters — e.g. broadband noise,
reverberation, or a chord — no single filter reaches 25% and the frame is
marked `NO_PREDICTION`.  The 1/8 = 12.5% uniform baseline means only
genuinely dominant pitches pass.

---

### Pass 2 — Onset gating

The xylophone is a percussive instrument: notes have a sharp attack and a
gradual exponential decay.  Without gating, the analyser emits continuous
predictions throughout the sustain tail, even when the note has mostly decayed
into noise.

**Onset detection** uses spectral flux on the total note-band energy:

```
onset detected  ⟺  E_total[f] / E_total[f−1]  ≥  ONSET_FLUX_RATIO  (default 3.0)
```

A threefold energy increase in a single 10 ms stride reliably catches the sharp
attack of a struck bar.

After an onset, the raw prediction is forwarded to the output for
`ONSET_HOLD_FRAMES` frames (default 40 ≈ 400 ms), then suppressed until the
next onset.  If a new onset is detected during the hold period the counter
resets, so rapid successive notes are handled correctly.

---

### Pass 3 — Temporal mode filter

Single-frame prediction errors (caused by FFT phase noise, short spectral
transients, or borderline filter energies) are removed by a **mode filter**:
for each output frame, the most common (plurality) note within a
`±SMOOTH_HALF_WIN` frame window (default ±2 frames = ±20 ms, giving a 50 ms
window) is used.  Frames with `NO_PREDICTION` are excluded from the vote.  If
no valid predictions exist in the window the frame retains `NO_PREDICTION`.

A side effect is that note blocks are extended by up to `SMOOTH_HALF_WIN`
frames at each edge, which smoothly rounds the onset and offset boundaries.

---

## Output encoding

`analyze_audio` returns a `Uint8Array` of length `num_frames`.  Each byte is:

- **0–7** — predicted note index (C6 = 0, … , C7 = 7)
- **255** — `NO_PREDICTION` (silent, ambiguous, or below dominance threshold)

The React front-end skips drawing piano-roll blocks for frames with value 255.

---

## Tunable constants

All constants are defined at the top of `analysis/src/lib.rs`.

| Constant | Default | Effect of increasing |
|---|---|---|
| `ENERGY_THRESHOLD` | `0.01` | Raises the silence floor; more frames become NO_PREDICTION |
| `NOTE_DOMINANCE_THRESHOLD` | `0.25` | Requires a clearer spectral peak; rejects more ambiguous frames |
| `ONSET_FLUX_RATIO` | `3.0` | Requires a sharper energy rise to declare an onset; misses softer strikes |
| `ONSET_HOLD_FRAMES` | `40` | Extends prediction blocks; captures more of the note's decay |
| `SMOOTH_HALF_WIN` | `2` | Wider smoothing window; fewer single-frame glitches but blurs boundaries |
