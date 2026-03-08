#!/usr/bin/env python3
"""
tune.py — print per-frame analysis stats to help calibrate threshold constants.

Usage:
    uv run tune.py <audio.m4a|wav|...>

Requires ffmpeg on PATH to decode the audio file to raw PCM.
"""

import subprocess
import sys
import struct
import numpy as np

# ── Analysis constants (mirror analysis/src/lib.rs) ──────────────────────────
SR_TARGET    = 48_000
FRAME_MS     = 25
STRIDE_MS    = 10
NOTE_FREQS   = [1046.50, 1174.66, 1318.51, 1396.91, 1567.98, 1760.00, 1975.53, 2093.00]
NOTE_NAMES   = ['C6', 'D6', 'E6', 'F6', 'G6', 'A6', 'B6', 'C7']

# Current threshold values
ENERGY_THRESHOLD       = 0.01    # global RMS silence gate
NOTE_DOMINANCE_THRESH  = 0.25    # winning filter fraction of total note-band energy
ONSET_FLUX_RATIO       = 3.0     # energy ratio to declare onset
ONSET_HOLD_FRAMES      = 120     # frames to hold after onset
SMOOTH_HALF_WIN        = 2       # mode filter half-window


def decode_to_mono_f32(path: str, sr: int) -> np.ndarray:
    """Use ffmpeg to decode any audio file to mono f32le PCM."""
    cmd = [
        "ffmpeg", "-i", path,
        "-ac", "1", "-ar", str(sr),
        "-f", "f32le", "-",
        "-loglevel", "error",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(result.stderr.decode(), file=sys.stderr)
        sys.exit(1)
    return np.frombuffer(result.stdout, dtype=np.float32)


def build_filterbank(frame_size: int, sr: int) -> np.ndarray:
    bin_hz  = sr / frame_size
    centers = np.array(NOTE_FREQS)
    edges   = np.concatenate([
        [centers[0] - (centers[1] - centers[0]) / 2],
        (centers[:-1] + centers[1:]) / 2,
        [centers[-1] + (centers[-1] - centers[-2]) / 2],
    ])
    half  = frame_size // 2 + 1
    freqs = np.arange(half) * bin_hz
    bank  = np.zeros((8, half))
    for n in range(8):
        fl, fc, fr = edges[n], centers[n], edges[n + 1]
        rise = (freqs >= fl) & (freqs <= fc)
        fall = (freqs >  fc) & (freqs <= fr)
        bank[n, rise] = (freqs[rise] - fl) / (fc - fl)
        bank[n, fall] = (fr - freqs[fall]) / (fr - fc)
    return bank


def analyze(data: np.ndarray, sr: int) -> None:
    frame_size = round(FRAME_MS  / 1000 * sr)
    stride     = round(STRIDE_MS / 1000 * sr)
    hann       = 0.5 * (1 - np.cos(2 * np.pi * np.arange(frame_size) / (frame_size - 1)))
    bank       = build_filterbank(frame_size, sr)
    num_frames = (len(data) - frame_size) // stride + 1

    print(f"File: {len(data)} samples  {len(data)/sr:.3f}s  {sr} Hz")
    print(f"Frames: {num_frames}  ({FRAME_MS}ms / {STRIDE_MS}ms stride)\n")
    print(f"Current thresholds: RMS≥{ENERGY_THRESHOLD}  dominance≥{NOTE_DOMINANCE_THRESH}  "
          f"flux≥{ONSET_FLUX_RATIO}  hold={ONSET_HOLD_FRAMES}  smooth±{SMOOTH_HALF_WIN}\n")

    header = f"{'Fr':>4}  {'t(s)':>5}  {'RMS':>8}  {'Winner':>6}  "  \
             f"{'Dom':>6}  {'TotalE':>10}  {'Flux':>6}  Flags"
    print(header)
    print("─" * len(header))

    prev_total = 1e-10
    hold = 0
    raw_preds = []

    for f in range(num_frames):
        s      = f * stride
        frame  = data[s : s + frame_size]
        rms    = float(np.sqrt(np.mean(frame ** 2)))
        t      = s / sr

        flags = []

        if rms < ENERGY_THRESHOLD:
            raw_preds.append(None)
            print(f"{f:>4}  {t:>5.3f}  {rms:>8.5f}  {'—':>6}  {'—':>6}  {'—':>10}  {'—':>6}  [silent]")
            prev_total = 1e-10
            if hold > 0: hold -= 1
            continue

        windowed = frame * hann
        mag      = np.abs(np.fft.rfft(windowed))
        energies = bank @ (mag ** 2)
        total    = float(energies.sum())
        flux     = total / max(prev_total, 1e-10)
        winner   = int(np.argmax(energies))
        dom      = energies[winner] / total if total > 1e-10 else 0.0

        if flux >= ONSET_FLUX_RATIO:
            hold = ONSET_HOLD_FRAMES
            flags.append("ONSET")

        if dom < NOTE_DOMINANCE_THRESH:
            flags.append("low-dom")
            raw_preds.append(None)
        else:
            raw_preds.append(winner)

        if hold > 0:
            flags.append(f"hold({hold})")
            hold -= 1

        print(f"{f:>4}  {t:>5.3f}  {rms:>8.5f}  {NOTE_NAMES[winner]:>6}  "
              f"{dom:>6.3f}  {total:>10.1f}  {flux:>6.2f}  {' '.join(flags)}")

        prev_total = total

    # Summary statistics
    valid = [p for p in raw_preds if p is not None]
    from collections import Counter
    counts = Counter(valid)
    print(f"\n── Summary ────────────────────────────────")
    print(f"Frames with prediction : {len(valid)} / {num_frames}")
    print(f"Frames silent (RMS)    : {num_frames - len(valid)}")
    print("Note distribution:")
    for i, name in enumerate(NOTE_NAMES):
        bar = "█" * counts.get(i, 0)
        print(f"  {name}: {counts.get(i,0):>4}  {bar}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: uv run {sys.argv[0]} <audio-file>")
        sys.exit(1)
    data = decode_to_mono_f32(sys.argv[1], SR_TARGET)
    analyze(data, SR_TARGET)
