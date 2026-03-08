use wasm_bindgen::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};

// Xylophone notes C6 to C7 (8 notes) with frequencies in Hz
const NOTE_FREQS: [f32; 8] = [
    1046.50, 1174.66, 1318.51, 1396.91, 1567.98, 1760.00, 1975.53, 2093.00,
];

// Frequency range shown in the spectrogram
const SPEC_FREQ_MIN: f32 = 500.0;
const SPEC_FREQ_MAX: f32 = 2500.0;

// Global silence gate: frames whose RMS is below this get NO_PREDICTION immediately (~-40 dB)
const ENERGY_THRESHOLD: f32 = 0.01;

// Per-note dominance: the winning filter must hold at least this fraction of the
// total note-band energy. Scale-invariant; uniform energy → each filter = 1/8 = 0.125.
const NOTE_DOMINANCE_THRESHOLD: f32 = 0.25;

// Onset detection: total note-band energy must rise by at least this ratio in one
// 10 ms stride to count as a new onset.
const ONSET_FLUX_RATIO: f32 = 3.0;

// Hold the predicted note for this many frames after an onset (~300 ms).
const ONSET_HOLD_FRAMES: usize = 30;

// Temporal smoothing: half-window for the mode filter (full window = 2×N+1 frames = 50 ms).
const SMOOTH_HALF_WIN: usize = 2;

// Sentinel returned for frames with no confident prediction (not a valid note index 0–7)
const NO_PREDICTION: u8 = 255;

fn hann_window(frame_size: usize) -> Vec<f32> {
    (0..frame_size)
        .map(|i| {
            0.5 * (1.0
                - (2.0 * std::f32::consts::PI * i as f32 / (frame_size - 1) as f32).cos())
        })
        .collect()
}

fn frame_count(num_samples: usize, frame_size: usize, stride: usize) -> usize {
    if num_samples < frame_size {
        0
    } else {
        (num_samples - frame_size) / stride + 1
    }
}

/// Analyze audio samples and return a predicted note index (0–7) per frame,
/// or NO_PREDICTION (255) for silent / ambiguous frames.
///
/// Pipeline:
///   Pass 1 – per-frame: Hann window → FFT → triangular filterbank →
///             per-note dominance threshold → raw prediction + note-band energy
///   Pass 2 – onset gating: detect sharp energy rises; only emit predictions
///             within ONSET_HOLD_FRAMES of an onset
///   Pass 3 – temporal smoothing: mode filter over a ±SMOOTH_HALF_WIN window
#[wasm_bindgen]
pub fn analyze_audio(samples: &[f32], sample_rate: f32) -> Vec<u8> {
    let frame_size = (0.025 * sample_rate).round() as usize;
    let stride    = (0.010 * sample_rate).round() as usize;
    let hann      = hann_window(frame_size);
    let mut planner = FftPlanner::<f32>::new();
    let fft         = planner.plan_fft_forward(frame_size);
    let filters     = build_filterbank(frame_size, sample_rate);
    let num_frames  = frame_count(samples.len(), frame_size, stride);

    // ── Pass 1: raw prediction + total note-band energy per frame ────────────
    let mut raw         = Vec::with_capacity(num_frames);
    let mut note_energy = Vec::with_capacity(num_frames);

    for f in 0..num_frames {
        let start = f * stride;
        let frame = &samples[start..start + frame_size];

        // Global silence gate
        let rms = (frame.iter().map(|&s| s * s).sum::<f32>() / frame_size as f32).sqrt();
        if rms < ENERGY_THRESHOLD {
            raw.push(NO_PREDICTION);
            note_energy.push(0.0f32);
            continue;
        }

        let mut buf: Vec<Complex<f32>> = frame
            .iter()
            .zip(hann.iter())
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();
        fft.process(&mut buf);

        let half = frame_size / 2 + 1;
        let mag: Vec<f32> = buf[..half].iter().map(|c| c.norm()).collect();

        let mut energies = [0.0f32; 8];
        for (i, filter) in filters.iter().enumerate() {
            energies[i] = filter.iter().zip(mag.iter()).map(|(&w, &m)| w * m * m).sum();
        }

        let total: f32 = energies.iter().sum();
        note_energy.push(total);

        // Per-note dominance check (scale-invariant)
        let (best_idx, &best_e) = energies
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        if total < 1e-10 || best_e / total < NOTE_DOMINANCE_THRESHOLD {
            raw.push(NO_PREDICTION);
        } else {
            raw.push(best_idx as u8);
        }
    }

    // ── Pass 2: onset gating ─────────────────────────────────────────────────
    // A note is reported only for ONSET_HOLD_FRAMES frames after an onset.
    let mut gated      = vec![NO_PREDICTION; num_frames];
    let mut hold: usize = 0;
    let mut prev_e     = 0.0f32;

    for f in 0..num_frames {
        let curr_e = note_energy[f];
        if curr_e > prev_e.max(1e-10) * ONSET_FLUX_RATIO {
            hold = ONSET_HOLD_FRAMES; // new onset: reset hold counter
        }
        if hold > 0 {
            gated[f] = raw[f];
            hold -= 1;
        }
        prev_e = curr_e;
    }

    // ── Pass 3: temporal smoothing (mode filter) ──────────────────────────────
    // For each frame, take the plurality note in a ±SMOOTH_HALF_WIN window.
    let mut result = vec![NO_PREDICTION; num_frames];
    for f in 0..num_frames {
        let lo = f.saturating_sub(SMOOTH_HALF_WIN);
        let hi = (f + SMOOTH_HALF_WIN + 1).min(num_frames);
        let mut counts = [0u32; 8];
        for &p in &gated[lo..hi] {
            if p != NO_PREDICTION {
                counts[p as usize] += 1;
            }
        }
        if let Some((note, _)) = counts
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .max_by_key(|(_, &c)| c)
        {
            result[f] = note as u8;
        }
    }

    result
}

/// Returns how many frequency bins `compute_spectrogram` produces per frame
/// (depends on sample rate).
#[wasm_bindgen]
pub fn spectrogram_num_bins(sample_rate: f32) -> usize {
    let frame_size = (0.025 * sample_rate).round() as usize;
    let bin_hz = sample_rate / frame_size as f32;
    let bin_min = (SPEC_FREQ_MIN / bin_hz).floor() as usize;
    let bin_max = ((SPEC_FREQ_MAX / bin_hz).ceil() as usize).min(frame_size / 2);
    bin_max.saturating_sub(bin_min) + 1
}

/// Compute a log-magnitude spectrogram over the same 25 ms / 10 ms frames.
///
/// Returns a flat `Vec<f32>` of shape `[num_frames × num_bins]` (row-major, bin 0
/// = lowest frequency) with values normalized to [0, 1] globally.
/// Frequency range covered: SPEC_FREQ_MIN–SPEC_FREQ_MAX Hz.
#[wasm_bindgen]
pub fn compute_spectrogram(samples: &[f32], sample_rate: f32) -> Vec<f32> {
    let frame_size = (0.025 * sample_rate).round() as usize;
    let stride = (0.010 * sample_rate).round() as usize;
    let bin_hz = sample_rate / frame_size as f32;
    let bin_min = (SPEC_FREQ_MIN / bin_hz).floor() as usize;
    let bin_max = ((SPEC_FREQ_MAX / bin_hz).ceil() as usize).min(frame_size / 2);
    let num_display_bins = bin_max.saturating_sub(bin_min) + 1;

    let hann = hann_window(frame_size);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(frame_size);
    let num_frames = frame_count(samples.len(), frame_size, stride);

    let mut raw = vec![0.0f32; num_frames * num_display_bins];
    for f in 0..num_frames {
        let start = f * stride;
        let frame = &samples[start..start + frame_size];
        let mut buf: Vec<Complex<f32>> = frame
            .iter()
            .zip(hann.iter())
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();
        fft.process(&mut buf);
        for b in 0..num_display_bins {
            let mag = buf[bin_min + b].norm();
            raw[f * num_display_bins + b] = mag.max(1e-6).ln();
        }
    }

    // Normalize globally to [0, 1]
    let min_val = raw.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max_val - min_val).max(1e-6);
    for v in raw.iter_mut() {
        *v = (*v - min_val) / range;
    }
    raw
}

/// Build a triangular filterbank: one filter per note, length = frame_size/2 + 1.
fn build_filterbank(frame_size: usize, sample_rate: f32) -> Vec<Vec<f32>> {
    let half = frame_size / 2 + 1;
    let bin_hz = sample_rate / frame_size as f32;
    let centers = NOTE_FREQS;

    // Edges: midpoints between adjacent centers, extrapolated at the boundaries
    let mut edges = Vec::with_capacity(centers.len() + 1);
    let left_edge = centers[0] - (centers[1] - centers[0]) / 2.0;
    edges.push(left_edge.max(0.0));
    for i in 0..centers.len() - 1 {
        edges.push((centers[i] + centers[i + 1]) / 2.0);
    }
    let right_edge = centers[centers.len() - 1]
        + (centers[centers.len() - 1] - centers[centers.len() - 2]) / 2.0;
    edges.push(right_edge);

    let mut filters = Vec::with_capacity(centers.len());
    for n in 0..centers.len() {
        let f_left = edges[n];
        let f_center = centers[n];
        let f_right = edges[n + 1];
        let mut filter = vec![0.0f32; half];
        for bin in 0..half {
            let freq = bin as f32 * bin_hz;
            filter[bin] = if freq >= f_left && freq <= f_center {
                (freq - f_left) / (f_center - f_left)
            } else if freq > f_center && freq <= f_right {
                (f_right - freq) / (f_right - f_center)
            } else {
                0.0
            };
        }
        filters.push(filter);
    }
    filters
}
