use wasm_bindgen::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};

// Xylophone notes C6 to C7 (8 notes) with frequencies in Hz
const NOTE_FREQS: [f32; 8] = [
    1046.50, 1174.66, 1318.51, 1396.91, 1567.98, 1760.00, 1975.53, 2093.00,
];

/// Human-readable names for the eight notes, indexed 0–7 (C6=0 … C7=7).
pub const NOTE_NAMES: [&str; 8] = ["C6", "D6", "E6", "F6", "G6", "A6", "B6", "C7"];

// Frequency range shown in the spectrogram
const SPEC_FREQ_MIN: f32 = 500.0;
const SPEC_FREQ_MAX: f32 = 2500.0;

// ── Tunable thresholds ────────────────────────────────────────────────────────

/// Global silence gate: frames whose RMS is below this get NO_PREDICTION (~−40 dB).
pub const ENERGY_THRESHOLD: f32 = 0.01;

/// Per-note dominance: the winning filter must hold at least this fraction of the
/// total note-band energy. Scale-invariant; uniform energy → each filter = 1/8 = 0.125.
/// Raised to 0.45 to reject broadband speech (which spreads energy across all filters).
pub const NOTE_DOMINANCE_THRESHOLD: f32 = 0.45;

/// Onset detection: total note-band energy must rise by at least this ratio in one
/// 10 ms stride to count as a new onset.
pub const ONSET_FLUX_RATIO: f32 = 3.0;

/// Hold the predicted note for this many frames after an onset (~1200 ms).
pub const ONSET_HOLD_FRAMES: usize = 120;

/// Temporal smoothing: half-window for the mode filter (full window = 2×N+1 frames = 50 ms).
pub const SMOOTH_HALF_WIN: usize = 2;

/// Sentinel returned for frames with no confident prediction (not a valid note index 0–7).
pub const NO_PREDICTION: u8 = 255;

// ── Per-frame statistics ──────────────────────────────────────────────────────

/// Full per-frame analysis data returned by [`analyze_full`].
pub struct FrameStats {
    /// Root-mean-square amplitude of the raw (unwindowed) frame.
    pub rms: f32,
    /// Note with the highest filterbank energy (0–7), or NO_PREDICTION if silent.
    /// This is the argmax *before* the dominance threshold is applied.
    pub winner: u8,
    /// Pass 1 output: winner after the dominance gate, or NO_PREDICTION.
    pub pass1: u8,
    /// Pass 2 output: pass1 after onset gating, or NO_PREDICTION.
    pub pass2: u8,
    /// Pass 3 output: pass2 after the temporal mode filter — matches the app.
    pub pass3: u8,
    /// Winner's share of total note-band energy (0.0 for silent frames).
    pub dominance: f32,
    /// Sum of all eight filterbank energies.
    pub note_energy: f32,
    /// Ratio of this frame's note_energy to the previous frame's (0.0 for frame 0).
    pub flux: f32,
    /// True if a new onset was detected at this frame.
    pub onset: bool,
}

// ── Internal helpers ──────────────────────────────────────────────────────────

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

// ── Public analysis API ───────────────────────────────────────────────────────

/// Run the full three-pass analysis pipeline and return per-frame statistics.
///
/// This is the primary analysis function.  The WASM-exported [`analyze_audio`]
/// is a thin wrapper that extracts only the `pass3` field from each element.
///
/// Pipeline:
///   Pass 1 – per-frame: Hann window → FFT → triangular filterbank →
///             per-note dominance threshold → raw prediction + note-band energy
///   Pass 2 – onset gating: detect sharp energy rises; only emit predictions
///             within ONSET_HOLD_FRAMES of an onset
///   Pass 3 – temporal smoothing: mode filter over a ±SMOOTH_HALF_WIN window
pub fn analyze_full(samples: &[f32], sample_rate: f32) -> Vec<FrameStats> {
    let frame_size = (0.025 * sample_rate).round() as usize;
    let stride     = (0.010 * sample_rate).round() as usize;
    let hann       = hann_window(frame_size);
    let mut planner = FftPlanner::<f32>::new();
    let fft         = planner.plan_fft_forward(frame_size);
    let filters     = build_filterbank(frame_size, sample_rate);
    let num_frames  = frame_count(samples.len(), frame_size, stride);

    // ── Pass 1 ───────────────────────────────────────────────────────────────
    let mut winner_vec  = Vec::with_capacity(num_frames);
    let mut pass1       = Vec::with_capacity(num_frames);
    let mut note_energy = Vec::with_capacity(num_frames);
    let mut rms_vec     = Vec::with_capacity(num_frames);
    let mut dom_vec     = Vec::with_capacity(num_frames);

    for f in 0..num_frames {
        let start = f * stride;
        let frame = &samples[start..start + frame_size];

        let rms = (frame.iter().map(|&s| s * s).sum::<f32>() / frame_size as f32).sqrt();
        rms_vec.push(rms);

        if rms < ENERGY_THRESHOLD {
            winner_vec.push(NO_PREDICTION);
            pass1.push(NO_PREDICTION);
            note_energy.push(0.0f32);
            dom_vec.push(0.0f32);
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

        let (best_idx, &best_e) = energies
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        winner_vec.push(best_idx as u8);

        let dom = if total > 1e-10 { best_e / total } else { 0.0 };
        dom_vec.push(dom);

        if total < 1e-10 || dom < NOTE_DOMINANCE_THRESHOLD {
            pass1.push(NO_PREDICTION);
        } else {
            pass1.push(best_idx as u8);
        }
    }

    // ── Pass 2: onset gating ──────────────────────────────────────────────────
    let mut pass2      = vec![NO_PREDICTION; num_frames];
    let mut hold: usize = 0;
    let mut prev_e     = 0.0f32;
    let mut onset_vec  = vec![false; num_frames];
    let mut flux_vec   = vec![0.0f32; num_frames];

    for f in 0..num_frames {
        let curr_e = note_energy[f];
        let flux = curr_e / prev_e.max(1e-10);
        flux_vec[f] = flux;

        if curr_e > prev_e.max(1e-10) * ONSET_FLUX_RATIO {
            hold = ONSET_HOLD_FRAMES;
            onset_vec[f] = true;
        }
        if hold > 0 {
            pass2[f] = pass1[f];
            hold -= 1;
        }
        prev_e = curr_e;
    }

    // ── Pass 3: temporal mode filter ──────────────────────────────────────────
    let mut pass3 = vec![NO_PREDICTION; num_frames];
    for f in 0..num_frames {
        let lo = f.saturating_sub(SMOOTH_HALF_WIN);
        let hi = (f + SMOOTH_HALF_WIN + 1).min(num_frames);
        let mut counts = [0u32; 8];
        for &p in &pass2[lo..hi] {
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
            pass3[f] = note as u8;
        }
    }

    // ── Assemble ──────────────────────────────────────────────────────────────
    (0..num_frames)
        .map(|f| FrameStats {
            rms:         rms_vec[f],
            winner:      winner_vec[f],
            pass1:       pass1[f],
            pass2:       pass2[f],
            pass3:       pass3[f],
            dominance:   dom_vec[f],
            note_energy: note_energy[f],
            flux:        flux_vec[f],
            onset:       onset_vec[f],
        })
        .collect()
}

/// Analyze audio samples and return a predicted note index (0–7) per frame,
/// or NO_PREDICTION (255) for silent / ambiguous frames.
///
/// This is the WASM-exported entry point used by the browser app.
/// It is a thin wrapper around [`analyze_full`] that returns only the
/// final (Pass 3) prediction for each frame.
#[wasm_bindgen]
pub fn analyze_audio(samples: &[f32], sample_rate: f32) -> Vec<u8> {
    analyze_full(samples, sample_rate)
        .into_iter()
        .map(|s| s.pass3)
        .collect()
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

// ── Filterbank ────────────────────────────────────────────────────────────────

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
