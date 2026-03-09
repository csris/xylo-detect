use rustfft::{num_complex::Complex, FftPlanner};
use wasm_bindgen::prelude::*;

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
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (frame_size - 1) as f32).cos())
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
    let stride = (0.010 * sample_rate).round() as usize;
    let hann = hann_window(frame_size);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(frame_size);
    let filters = build_filterbank(frame_size, sample_rate);
    let num_frames = frame_count(samples.len(), frame_size, stride);

    // ── Pass 1 ───────────────────────────────────────────────────────────────
    let mut winner_vec = Vec::with_capacity(num_frames);
    let mut pass1 = Vec::with_capacity(num_frames);
    let mut note_energy = Vec::with_capacity(num_frames);
    let mut rms_vec = Vec::with_capacity(num_frames);
    let mut dom_vec = Vec::with_capacity(num_frames);

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
            energies[i] = filter
                .iter()
                .zip(mag.iter())
                .map(|(&w, &m)| w * m * m)
                .sum();
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
    let mut pass2 = vec![NO_PREDICTION; num_frames];
    let mut hold: usize = 0;
    let mut prev_e = 0.0f32;
    let mut onset_vec = vec![false; num_frames];
    let mut flux_vec = vec![0.0f32; num_frames];

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
    for (f, out) in pass3.iter_mut().enumerate() {
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
            *out = note as u8;
        }
    }

    // ── Assemble ──────────────────────────────────────────────────────────────
    (0..num_frames)
        .map(|f| FrameStats {
            rms: rms_vec[f],
            winner: winner_vec[f],
            pass1: pass1[f],
            pass2: pass2[f],
            pass3: pass3[f],
            dominance: dom_vec[f],
            note_energy: note_energy[f],
            flux: flux_vec[f],
            onset: onset_vec[f],
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

/// Encode mono f32 PCM samples as a 16-bit mono WAV file.
///
/// Returns the raw WAV bytes. On the JS side, wrap with
/// `new Blob([result], { type: 'audio/wav' })` to get a playable blob.
/// The encoded samples use the same clamping and scaling as the TypeScript
/// version: positive samples scale to 0x7FFF, negative to 0x8000.
#[wasm_bindgen]
pub fn encode_pcm_to_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let num_samples = samples.len();
    let data_size = (num_samples * 2) as u32;
    let mut buf = Vec::with_capacity(44 + num_samples * 2);

    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_size).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // subchunk1 size (PCM)
    buf.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // num channels = mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    buf.extend_from_slice(&2u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());

    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let pcm: i16 = if clamped < 0.0 {
            (clamped * 0x8000u16 as f32) as i16
        } else {
            (clamped * 0x7FFFu16 as f32) as i16
        };
        buf.extend_from_slice(&pcm.to_le_bytes());
    }

    buf
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
        for (bin, val) in filter.iter_mut().enumerate() {
            let freq = bin as f32 * bin_hz;
            *val = if freq >= f_left && freq <= f_center {
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    fn make_sine(freq: f32, duration_secs: f32, amplitude: f32) -> Vec<f32> {
        let n = (duration_secs * SR) as usize;
        (0..n)
            .map(|i| amplitude * (2.0 * std::f32::consts::PI * freq * i as f32 / SR).sin())
            .collect()
    }

    /// Returns the most common non-NO_PREDICTION note across pass3 of all frames.
    fn dominant_pass3(frames: &[FrameStats]) -> Option<u8> {
        let mut counts = [0usize; 8];
        for f in frames {
            if f.pass3 != NO_PREDICTION {
                counts[f.pass3 as usize] += 1;
            }
        }
        counts
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i as u8)
    }

    // ── hann_window ───────────────────────────────────────────────────────────

    #[test]
    fn hann_endpoints_are_zero() {
        let w = hann_window(1200);
        assert!(w[0] < 1e-6, "w[0] = {}", w[0]);
        assert!(w[1199] < 1e-6, "w[1199] = {}", w[1199]);
    }

    #[test]
    fn hann_peak_near_center() {
        let w = hann_window(1024);
        let peak = w
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert!(
            (peak as isize - 511).abs() <= 1,
            "peak at {peak}, expected ~511"
        );
    }

    #[test]
    fn hann_values_in_unit_interval() {
        assert!(hann_window(1200).iter().all(|&v| v >= 0.0 && v <= 1.0));
    }

    #[test]
    fn hann_is_symmetric() {
        let w = hann_window(1200);
        for i in 0..600 {
            assert!((w[i] - w[1199 - i]).abs() < 1e-6, "asymmetry at {i}");
        }
    }

    #[test]
    fn hann_monotone_in_first_half() {
        let w = hann_window(1200);
        for i in 0..599 {
            assert!(
                w[i] <= w[i + 1],
                "not monotone at {i}: {} > {}",
                w[i],
                w[i + 1]
            );
        }
    }

    #[test]
    fn hann_no_nan_or_inf() {
        assert!(hann_window(4096).iter().all(|v| v.is_finite()));
    }

    // ── frame_count ───────────────────────────────────────────────────────────

    #[test]
    fn frame_count_empty_input() {
        assert_eq!(frame_count(0, 1200, 480), 0);
    }

    #[test]
    fn frame_count_shorter_than_frame() {
        assert_eq!(frame_count(1199, 1200, 480), 0);
    }

    #[test]
    fn frame_count_exactly_one_frame() {
        assert_eq!(frame_count(1200, 1200, 480), 1);
    }

    #[test]
    fn frame_count_two_frames() {
        assert_eq!(frame_count(1680, 1200, 480), 2); // 1200 + 480
    }

    #[test]
    fn frame_count_one_second_at_48k() {
        let fs = (0.025 * SR).round() as usize; // 1200
        let st = (0.010 * SR).round() as usize; // 480
        let n = SR as usize; // 48000
        let expected = (n - fs) / st + 1; // 98
        assert_eq!(frame_count(n, fs, st), expected);
    }

    #[test]
    fn frame_count_non_overlapping() {
        // stride = frame_size → non-overlapping windows
        assert_eq!(frame_count(4800, 1200, 1200), 4);
    }

    // ── build_filterbank ──────────────────────────────────────────────────────

    #[test]
    fn filterbank_has_eight_filters() {
        let fs = (0.025 * SR).round() as usize;
        assert_eq!(build_filterbank(fs, SR).len(), 8);
    }

    #[test]
    fn filterbank_filter_lengths() {
        let fs = (0.025 * SR).round() as usize;
        let expected = fs / 2 + 1;
        for (i, f) in build_filterbank(fs, SR).iter().enumerate() {
            assert_eq!(f.len(), expected, "filter {i} has wrong length");
        }
    }

    #[test]
    fn filterbank_values_in_unit_interval() {
        let fs = (0.025 * SR).round() as usize;
        for (i, f) in build_filterbank(fs, SR).iter().enumerate() {
            for (j, &v) in f.iter().enumerate() {
                assert!(v >= 0.0 && v <= 1.0, "filter {i} bin {j} = {v}");
            }
        }
    }

    #[test]
    fn filterbank_each_filter_has_nonzero_weights() {
        let fs = (0.025 * SR).round() as usize;
        for (i, f) in build_filterbank(fs, SR).iter().enumerate() {
            assert!(f.iter().any(|&v| v > 0.0), "filter {i} is all zeros");
        }
    }

    #[test]
    fn filterbank_peaks_near_note_frequencies() {
        let fs = (0.025 * SR).round() as usize;
        let bin_hz = SR / fs as f32;
        for (n, f) in build_filterbank(fs, SR).iter().enumerate() {
            let peak_bin = f
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            let peak_freq = peak_bin as f32 * bin_hz;
            let expected = NOTE_FREQS[n];
            assert!(
                (peak_freq - expected).abs() <= bin_hz,
                "filter {n} peak at {peak_freq} Hz, expected near {expected} Hz"
            );
        }
    }

    #[test]
    fn filterbank_no_nan_or_inf() {
        let fs = (0.025 * SR).round() as usize;
        for f in build_filterbank(fs, SR).iter() {
            assert!(f.iter().all(|v| v.is_finite()));
        }
    }

    // ── analyze_full — silence / edge cases ───────────────────────────────────

    #[test]
    fn silent_input_all_no_prediction() {
        let silence = vec![0.0f32; 48000];
        let frames = analyze_full(&silence, SR);
        assert!(!frames.is_empty());
        assert!(frames.iter().all(|f| f.pass1 == NO_PREDICTION));
        assert!(frames.iter().all(|f| f.pass2 == NO_PREDICTION));
        assert!(frames.iter().all(|f| f.pass3 == NO_PREDICTION));
    }

    #[test]
    fn empty_input_zero_frames() {
        assert_eq!(analyze_full(&[], SR).len(), 0);
    }

    #[test]
    fn too_short_input_zero_frames() {
        assert_eq!(analyze_full(&[0.0f32; 100], SR).len(), 0);
    }

    #[test]
    fn frame_count_matches_formula() {
        let signal = make_sine(NOTE_FREQS[0], 2.0, 0.5);
        let fs = (0.025 * SR).round() as usize;
        let st = (0.010 * SR).round() as usize;
        assert_eq!(
            analyze_full(&signal, SR).len(),
            frame_count(signal.len(), fs, st)
        );
    }

    #[test]
    fn sub_threshold_amplitude_is_silent() {
        // RMS of sine amplitude A = A/√2 ≈ 0.00354 < ENERGY_THRESHOLD 0.01
        let signal = make_sine(NOTE_FREQS[0], 1.0, 0.005);
        let frames = analyze_full(&signal, SR);
        assert!(frames.iter().all(|f| f.rms < ENERGY_THRESHOLD));
        assert!(frames.iter().all(|f| f.pass1 == NO_PREDICTION));
    }

    #[test]
    fn rms_is_nonnegative() {
        let signal = make_sine(NOTE_FREQS[0], 1.0, 0.5);
        assert!(analyze_full(&signal, SR).iter().all(|f| f.rms >= 0.0));
    }

    #[test]
    fn dominance_in_unit_interval() {
        let signal = make_sine(NOTE_FREQS[0], 1.0, 0.5);
        assert!(analyze_full(&signal, SR)
            .iter()
            .all(|f| f.dominance >= 0.0 && f.dominance <= 1.0));
    }

    #[test]
    fn note_energy_is_nonnegative() {
        let signal = make_sine(NOTE_FREQS[0], 1.0, 0.5);
        assert!(analyze_full(&signal, SR)
            .iter()
            .all(|f| f.note_energy >= 0.0));
    }

    #[test]
    fn analyze_full_is_deterministic() {
        let signal = make_sine(NOTE_FREQS[0], 1.0, 0.5);
        let p3_a: Vec<u8> = analyze_full(&signal, SR).iter().map(|f| f.pass3).collect();
        let p3_b: Vec<u8> = analyze_full(&signal, SR).iter().map(|f| f.pass3).collect();
        assert_eq!(p3_a, p3_b);
    }

    // ── Note detection — one test per note ────────────────────────────────────

    #[test]
    fn detects_c6() {
        assert_eq!(
            dominant_pass3(&analyze_full(&make_sine(NOTE_FREQS[0], 2.0, 0.5), SR)),
            Some(0)
        );
    }

    #[test]
    fn detects_d6() {
        assert_eq!(
            dominant_pass3(&analyze_full(&make_sine(NOTE_FREQS[1], 2.0, 0.5), SR)),
            Some(1)
        );
    }

    #[test]
    fn detects_e6() {
        assert_eq!(
            dominant_pass3(&analyze_full(&make_sine(NOTE_FREQS[2], 2.0, 0.5), SR)),
            Some(2)
        );
    }

    #[test]
    fn detects_f6() {
        assert_eq!(
            dominant_pass3(&analyze_full(&make_sine(NOTE_FREQS[3], 2.0, 0.5), SR)),
            Some(3)
        );
    }

    #[test]
    fn detects_g6() {
        assert_eq!(
            dominant_pass3(&analyze_full(&make_sine(NOTE_FREQS[4], 2.0, 0.5), SR)),
            Some(4)
        );
    }

    #[test]
    fn detects_a6() {
        assert_eq!(
            dominant_pass3(&analyze_full(&make_sine(NOTE_FREQS[5], 2.0, 0.5), SR)),
            Some(5)
        );
    }

    #[test]
    fn detects_b6() {
        assert_eq!(
            dominant_pass3(&analyze_full(&make_sine(NOTE_FREQS[6], 2.0, 0.5), SR)),
            Some(6)
        );
    }

    #[test]
    fn detects_c7() {
        assert_eq!(
            dominant_pass3(&analyze_full(&make_sine(NOTE_FREQS[7], 2.0, 0.5), SR)),
            Some(7)
        );
    }

    // ── analyze_audio wrapper ─────────────────────────────────────────────────

    #[test]
    fn analyze_audio_equals_pass3() {
        let signal = make_sine(NOTE_FREQS[0], 1.0, 0.5);
        let expected: Vec<u8> = analyze_full(&signal, SR).iter().map(|f| f.pass3).collect();
        assert_eq!(analyze_audio(&signal, SR), expected);
    }

    #[test]
    fn analyze_audio_length_matches_formula() {
        let signal = make_sine(NOTE_FREQS[0], 1.0, 0.5);
        let fs = (0.025 * SR).round() as usize;
        let st = (0.010 * SR).round() as usize;
        assert_eq!(
            analyze_audio(&signal, SR).len(),
            frame_count(signal.len(), fs, st)
        );
    }

    #[test]
    fn analyze_audio_values_are_valid() {
        let signal = make_sine(NOTE_FREQS[0], 1.0, 0.5);
        assert!(analyze_audio(&signal, SR)
            .iter()
            .all(|&b| b <= 7 || b == NO_PREDICTION));
    }

    // ── Onset detection ───────────────────────────────────────────────────────

    #[test]
    fn onset_fires_at_frame_zero() {
        let signal = make_sine(NOTE_FREQS[0], 1.0, 0.5);
        let frames = analyze_full(&signal, SR);
        assert!(
            frames[0].onset,
            "onset should fire at frame 0 when signal starts loud"
        );
    }

    #[test]
    fn onset_fires_after_silence() {
        // 20 frames of silence, then tone
        let st = (0.010 * SR).round() as usize;
        let fs = (0.025 * SR).round() as usize;
        let silence_samples = 20 * st + (fs - st);
        let mut signal = vec![0.0f32; silence_samples];
        signal.extend(make_sine(NOTE_FREQS[0], 1.5, 0.5));
        let frames = analyze_full(&signal, SR);
        let first_onset = frames.iter().position(|f| f.onset).expect("no onset found");
        // Onset should be near frame 20 (within ±2 due to frame alignment)
        assert!(
            (first_onset as isize - 20).abs() <= 2,
            "onset at frame {first_onset}, expected ~20"
        );
    }

    #[test]
    fn hold_window_passes_predictions() {
        let signal = make_sine(NOTE_FREQS[0], 2.0, 0.5);
        let frames = analyze_full(&signal, SR);
        for i in 0..ONSET_HOLD_FRAMES.min(frames.len()) {
            assert_ne!(
                frames[i].pass2, NO_PREDICTION,
                "frame {i} inside hold window has no prediction"
            );
        }
    }

    #[test]
    fn steady_tone_no_prediction_after_hold_expires() {
        let signal = make_sine(NOTE_FREQS[0], 2.0, 0.5);
        let frames = analyze_full(&signal, SR);
        if frames.len() > ONSET_HOLD_FRAMES + 5 {
            // A steady sine has flux ≈ 1.0 after frame 0, so no second onset fires.
            // Frames beyond the hold window should get NO_PREDICTION from pass2.
            let check = ONSET_HOLD_FRAMES + 3;
            assert_eq!(
                frames[check].pass2, NO_PREDICTION,
                "frame {check} after hold window should be NO_PREDICTION"
            );
        }
    }

    // ── encode_pcm_to_wav ─────────────────────────────────────────────────────

    #[test]
    fn wav_total_size() {
        for n in [0usize, 1, 100, 44100] {
            assert_eq!(encode_pcm_to_wav(&vec![0.0f32; n], 48000).len(), 44 + n * 2);
        }
    }

    #[test]
    fn wav_riff_magic() {
        assert_eq!(&encode_pcm_to_wav(&[], 48000)[0..4], b"RIFF");
    }

    #[test]
    fn wav_wave_magic() {
        assert_eq!(&encode_pcm_to_wav(&[], 48000)[8..12], b"WAVE");
    }

    #[test]
    fn wav_fmt_marker() {
        assert_eq!(&encode_pcm_to_wav(&[], 48000)[12..16], b"fmt ");
    }

    #[test]
    fn wav_data_marker() {
        assert_eq!(&encode_pcm_to_wav(&[], 48000)[36..40], b"data");
    }

    #[test]
    fn wav_riff_chunk_size() {
        let n = 200usize;
        let wav = encode_pcm_to_wav(&vec![0.0f32; n], 48000);
        let size = u32::from_le_bytes(wav[4..8].try_into().unwrap());
        assert_eq!(size, (36 + n * 2) as u32);
    }

    #[test]
    fn wav_pcm_format() {
        let wav = encode_pcm_to_wav(&[], 48000);
        assert_eq!(u16::from_le_bytes(wav[20..22].try_into().unwrap()), 1);
    }

    #[test]
    fn wav_mono_channel() {
        let wav = encode_pcm_to_wav(&[], 48000);
        assert_eq!(u16::from_le_bytes(wav[22..24].try_into().unwrap()), 1);
    }

    #[test]
    fn wav_sample_rate_field() {
        for sr in [44100u32, 48000] {
            let wav = encode_pcm_to_wav(&[0.0f32; 10], sr);
            assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), sr);
        }
    }

    #[test]
    fn wav_byte_rate() {
        let sr = 48000u32;
        let wav = encode_pcm_to_wav(&[0.0f32; 10], sr);
        assert_eq!(u32::from_le_bytes(wav[28..32].try_into().unwrap()), sr * 2);
    }

    #[test]
    fn wav_block_align() {
        let wav = encode_pcm_to_wav(&[0.0f32; 10], 48000);
        assert_eq!(u16::from_le_bytes(wav[32..34].try_into().unwrap()), 2);
    }

    #[test]
    fn wav_bits_per_sample() {
        let wav = encode_pcm_to_wav(&[0.0f32; 10], 48000);
        assert_eq!(u16::from_le_bytes(wav[34..36].try_into().unwrap()), 16);
    }

    #[test]
    fn wav_data_chunk_size() {
        let n = 300usize;
        let wav = encode_pcm_to_wav(&vec![0.0f32; n], 48000);
        assert_eq!(
            u32::from_le_bytes(wav[40..44].try_into().unwrap()),
            (n * 2) as u32
        );
    }

    #[test]
    fn wav_zero_sample_encodes_zero() {
        let wav = encode_pcm_to_wav(&[0.0f32], 48000);
        assert_eq!(i16::from_le_bytes(wav[44..46].try_into().unwrap()), 0);
    }

    #[test]
    fn wav_positive_full_scale() {
        let wav = encode_pcm_to_wav(&[1.0f32], 48000);
        assert_eq!(i16::from_le_bytes(wav[44..46].try_into().unwrap()), 0x7FFF);
    }

    #[test]
    fn wav_negative_full_scale() {
        let wav = encode_pcm_to_wav(&[-1.0f32], 48000);
        assert_eq!(
            i16::from_le_bytes(wav[44..46].try_into().unwrap()),
            -0x8000i16
        );
    }

    #[test]
    fn wav_clamping_above() {
        let wav = encode_pcm_to_wav(&[2.0f32], 48000);
        assert_eq!(i16::from_le_bytes(wav[44..46].try_into().unwrap()), 0x7FFF);
    }

    #[test]
    fn wav_clamping_below() {
        let wav = encode_pcm_to_wav(&[-2.0f32], 48000);
        assert_eq!(
            i16::from_le_bytes(wav[44..46].try_into().unwrap()),
            -0x8000i16
        );
    }

    #[test]
    fn wav_multi_sample_ordering() {
        let wav = encode_pcm_to_wav(&[0.0f32, 1.0, -1.0], 48000);
        assert_eq!(i16::from_le_bytes(wav[44..46].try_into().unwrap()), 0);
        assert_eq!(i16::from_le_bytes(wav[46..48].try_into().unwrap()), 0x7FFF);
        assert_eq!(
            i16::from_le_bytes(wav[48..50].try_into().unwrap()),
            -0x8000i16
        );
    }

    // ── compute_spectrogram ───────────────────────────────────────────────────

    #[test]
    fn spectrogram_num_bins_is_positive() {
        assert!(spectrogram_num_bins(SR) > 0);
        assert!(spectrogram_num_bins(44_100.0) > 0);
    }

    #[test]
    fn spectrogram_length_is_frames_times_bins() {
        let signal = make_sine(NOTE_FREQS[0], 1.0, 0.5);
        let fs = (0.025 * SR).round() as usize;
        let st = (0.010 * SR).round() as usize;
        let nf = frame_count(signal.len(), fs, st);
        let nb = spectrogram_num_bins(SR);
        assert_eq!(compute_spectrogram(&signal, SR).len(), nf * nb);
    }

    #[test]
    fn spectrogram_values_normalized_to_unit_interval() {
        let signal = make_sine(NOTE_FREQS[0], 1.0, 0.5);
        assert!(compute_spectrogram(&signal, SR)
            .iter()
            .all(|&v| v >= 0.0 && v <= 1.0));
    }

    #[test]
    fn spectrogram_no_nan_or_inf() {
        let signal = make_sine(NOTE_FREQS[0], 1.0, 0.5);
        assert!(compute_spectrogram(&signal, SR)
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn spectrogram_all_zeros_input_returns_all_zeros() {
        // All samples 0 → raw values all ln(1e-6), min==max → normalized to 0.
        let signal = vec![0.0f32; 48000];
        assert!(compute_spectrogram(&signal, SR)
            .iter()
            .all(|&v| v.abs() < 1e-5));
    }

    #[test]
    fn spectrogram_num_bins_consistent_with_output_row_width() {
        let signal = make_sine(NOTE_FREQS[0], 1.0, 0.5);
        let fs = (0.025 * SR).round() as usize;
        let st = (0.010 * SR).round() as usize;
        let nf = frame_count(signal.len(), fs, st);
        let spec = compute_spectrogram(&signal, SR);
        assert_eq!(spec.len() / nf, spectrogram_num_bins(SR));
    }
}
