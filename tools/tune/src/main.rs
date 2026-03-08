/// tune — per-frame analysis tool for calibrating xylo-detect thresholds.
///
/// Usage:
///   cargo run -p tune -- <audio-file>
///   cargo run -p tune -- --pass 1 <audio-file>   # Pass 1 only (dominance gate)
///   cargo run -p tune -- --pass 2 <audio-file>   # Passes 1–2 (onset gating)
///   cargo run -p tune -- --pass 3 <audio-file>   # All passes — matches app (default)
///
/// Requires ffmpeg on PATH.
use analysis::{
    analyze_full, FrameStats, NOTE_NAMES, NO_PREDICTION,
    ENERGY_THRESHOLD, NOTE_DOMINANCE_THRESHOLD, ONSET_FLUX_RATIO,
    ONSET_HOLD_FRAMES, SMOOTH_HALF_WIN,
};

const SAMPLE_RATE: f32 = 48_000.0;

fn decode_audio(path: &str) -> Result<Vec<f32>, String> {
    let out = std::process::Command::new("ffmpeg")
        .args([
            "-i", path,
            "-ac", "1",
            "-ar", "48000",
            "-f", "f32le",
            "pipe:1",
            "-loglevel", "error",
        ])
        .output()
        .map_err(|e| format!("ffmpeg not found: {e}"))?;

    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }

    let samples = out
        .stdout
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    Ok(samples)
}

fn note_label(note: u8) -> &'static str {
    if note == NO_PREDICTION {
        "  —"
    } else {
        NOTE_NAMES[note as usize]
    }
}

fn flux_display(flux: f32) -> String {
    if flux == 0.0 {
        "     —".to_string()
    } else if flux > 9_999.0 {
        " >9999".to_string()
    } else {
        format!("{flux:>6.2}")
    }
}

fn print_row(f: usize, sr: f32, s: &FrameStats, pass: u8) {
    let t = (f * (0.010 * sr as f32) as usize) as f32 / sr;
    let winner_label = if s.winner == NO_PREDICTION {
        "     —".to_string()
    } else {
        format!("{:>6}", NOTE_NAMES[s.winner as usize])
    };

    // Flags
    let mut flags = Vec::new();
    if s.rms < ENERGY_THRESHOLD {
        flags.push("[silent]");
    } else {
        if s.onset { flags.push("ONSET"); }
        if s.pass1 == NO_PREDICTION && s.rms >= ENERGY_THRESHOLD {
            flags.push("low-dom");
        }
    }

    let pred = match pass {
        1 => s.pass1,
        2 => s.pass2,
        _ => s.pass3,
    };

    println!(
        "{f:>4}  {t:>5.3}  {rms:>8.5}  {winner}  {dom:>6.3}  {energy:>10.1}  {flux}  {p1:>3}  {p2:>3}  {p3:>3}  {flags}",
        rms    = s.rms,
        winner = winner_label,
        dom    = s.dominance,
        energy = s.note_energy,
        flux   = flux_display(s.flux),
        p1     = note_label(s.pass1),
        p2     = note_label(s.pass2),
        p3     = note_label(s.pass3),
        flags  = flags.join(" "),
    );
    let _ = pred; // available if caller wants to filter rows
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse --pass N and the audio path
    let mut pass: u8 = 3;
    let mut path: Option<&str> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--pass" => {
                i += 1;
                pass = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(3).min(3).max(1);
            }
            arg => path = Some(arg),
        }
        i += 1;
    }

    let path = match path {
        Some(p) => p,
        None => {
            eprintln!("Usage: tune [--pass 1|2|3] <audio-file>");
            std::process::exit(1);
        }
    };

    let samples = match decode_audio(path) {
        Ok(s) => s,
        Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
    };

    let sr = SAMPLE_RATE;
    let frames = analyze_full(&samples, sr);
    let num_frames = frames.len();
    let frame_ms = 25;
    let stride_ms = 10;

    println!("File: {} samples  {:.3}s  {} Hz", samples.len(), samples.len() as f32 / sr, sr as u32);
    println!("Frames: {num_frames}  ({frame_ms}ms / {stride_ms}ms stride)");
    println!("Showing: Pass {pass} output  (P1=dominance  P2=onset-gated  P3=mode-filtered=app)");
    println!(
        "Thresholds: RMS≥{ENERGY_THRESHOLD}  dominance≥{NOTE_DOMINANCE_THRESHOLD}  \
         flux≥{ONSET_FLUX_RATIO}  hold={ONSET_HOLD_FRAMES}  smooth±{SMOOTH_HALF_WIN}"
    );
    println!();

    let header = "  Fr   t(s)       RMS  Winner     Dom      TotalE    Flux   P1   P2   P3  Flags";
    println!("{header}");
    println!("{}", "─".repeat(header.len()));

    for (f, s) in frames.iter().enumerate() {
        print_row(f, sr, s, pass);
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    let silent = frames.iter().filter(|s| s.rms < ENERGY_THRESHOLD).count();

    let summary = |label: &str, pred_fn: fn(&FrameStats) -> u8| {
        let preds: Vec<u8> = frames.iter().map(|s| pred_fn(s)).collect();
        let valid: Vec<u8> = preds.iter().copied().filter(|&p| p != NO_PREDICTION).collect();
        let mut counts = [0usize; 8];
        for &p in &valid { counts[p as usize] += 1; }
        println!("\n── {label} ────────────────────────────────────────");
        println!("Frames with prediction : {} / {num_frames}", valid.len());
        println!("Frames silent (RMS)    : {silent}");
        println!("Note distribution:");
        for (i, name) in NOTE_NAMES.iter().enumerate() {
            let n = counts[i];
            let bar: String = "█".repeat(n);
            println!("  {name}: {n:>4}  {bar}");
        }
    };

    summary("Pass 1 — dominance gate", |s| s.pass1);
    summary("Pass 2 — onset gating",   |s| s.pass2);
    summary("Pass 3 — mode filter (app output)", |s| s.pass3);
}
