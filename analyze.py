"""
analyze.py
===================

PURPOSE
-------
Analyze a short recording (e.g. your 5-second wobbly trumpet drone) and extract
the reusable *timbral* characteristics of the instrument, saving them to a
single .npz report file. That report is later consumed by
synthesize_drones.py to generate perfectly stable drones at any pitch.

WHY NOT JUST "SIMPLE FFT"?
---------------------------
A single FFT snapshot only gives you the frequency content at one instant, and
it doesn't separate:
  (a) the fundamental pitch (which wobbles due to human imperfection), from
  (b) the instrument's *timbre* -- the relative loudness of each harmonic /
      the shape of the instrument body's resonances, which is what actually
      makes it sound "trumpet-like" and is largely independent of pitch.

Instead, this script uses the WORLD vocoder (via the `pyworld` library), a
well-established analysis/synthesis tool from speech and singing synthesis
research, which cleanly decomposes the signal into three things:
  1. F0 contour       -- the wobbly pitch curve (kept only as metadata/reference,
                          NEVER used as a synthesis target -- your synthesized
                          drones will be locked to exact equal-temperament
                          frequencies instead).
  2. Spectral envelope -- a smooth, continuous curve describing how much energy
                          exists at every frequency, independent of exactly
                          which harmonic that energy belongs to. This is the
                          "resonance fingerprint" of the trumpet body and is
                          the main thing we want to reuse across all pitches.
  3. Aperiodicity      -- (OPTIONAL) how "noisy" vs. "tonal" the sound is in
                          each frequency band -- this is roughly the breath /
                          air noise component. You can turn this off entirely
                          to test how the drones sound with a purely tonal
                          (noise-free) timbre.

WHY A SMOOTH ENVELOPE INSTEAD OF FIXED HARMONIC RATIOS?
---------------------------------------------------------
A trumpet's body resonances sit at FIXED frequencies -- they don't move when
you play a different note. If we just recorded "harmonic 3 is 60% as loud as
harmonic 1" and blindly reapplied that ratio to a note two octaves away, we'd
break the physical relationship between harmonics and body resonances and it
would sound synthetic. By storing a continuous envelope curve (amplitude vs.
frequency in Hz, not vs. harmonic number), the synthesis script can sample
that same curve at whatever frequencies the new note's harmonics land on --
which is the physically correct way to carry a timbre across a wide pitch
range.

USAGE
-----
    python analyze.py my_trumpet_clip.mp3 -o trumpet_report.npz

    # To test how it sounds WITHOUT the breath-noise / aperiodicity data:
    python analyze.py my_trumpet_clip.mp3 -o trumpet_report_no_noise.npz --no-noise-filter

DEPENDENCIES
------------
    pip install numpy librosa pyworld soundfile
    (librosa's mp3 decoding also requires ffmpeg to be installed on your system)
"""

import argparse
import os

import numpy as np
import librosa
import pyworld as pw


def load_audio_mono_float64(path, target_sr=None):
    """
    Load an audio file (mp3, wav, etc.) as a mono float64 numpy array.

    pyworld specifically requires float64 input (not float32), so we cast
    explicitly here rather than relying on librosa's default float32 output.

    Parameters
    ----------
    path : str
        Path to the input audio file.
    target_sr : int or None
        If None, librosa preserves the file's original sample rate. You
        generally want this -- resampling before analysis can slightly blur
        the spectral envelope.

    Returns
    -------
    audio : np.ndarray, shape (n_samples,), dtype float64
    sr : int, the sample rate actually used
    """
    try:
        audio, sr = librosa.load(path, sr=target_sr, mono=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load '{path}'. If this is an mp3, make sure ffmpeg "
            f"is installed and on your PATH (librosa/audioread needs it to "
            f"decode mp3 files). Original error: {exc}"
        )

    # pyworld's C-extension functions expect float64, not librosa's default
    # float32 -- if you skip this cast, pyworld will raise a cryptic error.
    audio = audio.astype(np.float64)
    return audio, sr


def find_steady_region(f0, trim_fraction=0.05):
    """
    Locate the "steady" portion of the recording to analyze, trimming off the
    attack (breath onset, pitch settling in) and release (note tailing off)
    at each end of the sustained note -- both of which are unrepresentative
    of the instrument's core, sustained timbre.

    We do this by:
      1. Finding the first and last frame where pyworld detected a voiced
         pitch (f0 > 0) -- this brackets the sustained note.
      2. Trimming `trim_fraction` of that span off BOTH ends, so we keep only
         the stable middle portion.

    Parameters
    ----------
    f0 : np.ndarray
        Frame-by-frame fundamental frequency contour from pyworld (0 where
        unvoiced/silent).
    trim_fraction : float
        Fraction (0-0.5) of the voiced span to trim from each end.
        0.15 means "keep the middle 70%".

    Returns
    -------
    steady_start, steady_end : int
        Frame indices (into f0 / the spectral envelope array) bracketing the
        steady region to analyze.
    """
    voiced_frames = np.where(f0 > 0)[0]
    if len(voiced_frames) == 0:
        raise RuntimeError(
            "No voiced pitch was detected anywhere in the recording -- "
            "check that the clip actually contains a sustained note, or try "
            "a cleaner/louder recording."
        )

    start_idx = voiced_frames[0]
    end_idx = voiced_frames[-1]
    voiced_span = end_idx - start_idx

    trim_frames = int(round(voiced_span * trim_fraction))
    steady_start = start_idx + trim_frames
    steady_end = end_idx - trim_frames

    # Safety fallback: if the clip is very short and trimming would leave
    # nothing, just use the full voiced span instead.
    if steady_end <= steady_start:
        steady_start, steady_end = start_idx, end_idx

    return steady_start, steady_end


def analyze(input_path, output_path, compute_noise_profile=True, trim_fraction=0.05):
    """
    Run the full analysis pipeline and save a .npz report.

    Parameters
    ----------
    input_path : str
        Path to the source recording (mp3, wav, etc.)
    output_path : str
        Where to save the .npz report.
    compute_noise_profile : bool
        If True, also extract the aperiodicity (breath-noise) profile via
        pyworld's D4C algorithm and store it in the report. If False, this
        step is skipped entirely (faster, and lets you A/B test how the
        synthesized drones sound with vs. without a noise component).
    trim_fraction : float
        See find_steady_region() above.
    """
    print(f"Loading '{input_path}'...")
    audio, fs = load_audio_mono_float64(input_path)
    duration_sec = len(audio) / fs
    print(f"  Loaded {duration_sec:.2f}s of audio at {fs} Hz sample rate.")

    # --- Step 1: F0 (pitch) estimation -------------------------------------
    # harvest() is pyworld's high-accuracy F0 estimator. stonemask() then
    # refines those estimates for extra precision. This F0 contour WILL show
    # the trumpet's natural wobble -- that's expected and fine, because we
    # only use it as reference metadata, never as a synthesis target.
    print("Estimating pitch (F0) contour...")
    f0, t = pw.harvest(audio, fs)
    f0 = pw.stonemask(audio, f0, t, fs)

    # --- Step 2: Spectral envelope estimation -------------------------------
    # cheaptrick() computes a smooth spectral envelope for every analysis
    # frame. Shape: (n_frames, fft_size // 2 + 1). This is the "resonance
    # fingerprint" we care about most.
    print("Estimating spectral envelope...")
    spectral_envelope_frames = pw.cheaptrick(audio, f0, t, fs)
    fft_size = pw.get_cheaptrick_fft_size(fs)

    # Build the frequency (Hz) axis corresponding to each column of the
    # envelope array, so the synthesis script can later interpolate
    # "amplitude at frequency X Hz" rather than "amplitude at bin index N".
    freq_axis = np.linspace(0, fs / 2, spectral_envelope_frames.shape[1])

    # --- Step 3: Find the steady portion of the note to summarize ----------
    steady_start, steady_end = find_steady_region(f0, trim_fraction)
    steady_time_range = (t[steady_start], t[steady_end - 1])
    print(
        f"Using steady region {steady_time_range[0]:.2f}s - "
        f"{steady_time_range[1]:.2f}s for the representative timbre "
        f"(trimmed {trim_fraction*100:.0f}% off each end of the sustained note)."
    )

    steady_f0 = f0[steady_start:steady_end]
    voiced_steady_f0 = steady_f0[steady_f0 > 0]
    f0_median = float(np.median(voiced_steady_f0)) if len(voiced_steady_f0) else 0.0
    f0_std = float(np.std(voiced_steady_f0)) if len(voiced_steady_f0) else 0.0
    print(f"  Reference pitch of source clip: {f0_median:.2f} Hz "
          f"(wobble/std dev: {f0_std:.2f} Hz -- for info only, not used in synthesis)")

    # Collapse the steady region's per-frame envelopes down to a SINGLE
    # representative curve. We use the MEDIAN (not the mean) across frames
    # because it's more robust to the natural amplitude wobble/vibrato in a
    # human performance -- outlier frames (e.g. a brief dip) won't skew a
    # median the way they'd skew an average.
    steady_envelope_frames = spectral_envelope_frames[steady_start:steady_end]
    representative_envelope = np.median(steady_envelope_frames, axis=0)

    # --- Assemble the report -------------------------------------------------
    report = {
        "fs": fs,                                  # sample rate used during analysis
        "fft_size": fft_size,                      # FFT size used by pyworld for the envelope
        "freq_axis": freq_axis,                    # Hz value for each envelope column
        "spectral_envelope": representative_envelope,  # the main "timbre fingerprint" (1D)
        "f0_median": f0_median,                    # reference only -- NOT a synthesis target
        "f0_std": f0_std,                           # reference only -- shows how wobbly the source was
        "f0_contour": f0,                            # full per-frame pitch contour, for plotting/debugging
        "has_noise_profile": compute_noise_profile,  # tells synthesize_drones.py whether 'aperiodicity' exists
        "source_file": os.path.basename(input_path),
    }

    # --- Step 4 (OPTIONAL): Aperiodicity / breath-noise profile -------------
    if compute_noise_profile:
        print("Estimating aperiodicity (breath-noise) profile...")
        aperiodicity_frames = pw.d4c(audio, f0, t, fs)
        steady_aperiodicity_frames = aperiodicity_frames[steady_start:steady_end]
        representative_aperiodicity = np.median(steady_aperiodicity_frames, axis=0)
        report["aperiodicity"] = representative_aperiodicity
    else:
        print("Skipping noise/aperiodicity extraction (--no-noise-filter was set).")

    # --- Save --------------------------------------------------------------
    np.savez(output_path, **report)
    print(f"Saved report to '{output_path}'.")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Analyze a recording and extract a reusable timbre report (.npz)."
    )
    parser.add_argument("input_mp3", help="Path to the source recording (mp3, wav, etc.)")
    parser.add_argument(
        "-o", "--output", default="trumpet_report.npz",
        help="Path to write the .npz report to (default: trumpet_report.npz)"
    )
    parser.add_argument(
        "--no-noise-filter", action="store_true",
        help="Skip extracting the breath-noise/aperiodicity profile entirely. "
             "Use this to test how the synthesized drones sound WITHOUT any "
             "noise component (purely tonal)."
    )
    parser.add_argument(
        "--trim-fraction", type=float, default=0.15,
        help="Fraction of the sustained note's attack/release to trim off "
             "each end before summarizing the timbre (default: 0.15)."
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    analyze(
        input_path=args.input_mp3,
        output_path=args.output,
        compute_noise_profile=not args.no_noise_filter,
        trim_fraction=args.trim_fraction,
    )