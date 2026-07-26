"""
synthesize_drones.py
=====================

PURPOSE
-------
Reads the .npz timbre report produced by analyze_trumpet.py and generates
perfectly stable drone recordings for every pitch from a chosen octave range
(default: octave 2 through octave 6, i.e. C2 - B6), using additive synthesis
driven by the extracted spectral envelope.

HOW THE SYNTHESIS WORKS
------------------------
For each target note:
  1. Compute its EXACT 12-tone-equal-temperament frequency mathematically
     (never derived from the trumpet's wobbly recorded pitch -- this keeps
     your tuning-app pitches dead accurate).
  2. Generate a sine wave for every harmonic of that frequency (F0, 2*F0,
     3*F0, ...) up to the Nyquist frequency.
  3. Look up each harmonic's amplitude by SAMPLING the stored spectral
     envelope curve at that harmonic's exact frequency (not by reusing fixed
     ratios) -- this is what correctly preserves the trumpet's resonance
     character across a pitch range no real trumpet could ever play.
  4. Sum all the harmonics together. Because every harmonic's frequency,
     amplitude, and phase are perfectly fixed for the whole duration, the
     result has ZERO wobble by construction -- "perfectly stable" comes for
     free from this method, no extra stabilization needed.
  5. (Optional) Blend in a noise bed shaped by the aperiodicity data from the
     report, if you analyzed with the noise filter enabled and want to test
     how much realism it adds.

EFFICIENCY: WE DON'T LITERALLY RENDER 3 CONTINUOUS MINUTES OF UNIQUE AUDIO
----------------------------------------------------------------------------
Since the tone is static (fixed frequencies/amplitudes/phases), it doesn't
need 3 minutes of unique synthesis. Instead we:
  1. Render a short buffer (a few seconds).
  2. Turn it into a SEAMLESS LOOP using an equal-power crossfade at the loop
     boundary (this works regardless of exact phase alignment, so it never
     compromises tuning accuracy).
  3. Tile that short loop to fill the full requested duration.
This is dramatically faster and cheaper than synthesizing 3 unique minutes
per pitch, and it's the same trick pad/drone instruments use internally.

USAGE
-----
    python synthesize_drones.py trumpet_report.npz -o drones/

    # Also test with the breath-noise component blended in (only works if
    # the report was analyzed WITHOUT --no-noise-filter):
    python synthesize_drones.py trumpet_report.npz -o drones_with_noise/ --use-noise

DEPENDENCIES
------------
    pip install numpy soundfile
"""

import argparse
import os

import numpy as np
import soundfile as sf


# The 12 pitch classes in a chromatic scale, used to build note names like "C4", "F#3".
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_to_freq(note_name, octave):
    """
    Convert a note name + octave (scientific pitch notation, A4 = 440 Hz) to
    its exact 12-tone-equal-temperament frequency in Hz.

    This is a pure mathematical formula -- it has nothing to do with the
    trumpet recording. That's intentional: pitch accuracy for a tuning app
    must come from math, not from an imperfect human performance.
    """
    semitone_index = NOTE_NAMES.index(note_name)
    # MIDI note number, using the convention that C4 = MIDI note 60.
    midi_number = (octave + 1) * 12 + semitone_index
    freq_hz = 440.0 * (2.0 ** ((midi_number - 69) / 12.0))
    return freq_hz


def load_report(path):
    """Load the .npz report into a plain dict of numpy arrays/scalars."""
    data = np.load(path)
    return {key: data[key] for key in data.files}


def make_lookup_function(freq_axis, curve):
    """
    Build a function that returns an interpolated value from `curve` at any
    arbitrary frequency, by linearly interpolating between the sampled
    points in `freq_axis`.

    Frequencies above the analyzed range are clamped to the highest analyzed
    frequency's value (rather than extrapolated), which is a safe default
    since spectral envelopes naturally taper toward zero near the top of the
    analyzed range anyway.
    """
    max_freq = freq_axis[-1]

    def lookup(freq_query):
        freq_clamped = np.clip(freq_query, 0.0, max_freq)
        return float(np.interp(freq_clamped, freq_axis, curve))

    return lookup


def synth_periodic_tone(freq0, duration_sec, sr, envelope_lookup,
                         aperiodicity_lookup=None, use_noise=False):
    """
    Generate a purely periodic (perfectly stable) additive-synthesis tone at
    fundamental frequency `freq0`, shaped by the stored spectral envelope.

    Parameters
    ----------
    freq0 : float
        Target fundamental frequency in Hz (exact, from note_to_freq()).
    duration_sec : float
        Length of the buffer to generate (this is the short loop buffer, NOT
        the full 3-minute output -- see tile_to_duration() below).
    sr : int
        Output sample rate.
    envelope_lookup : callable
        Function built by make_lookup_function() for the spectral envelope.
    aperiodicity_lookup : callable or None
        Function built by make_lookup_function() for the aperiodicity curve,
        or None if the report has no noise profile.
    use_noise : bool
        If True (and aperiodicity_lookup is available), each harmonic's
        amplitude is scaled DOWN by its periodic-energy ratio
        (sqrt(1 - aperiodicity)), since some of that harmonic's original
        energy was noise rather than tone. The noise itself is added
        separately in synth_noise_bed().

    Returns
    -------
    signal : np.ndarray, shape (n_samples,)
    n_samples : int
    """
    n_samples = int(round(duration_sec * sr))
    t = np.arange(n_samples) / sr
    signal = np.zeros(n_samples, dtype=np.float64)
    nyquist = sr / 2.0

    harmonic_number = 1
    while True:
        harmonic_freq = freq0 * harmonic_number
        if harmonic_freq >= nyquist:
            break  # stop once harmonics go above what this sample rate can represent

        amplitude = envelope_lookup(harmonic_freq)

        if use_noise and aperiodicity_lookup is not None:
            aperiodicity_ratio = aperiodicity_lookup(harmonic_freq)  # 0 = fully tonal, 1 = fully noise
            periodic_ratio = np.sqrt(max(0.0, 1.0 - aperiodicity_ratio))
            amplitude = amplitude * periodic_ratio

        # Fixed frequency, fixed amplitude, fixed phase (starts at 0) for the
        # ENTIRE duration -- this is exactly why the result is "perfectly
        # stable" with zero wobble.
        signal += amplitude * np.sin(2.0 * np.pi * harmonic_freq * t)

        harmonic_number += 1

    return signal, n_samples


def synth_noise_bed(n_samples, sr, envelope_lookup, aperiodicity_lookup, rng):
    """
    Generate a bed of filtered noise shaped by the envelope * aperiodicity
    curves, representing the breath/air noise component of the instrument.

    Method: generate white noise, take its FFT, replace the magnitude at
    every frequency bin with our target magnitude (derived from the report),
    keep the random phase, then inverse-FFT back to the time domain. This
    gives us noise with an arbitrary, precisely controlled frequency shape.
    """
    white_noise = rng.standard_normal(n_samples)
    spectrum = np.fft.rfft(white_noise)
    bin_freqs = np.fft.rfftfreq(n_samples, d=1.0 / sr)

    # Target magnitude at each FFT bin: envelope * sqrt(aperiodicity ratio).
    # sqrt() because aperiodicity is roughly an energy (power) ratio, and we
    # want an amplitude scale factor.
    target_amplitude = np.array([
        envelope_lookup(f) * np.sqrt(max(0.0, aperiodicity_lookup(f)))
        for f in bin_freqs
    ])

    # Normalize the white noise spectrum's magnitude to 1 at every bin
    # (keeping its random phase), then impose our target shape on top of it.
    magnitude = np.abs(spectrum)
    magnitude[magnitude == 0] = 1e-12  # avoid divide-by-zero
    shaped_spectrum = (spectrum / magnitude) * target_amplitude

    shaped_noise = np.fft.irfft(shaped_spectrum, n=n_samples)
    return shaped_noise


def make_seamless_loop(signal, sr, crossfade_sec=0.5):
    """
    Turn a buffer into a seamlessly loopable clip using an equal-power
    crossfade between its tail and its head.

    Why crossfade instead of relying on exact-cycle-count phase alignment?
    Because forcing the buffer length to land on an exact whole number of
    periods can require nudging the sample count (and therefore the
    effective frequency) very slightly -- undesirable for a tuning app where
    pitch accuracy matters. A crossfade sidesteps that entirely: it works
    regardless of phase alignment and never touches the target frequency.

    Parameters
    ----------
    signal : np.ndarray
        The rendered buffer (tone, optionally + noise bed).
    sr : int
    crossfade_sec : float
        Length of the crossfade region, in seconds.

    Returns
    -------
    looped : np.ndarray
        A shorter buffer (len(signal) - fade_samples) that loops seamlessly
        when repeated back-to-back.
    """
    fade_samples = int(round(crossfade_sec * sr))
    if fade_samples * 2 >= len(signal):
        # Safety fallback for very short buffers: use a quarter of the length.
        fade_samples = max(1, len(signal) // 4)

    loop_len = len(signal) - fade_samples
    head = signal[:fade_samples]
    tail = signal[loop_len:loop_len + fade_samples]

    # Equal-power crossfade curves (sqrt taper) avoid the slight volume dip
    # you'd get from a plain linear crossfade.
    fade_in = np.linspace(0.0, 1.0, fade_samples)
    fade_out = 1.0 - fade_in
    fade_in_eq = np.sqrt(fade_in)
    fade_out_eq = np.sqrt(fade_out)

    blended_boundary = head * fade_in_eq + tail * fade_out_eq

    looped = signal[:loop_len].copy()
    looped[:fade_samples] = blended_boundary
    return looped


def tile_to_duration(loop_signal, sr, target_duration_sec, edge_fade_sec=0.05):
    """
    Repeat a seamless loop buffer until it reaches (at least) the target
    duration, then trim to exactly that length and apply a very short
    fade-in/out at the absolute start/end of the FULL file (not at each
    internal loop repeat) to avoid a click when playback starts/stops.
    """
    target_samples = int(round(target_duration_sec * sr))
    n_repeats = int(np.ceil(target_samples / len(loop_signal)))
    full = np.tile(loop_signal, n_repeats)[:target_samples]

    edge_fade_samples = int(round(edge_fade_sec * sr))
    if edge_fade_samples > 0 and edge_fade_samples * 2 < len(full):
        fade_curve = np.linspace(0.0, 1.0, edge_fade_samples)
        full[:edge_fade_samples] *= fade_curve
        full[-edge_fade_samples:] *= fade_curve[::-1]

    return full


def normalize_peak(signal, target_peak=0.9):
    """Scale the signal so its peak absolute value hits target_peak (headroom to avoid clipping)."""
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * target_peak
    return signal


def generate_all_drones(report_path, output_dir, sr=44100, duration_min=3.0,
                         octave_start=2, octave_end=6, loop_seconds=6.0,
                         crossfade_seconds=0.5, use_noise=False, noise_mix=1.0,
                         seed=0):
    """
    Main driver: loads the report once, then generates and writes one .mp3
    file per pitch across the requested octave range.
    """
    report = load_report(report_path)
    freq_axis = report["freq_axis"]
    envelope = report["spectral_envelope"]
    envelope_lookup = make_lookup_function(freq_axis, envelope)

    has_noise_profile = bool(report["has_noise_profile"])
    aperiodicity_lookup = None
    if has_noise_profile:
        aperiodicity = report["aperiodicity"]
        aperiodicity_lookup = make_lookup_function(freq_axis, aperiodicity)
    elif use_noise:
        print("NOTE: this report has no noise profile (it was analyzed with "
              "--no-noise-filter). Ignoring --use-noise and generating a "
              "purely tonal signal instead.")
        use_noise = False

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    total_notes = (octave_end - octave_start + 1) * len(NOTE_NAMES)
    print(f"Generating {total_notes} drones "
          f"(octave {octave_start} to {octave_end}), "
          f"{duration_min} min each, noise={'on' if use_noise else 'off'}...")

    for octave in range(octave_start, octave_end + 1):
        for note_name in NOTE_NAMES:
            freq0 = note_to_freq(note_name, octave)

            # 1. Render a short buffer (tone, + optional noise bed).
            tone, n_samples = synth_periodic_tone(
                freq0, loop_seconds, sr, envelope_lookup,
                aperiodicity_lookup, use_noise
            )
            if use_noise and aperiodicity_lookup is not None:
                noise_bed = synth_noise_bed(
                    n_samples, sr, envelope_lookup, aperiodicity_lookup, rng
                )
                tone = tone + noise_bed * noise_mix

            # 2. Make it a seamless loop, then tile it out to full length.
            looped = make_seamless_loop(tone, sr, crossfade_seconds)
            full_length_signal = tile_to_duration(looped, sr, duration_min * 60.0)

            # 3. Normalize and write to disk.
            full_length_signal = normalize_peak(full_length_signal, target_peak=0.9)

            safe_note_name = note_name.replace("#", "Sharp")  # '#' is awkward in filenames
            filename = f"{safe_note_name}{octave}.mp3"
            filepath = os.path.join(output_dir, filename)
            sf.write(filepath, full_length_signal.astype(np.float32), sr, format='MP3')
            print(f"  wrote {filepath}  (f0 = {freq0:.2f} Hz)")

    print("Done.")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generate stable drone recordings from a .npz timbre report."
    )
    parser.add_argument("report", help="Path to the .npz report from analyze_trumpet.py")
    parser.add_argument("-o", "--output-dir", default="drones",
                         help="Directory to write the generated .mp3 files into")
    parser.add_argument("--sr", type=int, default=44100, help="Output sample rate")
    parser.add_argument("--duration-min", type=float, default=3.0,
                         help="Length of each output drone, in minutes")
    parser.add_argument("--octave-start", type=int, default=2)
    parser.add_argument("--octave-end", type=int, default=6)
    parser.add_argument("--loop-seconds", type=float, default=6.0,
                         help="Length of the internal loop buffer before tiling (default: 6s)")
    parser.add_argument("--crossfade-seconds", type=float, default=0.5,
                         help="Length of the crossfade used to make the loop seamless")
    parser.add_argument("--use-noise", action="store_true",
                         help="Blend in the extracted breath-noise component "
                              "(only works if the report has one -- see analyze_trumpet.py)")
    parser.add_argument("--noise-mix", type=float, default=1.0,
                         help="Multiplier on the noise bed's level, for creative control (default: 1.0)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the noise generator")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    generate_all_drones(
        report_path=args.report,
        output_dir=args.output_dir,
        sr=args.sr,
        duration_min=args.duration_min,
        octave_start=args.octave_start,
        octave_end=args.octave_end,
        loop_seconds=args.loop_seconds,
        crossfade_seconds=args.crossfade_seconds,
        use_noise=args.use_noise,
        noise_mix=args.noise_mix,
        seed=args.seed,
    )