# Audio-Proccessing-Generation

A set of tools developed by noiritgc to process real audios and generate synthesized pitches based on the harmonic structures.

Possible uses:

You have a sample pitch drone but its short and has imperfections. You can use these tools to synthesize the sound, and up to your desire length!

## How it works

There are two stages, split into two scripts.

**`analyze.py`** takes a short recording of a single sustained pitch and pulls out its underlying timbre: a smooth spectral envelope describing how the source's resonances shape the sound across frequency, plus (optionally) a breath/noise profile. It's built around the WORLD vocoder (`pyworld`), the same kind of decomposition used in a lot of speech and singing synthesis work. The result gets saved as a `.npz` report - a small file with everything the next script needs.

**`synthesize_drones.py`** reads that report back and rebuilds the sound from scratch at exact, mathematically correct pitches using additive synthesis - summing sine waves at each harmonic, with each harmonic's amplitude sampled from the extracted envelope curve. Since every harmonic's frequency, amplitude and phase stay fixed for the whole note, the output has zero wobble regardless of how unstable the source recording was.

Rather than rendering minutes of unique audio for every note, the synthesis script builds a few seconds of tone, crossfades it into a seamless loop, and tiles that loop out to whatever length you ask for. Generating 60 three-minute drones this way takes seconds, not minutes.

### Why not just pitch-shift the original clip

Stretching a recording directly, or reusing the same harmonic-to-harmonic ratios at a much higher or lower pitch, tends to fall apart a few octaves away from the source note. Real instrument bodies have resonances that sit at fixed frequencies - they don't move with whatever note is being played. Sampling a continuous envelope curve at each new harmonic's frequency (instead of copying fixed ratios) keeps that resonance character intact across a much wider range than the source instrument could ever actually play.

## Requirements

- Python 3.9+
- `numpy`
- `librosa`
- `pyworld`
- `soundfile`
- `ffmpeg` on your PATH if you're analyzing compressed formats like mp3 (librosa needs it to decode them)

```bash
pip install numpy librosa pyworld soundfile
```

`pyworld` depends on `pkg_resources`, which recent versions of `setuptools` (81+) no longer ship. If you hit:

```
ModuleNotFoundError: No module named 'pkg_resources'
```

pin setuptools down and reinstall:

```bash
pip install "setuptools<81"
```

## Usage

### 1. Analyze a source recording

```bash
python analyze.py source_clip.mp3 -o report.npz
```

| Flag | Default | Description |
|---|---|---|
| `-o, --output` | `report.npz` | Where to save the report |
| `--no-noise-filter` | off | Skip extracting the breath/noise profile entirely |
| `--trim-fraction` | `0.15` | How much of the note's attack and release to discard before summarizing the timbre |

Worth running twice - once with `--no-noise-filter` and once without - to compare how much the noise component actually adds before settling on one for synthesis.

### 2. Generate drones from a report

```bash
python synthesize_drones.py report.npz -o drones/
```

| Flag | Default | Description |
|---|---|---|
| `-o, --output-dir` | `drones` | Output folder for the generated .wav files |
| `--sr` | `44100` | Output sample rate |
| `--duration-min` | `3.0` | Length of each generated drone, in minutes |
| `--octave-start` / `--octave-end` | `2` / `6` | Pitch range to generate (scientific pitch notation) |
| `--loop-seconds` | `6.0` | Length of the internal loop buffer before tiling |
| `--crossfade-seconds` | `0.5` | Crossfade length used to make the loop seamless |
| `--use-noise` | off | Blend in the noise profile from the report, if one exists |
| `--noise-mix` | `1.0` | Multiplier on the noise bed's level |
| `--seed` | `0` | Random seed for the noise generator |

This writes one `.wav` file per note across the requested range, named after the note and its exact frequency, e.g. `C4_261.63Hz.wav`.

## Repository Contents

### Pre-generated Files

#### Analysis Reports (`analysis/`)
The repo includes pre-analyzed timbre reports for three instruments:
- **`trombone_analysis.npz`** - Spectral and aperiodicity data extracted from a trombone sustain
- **`trumpet_analysis.npz`** - Spectral and aperiodicity data extracted from a trumpet sustain
- **`violin_analysis.npz`** - Spectral and aperiodicity data extracted from a violin sustain

#### Generated Drone Collections
Each of the following directories contains **48 chromatic drone files** (C2 through G#6) in MP3 format, synthesized from the corresponding instrument's analysis report:
- **`trombone/`** - Trombone drones across a 5-octave range
- **`trumpet/`** - Trumpet drones across a 5-octave range  
- **`violin/`** - Violin drones across a 5-octave range

Each file is named by note and frequency (e.g., `C4_261.63Hz.mp3`) and is approximately 3 minutes long, perfect for:
- Tuning reference tones
- Drones for music practice
- Meditative/ambient soundscapes
- Matching instrument timbre for harmonic exploration

### Scripts

- **`analyze.py`** - Extract instrument timbre from a short recording (entry point for custom instruments)
- **`synthesize_drones.py`** - Generate drones from a timbre report

## Notes

- Works on any single sustained pitch - strings, voice, winds, synth pads all work fine. Inharmonic sources like bells or most percussion aren't a good fit, since the analysis assumes energy organized around a fundamental and its integer multiples.
- Source clips don't need to be long. A couple of seconds of clean, steady sustain is usually enough once the attack and release get trimmed off.
- The `.npz` reports are just plain numpy arrays under the hood (spectral envelope, frequency axis, optional aperiodicity, reference pitch). Easy to inspect, plot, or blend between two different reports if you want a hybrid timbre.