#!/usr/bin/env python3
"""
Generate test fixture files for stability testing.

For a given input audio file:
  - Extracts 60 seconds from the MIDDLE
  - Saves as {output_base}_original.wav  (constant tempo)
  - Applies linear ramp 1.0x -> 1.3x and saves as {output_base}_increasing.wav
  - Applies linear ramp 1.0x -> 0.7x and saves as {output_base}_decreasing.wav

Usage:
  python3 test/scripts/generate_tempo_variants.py <input.wav> <output_base>

Example:
  python3 test/scripts/generate_tempo_variants.py \\
    test/e2e/fixtures/techno_original.wav \\
    test/e2e/fixtures/techno
"""

import sys
import numpy as np
import librosa
import soundfile as sf


def apply_tempo_ramp(clip, start_rate, end_rate, sr, n_steps=200):
    """
    Apply a linear tempo ramp to a clip using time_stretch.

    Args:
        clip: numpy array of audio samples
        start_rate: initial stretch rate (1.0 = original speed)
        end_rate: final stretch rate
        sr: sample rate
        n_steps: number of chunks for smooth ramp

    Returns:
        numpy array with tempo-ramped audio
    """
    n = len(clip)
    samples_per_step = n // n_steps
    segments = []

    for i in range(n_steps):
        progress = i / n_steps
        rate = start_rate + (end_rate - start_rate) * progress
        s = i * samples_per_step
        e = (i + 1) * samples_per_step if i < n_steps - 1 else n
        seg = clip[s:e]
        if len(seg) > 0:
            stretched = librosa.effects.time_stretch(seg, rate=rate)
            segments.append(stretched)

    return np.concatenate(segments)


def generate_variants(input_path, output_base):
    """Generate original, increasing, and decreasing tempo variants."""
    print(f"Loading {input_path}...")
    y, sr = librosa.load(input_path, sr=44100, mono=True)
    duration = len(y) / sr
    print(f"  Duration: {duration:.1f}s at {sr}Hz")

    if duration < 60:
        print(f"ERROR: Input is only {duration:.1f}s; need at least 60s")
        sys.exit(1)

    # Extract middle 60 seconds
    start_sec = (duration - 60) / 2
    print(f"  Extracting 60s from {start_sec:.1f}s to {start_sec + 60:.1f}s (middle)")
    clip = librosa.load(input_path, sr=44100, mono=True,
                        offset=start_sec, duration=60)[0]
    print(f"  Clip: {len(clip)} samples ({len(clip)/sr:.1f}s)")

    # Save original (unchanged 60s clip)
    original_path = f"{output_base}_original.wav"
    sf.write(original_path, clip, sr)
    print(f"  Saved: {original_path}")

    # Generate increasing (1.0x -> 1.3x)
    print("  Generating increasing variant (1.0x -> 1.3x)...")
    increasing = apply_tempo_ramp(clip, 1.0, 1.3, sr)
    increasing_path = f"{output_base}_increasing.wav"
    sf.write(increasing_path, increasing, sr)
    print(f"  Saved: {increasing_path} ({len(increasing)/sr:.1f}s)")

    # Generate decreasing (1.0x -> 0.7x)
    print("  Generating decreasing variant (1.0x -> 0.7x)...")
    decreasing = apply_tempo_ramp(clip, 1.0, 0.7, sr)
    decreasing_path = f"{output_base}_decreasing.wav"
    sf.write(decreasing_path, decreasing, sr)
    print(f"  Saved: {decreasing_path} ({len(decreasing)/sr:.1f}s)")

    print("Done.")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: generate_tempo_variants.py <input.wav> <output_base>")
        print("\nExample:")
        print("  python3 generate_tempo_variants.py techno_original.wav test/e2e/fixtures/techno")
        sys.exit(1)

    generate_variants(sys.argv[1], sys.argv[2])
