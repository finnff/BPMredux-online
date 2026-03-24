#!/usr/bin/env python3
"""
Generate variable-BPM test files for stability testing.
Takes a 60s clip and applies tempo changes:
  - 0-5s:   ramp from 1.0x to 1.2x speed
  - 5-25s:  ramp from 1.2x to 0.72x speed
  - 25-60s: hold at 0.72x speed
"""

import sys
import numpy as np
import librosa
import soundfile as sf

def generate_tempo_variant(input_path, output_path, start_sec=30, duration=60):
    """
    Generate a tempo-varying test file.

    Args:
        input_path: Path to input WAV file
        output_path: Path for output WAV file
        start_sec: Start time in input file (default 30s)
        duration: Duration of clip to extract (default 60s)
    """
    print(f"Loading {input_path}...")
    # Load audio
    y, sr = librosa.load(input_path, sr=44100, mono=True,
                         offset=start_sec, duration=duration)
    print(f"Loaded {len(y)} samples at {sr}Hz ({len(y)/sr:.1f}s)")

    # Create time-stretched segments using resampling
    output_segments = []

    print("Creating tempo ramp segments...")

    # Segment 1: Speed up from 1.0 to 1.2 (0-5s)
    # We want 5 seconds of output at varying rates
    # Use small increments for smooth ramp
    n_steps = 100
    segment_duration = 5.0  # seconds in original
    samples_per_step = int((segment_duration * sr) / n_steps)
    current_idx = 0

    for i in range(n_steps):
        progress = i / n_steps
        rate = 1.0 + 0.2 * progress  # 1.0 to 1.2
        start_idx = int(progress * segment_duration * sr)
        end_idx = int((progress + 1/n_steps) * segment_duration * sr)
        segment = y[start_idx:end_idx]

        if len(segment) > 0:
            # Time stretch (preserves pitch, changes tempo)
            stretched = librosa.effects.time_stretch(segment, rate=rate)
            output_segments.append(stretched)

    print(f"  Segment 1 (speed up 1.0 -> 1.2x): {len(output_segments[-1]) / sr:.2f}s")

    # Segment 2: Slow down from 1.2 to 0.72 (5-25s = 20s original time)
    n_steps = 200
    segment_duration = 20.0
    for i in range(n_steps):
        progress = i / n_steps
        rate = 1.2 - 0.48 * progress  # 1.2 to 0.72
        start_idx = int(5.0 * sr + progress * segment_duration * sr)
        end_idx = int(5.0 * sr + (progress + 1/n_steps) * segment_duration * sr)
        segment = y[start_idx:end_idx]

        if len(segment) > 0:
            stretched = librosa.effects.time_stretch(segment, rate=rate)
            output_segments.append(stretched)

    print(f"  Segment 2 (slow down 1.2 -> 0.72x): {len(output_segments[-1]) / sr:.2f}s")

    # Segment 3: Hold at 0.72 (25-60s = 35s original time)
    start_idx = int(25.0 * sr)
    segment = y[start_idx:int(60.0 * sr)]
    stretched = librosa.effects.time_stretch(segment, rate=0.72)
    output_segments.append(stretched)
    print(f"  Segment 3 (hold 0.72x): {len(stretched) / sr:.2f}s")

    # Concatenate and save
    print("Concatenating segments...")
    output = np.concatenate(output_segments)
    print(f"Total output duration: {len(output) / sr:.2f}s")

    print(f"Writing to {output_path}...")
    sf.write(output_path, output, sr)
    print(f"Done! Generated {output_path}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: generate_tempo_variants.py <input.wav> <output.wav>")
        print("\nExample:")
        print("  python3 generate_tempo_variants.py techno_original.wav techno_variant.wav")
        sys.exit(1)

    generate_tempo_variant(sys.argv[1], sys.argv[2])
