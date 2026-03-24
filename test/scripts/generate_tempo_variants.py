#!/usr/bin/env python3
"""
Generate stable test files for stability testing.

Extracts 60 seconds from the MIDDLE of a track and creates:
  - {base}_original.wav - 60s constant tempo (unchanged)
  - {base}_increasing.wav - 60s with tempo 1.0x -> 1.3x
  - {base}_decreasing.wav - 60s with tempo 1.0x -> 0.7x

Usage:
  python3.12 test/scripts/generate_tempo_variants.py <input.wav> <output_base>
"""

import sys
import os
import numpy as np
import librosa
import soundfile as sf


def apply_tempo_ramp(audio, start_rate, end_rate, sr):
    """
    Apply a linear tempo ramp using segment-based resampling.
    Splits audio into segments and applies different rates to each.
    """
    duration = len(audio) / sr
    
    # Split into 1-second segments
    segment_duration = 1.0
    num_segments = int(duration / segment_duration)
    segment_samples = int(segment_duration * sr)
    
    output_segments = []
    
    for i in range(num_segments):
        # Calculate rate for this segment (linear ramp)
        progress = (i + 0.5) / num_segments  # midpoint of segment
        rate = start_rate + (end_rate - start_rate) * progress
        
        # Get segment
        start_sample = i * segment_samples
        end_sample = min(start_sample + segment_samples, len(audio))
        segment = audio[start_sample:end_sample]
        
        if len(segment) > 0:
            # Time stretch the segment
            stretched = librosa.effects.time_stretch(segment, rate=rate)
            output_segments.append(stretched)
    
    # Concatenate all segments
    output = np.concatenate(output_segments)
    
    # Trim or pad to exact duration
    target_samples = int(duration * sr)
    if len(output) > target_samples:
        output = output[:target_samples]
    elif len(output) < target_samples:
        output = np.pad(output, (0, target_samples - len(output)), mode='constant')
    
    return output


def extract_middle_clip(input_path, duration=60):
    """Extract the middle 60 seconds from an audio file."""
    print(f"Loading {input_path}...")
    y, sr = librosa.load(input_path, sr=44100, mono=True)
    file_duration = len(y) / sr
    
    print(f"Source file duration: {file_duration:.1f}s")
    
    if file_duration < duration:
        raise ValueError(f"File duration ({file_duration:.1f}s) is less than requested ({duration}s)")
    
    start_sec = (file_duration - duration) / 2
    end_sec = start_sec + duration
    
    print(f"Extracting middle {duration}s from {start_sec:.1f}s to {end_sec:.1f}s")
    
    start_sample = int(start_sec * sr)
    end_sample = start_sample + int(duration * sr)
    clip = y[start_sample:end_sample]
    
    print(f"Extracted clip: {len(clip)/sr:.1f}s ({len(clip)} samples)")
    
    return clip, sr


def generate_tempo_variants(input_path, output_base):
    """Generate original, increasing, and decreasing tempo variants."""
    clip, sr = extract_middle_clip(input_path, duration=60)
    
    # Original
    original_path = f"{output_base}_original.wav"
    print(f"Writing {original_path}...")
    sf.write(original_path, clip, sr)
    print(f"  Created: {original_path} ({len(clip)/sr:.1f}s)")
    
    # Increasing (1.0x -> 1.3x)
    increasing_path = f"{output_base}_increasing.wav"
    print(f"Generating {increasing_path} (1.0x -> 1.3x)...")
    increasing = apply_tempo_ramp(clip, 1.0, 1.3, sr)
    sf.write(increasing_path, increasing, sr)
    print(f"  Created: {increasing_path} ({len(increasing)/sr:.1f}s)")
    
    # Decreasing (1.0x -> 0.7x)
    decreasing_path = f"{output_base}_decreasing.wav"
    print(f"Generating {decreasing_path} (1.0x -> 0.7x)...")
    decreasing = apply_tempo_ramp(clip, 1.0, 0.7, sr)
    sf.write(decreasing_path, decreasing, sr)
    print(f"  Created: {decreasing_path} ({len(decreasing)/sr:.1f}s)")
    
    print("\nDone! Generated 3 files:")
    print(f"  - {original_path}")
    print(f"  - {increasing_path}")
    print(f"  - {decreasing_path}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_base = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found")
        sys.exit(1)
    
    generate_tempo_variants(input_path, output_base)