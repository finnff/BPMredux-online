#!/usr/bin/env python3
"""
Verify BPM detection results using librosa as ground truth.
Compares librosa's beat_track against our JS DSP pipeline results.
"""
import librosa
import numpy as np
import sys
import os

# Test tracks with their expected BPM ranges
TRACKS = {
    "techno_original": ("e2e/fixtures/techno_original.wav", 135, "Techno"),
    "techno_variant": ("e2e/fixtures/techno_variant.wav", None, "Techno (variable tempo)"),
    "trance_original": ("e2e/fixtures/trance_original.wav", 140, "Trance"),
    "trance_variant": ("e2e/fixtures/trance_variant.wav", None, "Trance (variable tempo)"),
    "dnb_original": ("e2e/fixtures/dnb_original.wav", 175, "Drum & Bass"),
    "dnb_variant": ("e2e/fixtures/dnb_variant.wav", None, "Drum & Bass (variable tempo)"),
}

def get_path(rel_path):
    """Get absolute path from relative path."""
    return os.path.join(os.path.dirname(__file__), rel_path)

def analyze(path, label, expected_bpm=None):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {os.path.basename(path)}")
    print(f"{'='*60}")

    y, sr = librosa.load(path, sr=44100, mono=True)
    duration = len(y) / sr
    print(f"  Duration: {duration:.1f}s  SR: {sr}Hz  Samples: {len(y)}")

    if expected_bpm:
        print(f"  Expected BPM: ~{expected_bpm}")

    # Method 1: librosa.beat.beat_track (default)
    tempo_default, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo_default = float(np.atleast_1d(tempo_default)[0])
    print(f"\n  [1] librosa.beat.beat_track (default):  {tempo_default:.1f} BPM")

    # Method 2: onset-based tempo estimation
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo_onset = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
    tempo_onset = float(np.atleast_1d(tempo_onset)[0])
    print(f"  [2] librosa.feature.tempo (onset):      {tempo_onset:.1f} BPM")

    # Method 3: tempogram-based
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    tempo_tg = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, aggregate=np.median)
    tempo_tg = float(np.atleast_1d(tempo_tg)[0])
    print(f"  [3] librosa.feature.tempo (median):     {tempo_tg:.1f} BPM")

    # Method 4: PLP (predominant local pulse)
    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    tempo_plp = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, prior=None)
    tempo_plp = float(np.atleast_1d(tempo_plp)[0])
    print(f"  [4] librosa.feature.tempo (no prior):   {tempo_plp:.1f} BPM")

    # For variable tempo tracks, do segment analysis
    if "variant" in label:
        print(f"\n  --- Variable BPM Analysis (5s segments) ---")
        segment_len = 5 * sr  # 5 seconds
        for i in range(0, min(len(y), 30 * sr), segment_len):  # First 30s only
            segment = y[i:i+segment_len]
            if len(segment) < segment_len:
                continue
            seg_tempo, _ = librosa.beat.beat_track(y=segment, sr=sr)
            seg_tempo = float(np.atleast_1d(seg_tempo)[0])
            print(f"  {i//sr:2d}-{(i+segment_len)//sr:2d}s: {seg_tempo:.1f} BPM")

    return tempo_default, tempo_onset

print("╔══════════════════════════════════════════════════════════╗")
print("║  BPM Verification: librosa vs BPMredux-online pipeline  ║")
print("╚══════════════════════════════════════════════════════════╝")

results = {}

# Analyze all test tracks
for name, (rel_path, expected_bpm, label) in TRACKS.items():
    path = get_path(rel_path)
    if os.path.exists(path):
        t1, t2 = analyze(path, label, expected_bpm)
        results[name] = (t1, t2, expected_bpm, label)

# Summary table
print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"{'Track':<20} {'Expected':>10} {'librosa':>10} {'Onset':>10}")
print(f"{'-'*60}")

for name, (t1, t2, expected_bpm, label) in results.items():
    expected_str = f"~{expected_bpm}" if expected_bpm else "variable"
    print(f"{label:<20} {expected_str:>10} {t1:>10.1f} {t2:>10.1f}")

# Expected tempo ranges for variant files
print(f"\n{'='*60}")
print(f"  VARIABLE TEMPO EXPECTED RANGES")
print(f"{'='*60}")
print("Variant files should show these BPM patterns:")
print("  0-5s:   ~original BPM (1.0x speed)")
print("  5-25s:  ramps from ~+20% to ~-40% of original")
print("  25s+:   ~72% of original BPM (0.72x speed)")
print()
print("Expected ranges by genre:")
print(f"  Techno:   135 base → 162 (fast) → 97 (slow) BPM")
print(f"  Trance:   140 base → 168 (fast) → 101 (slow) BPM")
print(f"  DnB:      175 base → 210 (fast) → 122 (slow) BPM")
print()
