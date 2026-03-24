#!/usr/bin/env python3
"""
Verify BPM detection results using librosa as ground truth.
Compares librosa's beat_track against our JS DSP pipeline results.
"""
import librosa
import numpy as np
import sys
import os

WAV_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sandstorm_30s_44100.wav")
FULL_WAV = os.path.join(os.path.dirname(__file__), "fixtures", "test_track.wav")

# Our E2E pipeline result
OUR_BPM = 136.4

def analyze(path, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {path}")
    print(f"{'='*60}")

    y, sr = librosa.load(path, sr=44100, mono=True)
    duration = len(y) / sr
    print(f"  Duration: {duration:.1f}s  SR: {sr}Hz  Samples: {len(y)}")

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

    return tempo_default, tempo_onset

print("╔══════════════════════════════════════════════════════════╗")
print("║  BPM Verification: librosa vs BPMredux-online pipeline  ║")
print("╚══════════════════════════════════════════════════════════╝")

results = {}

# Analyze the 30s clip (same as our E2E test)
if os.path.exists(WAV_PATH):
    t1, t2 = analyze(WAV_PATH, "30s clip (same as E2E test)")
    results["30s_clip"] = (t1, t2)

# Analyze full track
if os.path.exists(FULL_WAV):
    t1, t2 = analyze(FULL_WAV, "Full track")
    results["full"] = (t1, t2)

# Comparison
print(f"\n{'='*60}")
print(f"  COMPARISON")
print(f"{'='*60}")
print(f"  Known BPM (Darude - Sandstorm):    136 BPM")
print(f"  Our pipeline (E2E test result):     {OUR_BPM:.1f} BPM")

if "30s_clip" in results:
    t1, t2 = results["30s_clip"]
    print(f"  librosa beat_track (30s clip):      {t1:.1f} BPM")
    print(f"  librosa onset tempo (30s clip):     {t2:.1f} BPM")
    diff = abs(OUR_BPM - t1)
    print(f"\n  Δ (ours vs librosa beat_track):     {diff:.1f} BPM")
    if diff < 5:
        print(f"  ✅ EXCELLENT — within 5 BPM of librosa")
    elif diff < 10:
        print(f"  ✓ GOOD — within 10 BPM of librosa")
    else:
        # Check for half/double tempo relationship
        if abs(OUR_BPM - t1 * 2) < 5 or abs(OUR_BPM - t1 / 2) < 5:
            print(f"  ⚠ HARMONIC — half/double tempo relationship detected")
        else:
            print(f"  ✗ MISMATCH — >10 BPM difference")

if "full" in results:
    t1, _ = results["full"]
    print(f"\n  librosa beat_track (full track):    {t1:.1f} BPM")

print()
