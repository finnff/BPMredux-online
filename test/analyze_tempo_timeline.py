#!/usr/bin/env python3
"""
Analyze BPM timelines for E2E test fixtures using librosa as ground truth.

For each test fixture file, computes instantaneous BPM at 2-second intervals
using a sliding window, then outputs bpm_timelines.json for the E2E test suite.

Usage:
  python3 test/analyze_tempo_timeline.py
"""

import os
import json
import numpy as np
import librosa

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'e2e', 'fixtures')
OUTPUT_JSON = os.path.join(FIXTURES_DIR, 'bpm_timelines.json')

# Window size for local BPM estimation (in seconds)
WINDOW_SEC = 8.0
# Step between analysis points (in seconds)
STEP_SEC = 2.0

# Expected patterns for each file (hardcoded ground truth)
FILE_META = {
    'techno_original':    {'pattern': 'constant', 'baseBpm': None},
    'techno_increasing':  {'pattern': 'increasing', 'startRate': 1.0, 'endRate': 1.3},
    'techno_decreasing':  {'pattern': 'decreasing', 'startRate': 1.0, 'endRate': 0.7},
    'trance_original':    {'pattern': 'constant', 'baseBpm': None},
    'trance_increasing':  {'pattern': 'increasing', 'startRate': 1.0, 'endRate': 1.3},
    'trance_decreasing':  {'pattern': 'decreasing', 'startRate': 1.0, 'endRate': 0.7},
    'dnb_original':       {'pattern': 'constant', 'baseBpm': None},
    'dnb_increasing':     {'pattern': 'increasing', 'startRate': 1.0, 'endRate': 1.3},
    'dnb_decreasing':     {'pattern': 'decreasing', 'startRate': 1.0, 'endRate': 0.7},
}


def estimate_local_bpm(y, sr, center_sec, window_sec=WINDOW_SEC):
    """Estimate BPM at a given time position using a local window."""
    half = window_sec / 2
    start = max(0, int((center_sec - half) * sr))
    end = min(len(y), int((center_sec + half) * sr))
    segment = y[start:end]

    if len(segment) < sr * 2:
        return None

    onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
    return float(np.atleast_1d(tempo)[0])


def analyze_file(name, path):
    """Analyze a single file and return timeline data."""
    print(f"\n  {name}")
    y, sr = librosa.load(path, sr=44100, mono=True)
    duration = len(y) / sr
    print(f"    Duration: {duration:.1f}s")

    # Global BPM estimate
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    global_bpm = float(np.atleast_1d(
        librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
    )[0])
    print(f"    Global BPM: {global_bpm:.1f}")

    # Build timeline at STEP_SEC intervals
    timeline = []
    t = 0.0
    while t <= duration:
        bpm = estimate_local_bpm(y, sr, t)
        if bpm is not None:
            timeline.append({'time': round(t, 1), 'bpm': round(bpm, 2)})
        t += STEP_SEC

    meta = FILE_META.get(name, {'pattern': 'unknown'})
    result = {
        'baseBpm': round(global_bpm),
        'pattern': meta['pattern'],
        'timeline': timeline,
    }

    if 'startRate' in meta:
        result['startRate'] = meta['startRate']
        result['endRate'] = meta['endRate']

    # Print a few sample points
    for entry in timeline[::5]:
        print(f"    t={entry['time']:5.1f}s -> {entry['bpm']:.1f} BPM")

    return result


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  BPM Timeline Analysis (librosa ground truth)        ║")
    print("╚══════════════════════════════════════════════════════╝")

    results = {}
    missing = []

    for name in FILE_META:
        path = os.path.join(FIXTURES_DIR, f'{name}.wav')
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            missing.append(name)
            continue
        results[name] = analyze_file(name, path)

    if missing:
        print(f"\nWARNING: {len(missing)} files missing, skipped: {missing}")

    # Write JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote: {OUTPUT_JSON}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"{'File':<25} {'Pattern':<12} {'Base BPM':>8}  {'Start':>7}  {'End':>7}")
    print(f"{'-'*60}")
    for name, data in results.items():
        timeline = data['timeline']
        start_bpm = timeline[0]['bpm'] if timeline else 0
        end_bpm = timeline[-1]['bpm'] if timeline else 0
        print(f"{name:<25} {data['pattern']:<12} {data['baseBpm']:>8}  {start_bpm:>7.1f}  {end_bpm:>7.1f}")


if __name__ == '__main__':
    main()