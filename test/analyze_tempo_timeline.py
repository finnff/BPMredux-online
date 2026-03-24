#!/usr/bin/env python3
"""
Analyze BPM timelines for test files using librosa.

Generates ground truth BPM timelines for E2E stability testing.
Analyzes all test files and outputs JSON with per-second BPM data.

Usage:
  python3.12 test/analyze_tempo_timeline.py
"""

import os
import json
import numpy as np
import librosa
import soundfile as sf


def analyze_bpm_timeline(audio_path, window_size=4.0, hop_size=2.0):
    """
    Analyze BPM over time using librosa's beat tracking on sliding windows.
    """
    print(f"  Analyzing {os.path.basename(audio_path)}...")
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = len(y) / sr
    
    # Create time points for analysis
    time_points = np.arange(0, duration, hop_size)
    bpm_timeline = []
    
    for t in time_points:
        # Define window boundaries
        start_sample = max(0, int((t - window_size/2) * sr))
        end_sample = min(len(y), int((t + window_size/2) * sr))
        
        if end_sample - start_sample < int(0.5 * sr):
            continue
        
        window = y[start_sample:end_sample]
        
        # Estimate tempo for this window
        try:
            tempo, beats = librosa.beat.beat_track(y=window, sr=sr)
            tempo_val = float(tempo) if hasattr(tempo, '__float__') else tempo[0]
            
            if 60 <= tempo_val <= 240:
                bpm_timeline.append({
                    "time": round(t, 2),
                    "bpm": round(tempo_val, 1)
                })
        except Exception as e:
            pass
    
    # If we didn't get enough data, use single estimate
    if len(bpm_timeline) < 2:
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo_val = float(tempo) if hasattr(tempo, '__float__') else tempo[0]
            bpm_timeline = [
                {"time": 0, "bpm": round(tempo_val, 1)},
                {"time": duration, "bpm": round(tempo_val, 1)}
            ]
        except:
            bpm_timeline = [{"time": 0, "bpm": 0}]
    
    return bpm_timeline


def classify_pattern(bpm_timeline, filename):
    """Classify the BPM pattern from the timeline based on filename."""
    if "increasing" in filename:
        return "increasing"
    elif "decreasing" in filename:
        return "decreasing"
    else:
        return "constant"


def analyze_all_tracks(fixtures_dir="test/e2e/fixtures"):
    """Analyze all test tracks and generate timeline JSON."""
    
    # Find all test files (our new naming scheme)
    test_files = []
    for f in os.listdir(fixtures_dir):
        if f.endswith(".wav") and not f.endswith("_variant.wav"):
            test_files.append(f)
    
    test_files.sort()
    
    print(f"Found {len(test_files)} test files: {', '.join(test_files)}\n")
    
    results = {}
    
    for filename in test_files:
        filepath = os.path.join(fixtures_dir, filename)
        base_name = filename.replace(".wav", "")
        
        print(f"Processing: {filename}")
        
        # Get base BPM from file (or estimate)
        y, sr = librosa.load(filepath, sr=None, mono=True)
        try:
            base_tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            base_bpm = round(float(base_tempo) if hasattr(base_tempo, '__float__') else base_tempo[0], 1)
        except:
            base_bpm = 136
        
        # Analyze timeline
        timeline = analyze_bpm_timeline(filepath)
        
        # Classify pattern based on filename
        pattern = classify_pattern(timeline, base_name)
        
        # Build result
        result = {
            "baseBpm": base_bpm,
            "pattern": pattern,
            "timeline": timeline
        }
        
        # Add pattern-specific fields
        if pattern == "increasing":
            start_bpm = timeline[0]["bpm"] if timeline else base_bpm
            end_bpm = timeline[-1]["bpm"] if timeline else base_bpm
            result["startRate"] = round(start_bpm / base_bpm, 2)
            result["endRate"] = round(end_bpm / base_bpm, 2)
        elif pattern == "decreasing":
            start_bpm = timeline[0]["bpm"] if timeline else base_bpm
            end_bpm = timeline[-1]["bpm"] if timeline else base_bpm
            result["startRate"] = round(start_bpm / base_bpm, 2)
            result["endRate"] = round(end_bpm / base_bpm, 2)
        
        results[base_name] = result
        
        # Print summary
        print(f"  Base BPM: {base_bpm}")
        print(f"  Pattern: {pattern}")
        print(f"  Timeline points: {len(timeline)}")
        if timeline:
            print(f"  Start: {timeline[0]['bpm']} BPM, End: {timeline[-1]['bpm']} BPM")
        print()
    
    # Write JSON output
    output_path = os.path.join(fixtures_dir, "bpm_timelines.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Written: {output_path}")
    return results


def print_summary(results):
    """Print a human-readable summary of the analysis."""
    print("\n" + "="*70)
    print("BPM TIMELINE ANALYSIS SUMMARY")
    print("="*70)
    print(f"{'Track':<25} {'Base BPM':>10} {'Pattern':>12} {'Start':>8} {'End':>8}")
    print("-"*70)
    
    for name, data in sorted(results.items()):
        timeline = data.get("timeline", [])
        start_bpm = timeline[0]["bpm"] if timeline else data["baseBpm"]
        end_bpm = timeline[-1]["bpm"] if timeline else data["baseBpm"]
        
        print(f"{name:<25} {data['baseBpm']:>10.1f} {data['pattern']:>12} "
              f"{start_bpm:>8.1f} {end_bpm:>8.1f}")
    
    print("="*70)


if __name__ == "__main__":
    print("Analyzing BPM timelines for stability test files...\n")
    results = analyze_all_tracks()
    print_summary(results)