/**
 * End-to-end stability slider tests
 * Tests BPM detection with different stability settings on tempo-varying tracks.
 *
 * Run: node --experimental-vm-modules test/e2e/test-e2e-stability.mjs
 */

import { describe, it, after } from 'node:test';
import assert from 'node:assert';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { FFTProcessor } from '../../src/audio/fft-processor.js';
import { BandFilter } from '../../src/audio/band-filter.js';
import { OnsetDetector } from '../../src/audio/onset-detector.js';
import { TempoEstimator } from '../../src/audio/tempo-estimator.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Constants matching the production pipeline
const SAMPLE_RATE = 44100;
const FFT_SIZE = 4096;
const HOP_SIZE = 1024;
const ODF_INTERVAL = SAMPLE_RATE / 100; // 441 samples between ODF ticks
const AMPLITUDE_THRESHOLD = 0.05;

/**
 * Parse a 16-bit mono PCM WAV file.
 */
function readWavAsFloat32(filePath) {
  const buf = readFileSync(filePath);
  const headerSize = 44;
  const sampleCount = (buf.length - headerSize) / 2;
  const float32 = new Float32Array(sampleCount);

  for (let i = 0; i < sampleCount; i++) {
    const int16 = buf.readInt16LE(headerSize + i * 2);
    float32[i] = int16 / 32768.0;
  }

  return float32;
}

/**
 * Stability presets matching the slider values
 */
const STABILITY_PRESETS = {
  responsive: 0,   // STAB=0 - most responsive
  balanced: 50,    // STAB=50 - balanced (default)
  stable: 100      // STAB=100 - most stable
};

/**
 * Ground truth BPM timelines loaded from JSON
 */
let bpmTimelines = {};

/**
 * Interpolate BPM at a given time from the timeline
 */
function getGroundTruthBpm(timeline, time) {
  if (!timeline || timeline.length === 0) return null;
  if (time <= timeline[0].time) return timeline[0].bpm;
  if (time >= timeline[timeline.length - 1].time) return timeline[timeline.length - 1].bpm;
  
  // Find surrounding points
  for (let i = 0; i < timeline.length - 1; i++) {
    if (time >= timeline[i].time && time <= timeline[i + 1].time) {
      const t0 = timeline[i].time;
      const t1 = timeline[i + 1].time;
      const bpm0 = timeline[i].bpm;
      const bpm1 = timeline[i + 1].bpm;
      
      // Linear interpolation
      const ratio = (time - t0) / (t1 - t0);
      return bpm0 + ratio * (bpm1 - bpm0);
    }
  }
  
  return timeline[timeline.length - 1].bpm;
}

/**
 * Process a WAV file through the DSP pipeline with given stability setting.
 */
function processTrackWithStability(samples, stabilityLevel, trackName) {
  const fftProcessor = new FFTProcessor(FFT_SIZE);
  const bandFilter = new BandFilter();
  const onsetDetector = new OnsetDetector();
  const tempoEstimator = new TempoEstimator();

  const stabilityValue = stabilityLevel / 100;
  tempoEstimator.setStability(stabilityValue);
  onsetDetector.setStability(stabilityValue);

  tempoEstimator.bpmRangeMin = 60;
  tempoEstimator.bpmRangeMax = 240;

  const ringBuffer = new Float32Array(FFT_SIZE);
  let ringWritePos = 0;
  let samplesUntilEmit = FFT_SIZE;

  let odfSamplesProcessed = 0;
  let odfAccumulator = 0;
  let lastOnset = false;

  const bpmReadings = [];
  let finalBpm = 0;
  let finalConfidence = 0;
  let totalFrames = 0;
  let onsetCount = 0;

  for (let i = 0; i < samples.length; i++) {
    ringBuffer[ringWritePos] = samples[i];
    ringWritePos = (ringWritePos + 1) % FFT_SIZE;
    samplesUntilEmit--;

    if (samplesUntilEmit <= 0) {
      const frame = new Float32Array(FFT_SIZE);
      for (let j = 0; j < FFT_SIZE; j++) {
        frame[j] = ringBuffer[(ringWritePos + j) % FFT_SIZE];
      }

      const magnitudes = fftProcessor.process(frame);
      const bandEnergy = bandFilter.filter(magnitudes);

      let sumSq = 0;
      for (let k = 0; k < frame.length; k++) {
        sumSq += frame[k] * frame[k];
      }
      const rmsAmplitude = Math.sqrt(sumSq / frame.length);

      const timeMs = (i / SAMPLE_RATE) * 1000;

      if (rmsAmplitude < AMPLITUDE_THRESHOLD) {
        lastOnset = false;
      } else {
        lastOnset = onsetDetector.process(magnitudes, bandEnergy, timeMs);
      }

      if (lastOnset) onsetCount++;

      odfSamplesProcessed += HOP_SIZE;
      while (odfAccumulator + ODF_INTERVAL <= odfSamplesProcessed) {
        odfAccumulator += ODF_INTERVAL;
        const result = tempoEstimator.addOnsetSample(lastOnset);
        if (result) {
          bpmReadings.push({
            time: i / SAMPLE_RATE,
            bpm: result.bpm,
            confidence: result.confidence
          });
          finalBpm = result.bpm;
          finalConfidence = result.confidence;
        }
      }

      samplesUntilEmit = HOP_SIZE;
      totalFrames++;
    }
  }

  return {
    finalBpm,
    finalConfidence,
    totalFrames,
    onsetCount,
    bpmReadings,
    jitter: calculateJitter(bpmReadings),
    responseTime: calculateResponseTime(bpmReadings),
    mae: calculateMAE(bpmReadings, trackName),
    trackingError: calculateTrackingError(bpmReadings, trackName)
  };
}

function calculateJitter(readings) {
  if (readings.length < 2) return 0;
  const stableReadings = readings.filter(r => r.time > 10);
  if (stableReadings.length < 2) return 0;

  const mean = stableReadings.reduce((sum, r) => sum + r.bpm, 0) / stableReadings.length;
  const variance = stableReadings.reduce((sum, r) => sum + (r.bpm - mean) ** 2, 0) / stableReadings.length;
  return Math.sqrt(variance);
}

function calculateResponseTime(readings) {
  if (readings.length < 2) return null;
  const firstValid = readings.find(r => r.confidence > 0.1);
  return firstValid ? firstValid.time : null;
}

function calculateMAE(readings, trackName) {
  if (readings.length === 0) return 0;
  
  const timeline = bpmTimelines[trackName]?.timeline;
  if (!timeline) return 0;
  
  let totalError = 0;
  let count = 0;
  
  for (const reading of readings) {
    const groundTruth = getGroundTruthBpm(timeline, reading.time);
    if (groundTruth !== null) {
      totalError += Math.abs(reading.bpm - groundTruth);
      count++;
    }
  }
  
  return count > 0 ? totalError / count : 0;
}

function calculateTrackingError(readings, trackName) {
  if (readings.length < 2) return 0;
  
  const timeline = bpmTimelines[trackName]?.timeline;
  if (!timeline) return 0;
  
  const gtStart = getGroundTruthBpm(timeline, 0);
  const gtEnd = getGroundTruthBpm(timeline, 60);
  if (gtStart === null || gtEnd === null) return 0;
  
  const expectedSlope = (gtEnd - gtStart) / 60;
  
  const validReadings = readings.filter(r => r.confidence > 0.1);
  if (validReadings.length < 2) return 0;
  
  const firstReading = validReadings[0];
  const lastReading = validReadings[validReadings.length - 1];
  const actualSlope = (lastReading.bpm - firstReading.bpm) / (lastReading.time - firstReading.time);
  
  if (Math.abs(expectedSlope) < 0.1) return 0;
  
  return Math.abs(actualSlope - expectedSlope) / Math.abs(expectedSlope);
}

function formatNum(num, decimals = 1) {
  return num.toFixed(decimals);
}

// Load ground truth BPM timelines
const timelinesPath = resolve(__dirname, 'fixtures/bpm_timelines.json');
try {
  const timelinesData = readFileSync(timelinesPath, 'utf-8');
  bpmTimelines = JSON.parse(timelinesData);
  console.log(`Loaded ground truth timelines from ${timelinesPath}\n`);
} catch (err) {
  console.warn(`Warning: Could not load BPM timelines: ${err.message}`);
}

/**
 * Test tracks - 9 files (3 genres x 3 variants)
 */
const TEST_TRACKS = [
  { name: 'techno_original', file: 'techno_original.wav', baseBpm: 136, pattern: 'constant' },
  { name: 'techno_increasing', file: 'techno_increasing.wav', baseBpm: 136, pattern: 'increasing' },
  { name: 'techno_decreasing', file: 'techno_decreasing.wav', baseBpm: 136, pattern: 'decreasing' },
  { name: 'trance_original', file: 'trance_original.wav', baseBpm: 140, pattern: 'constant' },
  { name: 'trance_increasing', file: 'trance_increasing.wav', baseBpm: 140, pattern: 'increasing' },
  { name: 'trance_decreasing', file: 'trance_decreasing.wav', baseBpm: 140, pattern: 'decreasing' },
  { name: 'dnb_original', file: 'dnb_original.wav', baseBpm: 175, pattern: 'constant' },
  { name: 'dnb_increasing', file: 'dnb_increasing.wav', baseBpm: 175, pattern: 'increasing' },
  { name: 'dnb_decreasing', file: 'dnb_decreasing.wav', baseBpm: 175, pattern: 'decreasing' }
];

const results = [];

describe('Stability Slider E2E Tests', () => {
  TEST_TRACKS.forEach(track => {
    Object.entries(STABILITY_PRESETS).forEach(([presetName, presetValue]) => {
      it(`STAB=${presetValue} (${presetName}) - ${track.name}`, () => {
        const wavPath = resolve(__dirname, 'fixtures', track.file);
        const samples = readWavAsFloat32(wavPath);
        const durationSec = samples.length / SAMPLE_RATE;

        console.log(`\n  Testing: ${track.file} (${durationSec.toFixed(1)}s) at STAB=${presetValue}`);

        const result = processTrackWithStability(samples, presetValue, track.name);

        console.log(`  Results: BPM=${formatNum(result.finalBpm)} conf=${formatNum(result.finalConfidence, 2)}`);
        console.log(`  MAE: ${formatNum(result.mae, 1)} BPM | Jitter: ${formatNum(result.jitter, 1)}`);
        console.log(`  Tracking error: ${formatNum(result.trackingError, 2)} | Readings: ${result.bpmReadings.length}`);

        assert.ok(result.bpmReadings.length > 10, `Should have >10 readings, got ${result.bpmReadings.length}`);

        results.push({
          trackName: track.name,
          stabilityLevel: presetValue,
          presetName: presetName,
          mae: result.mae,
          responseLag: result.responseTime,
          trackingError: result.trackingError,
          jitter: result.jitter,
          finalBpm: result.finalBpm
        });
      });
    });
  });
  
  after(() => {
    printSummary(results);
  });
});

function printSummary(results) {
  console.log('\n' + '='.repeat(78));
  console.log('  STABILITY vs ACCURACY ANALYSIS');
  console.log('='.repeat(78));
  console.log('  Track'.padEnd(20), '│ STAB │ MAE   │ Lag     │ Jitter  │ Tracking Err');
  console.log('-'.repeat(78));
  
  const sorted = results.sort((a, b) => {
    if (a.trackName !== b.trackName) return a.trackName.localeCompare(b.trackName);
    return a.stabilityLevel - b.stabilityLevel;
  });
  
  for (const r of sorted) {
    const lagStr = r.responseLag !== null ? `${formatNum(r.responseLag, 1)}s` : 'N/A';
    console.log(
      r.trackName.padEnd(20), '│',
      formatNum(r.stabilityLevel, 0).padStart(4), '  │',
      formatNum(r.mae, 1).padStart(5), '  │',
      lagStr.padEnd(7), '│',
      formatNum(r.jitter, 1).padStart(7), '  │',
      formatNum(r.trackingError, 2)
    );
  }
  
  console.log('='.repeat(78));
  console.log('');
}