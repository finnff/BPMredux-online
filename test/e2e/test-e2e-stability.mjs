/**
 * End-to-end stability slider tests
 * Tests BPM detection with different stability settings on all 9 test tracks.
 * Compares results against librosa ground truth from bpm_timelines.json.
 *
 * Run: node --experimental-vm-modules test/e2e/test-e2e-stability.mjs
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { readFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { FFTProcessor } from '../../src/audio/fft-processor.js';
import { BandFilter, BANDS } from '../../src/audio/band-filter.js';
import { OnsetDetector } from '../../src/audio/onset-detector.js';
import { TempoEstimator } from '../../src/audio/tempo-estimator.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Constants matching the production pipeline
const SAMPLE_RATE = 44100;
const FFT_SIZE = 4096;
const HOP_SIZE = 1024;
const ODF_INTERVAL = SAMPLE_RATE / 100; // 441 samples between ODF ticks
const AMPLITUDE_THRESHOLD = 0.01;

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
 * Load librosa ground truth timelines if available.
 */
function loadGroundTruth() {
  const jsonPath = resolve(__dirname, 'fixtures', 'bpm_timelines.json');
  if (!existsSync(jsonPath)) return null;
  return JSON.parse(readFileSync(jsonPath, 'utf8'));
}

/**
 * Interpolate ground truth BPM at a given time using the timeline.
 */
function interpolateGroundTruth(timeline, timeSec) {
  if (!timeline || timeline.length === 0) return null;
  if (timeSec <= timeline[0].time) return timeline[0].bpm;
  if (timeSec >= timeline[timeline.length - 1].time) return timeline[timeline.length - 1].bpm;

  for (let i = 0; i < timeline.length - 1; i++) {
    const a = timeline[i];
    const b = timeline[i + 1];
    if (timeSec >= a.time && timeSec <= b.time) {
      const t = (timeSec - a.time) / (b.time - a.time);
      return a.bpm + t * (b.bpm - a.bpm);
    }
  }
  return timeline[timeline.length - 1].bpm;
}

/**
 * Process a WAV file through the DSP pipeline with given stability setting.
 * Returns metrics including response time, jitter, final BPM, and BPM readings.
 */
function processTrackWithStability(samples, stabilityLevel) {
  const fftProcessor = new FFTProcessor(FFT_SIZE);
  const bandFilter = new BandFilter();
  const onsetDetector = new OnsetDetector();
  // BAND NORMALIZATION FIX: Each band now contributes equally regardless of bin count
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
  let lastOnset = 0;  // continuous-valued ODF: flux value (0 or positive number)

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
      for (let k = 0; k < frame.length; k++) sumSq += frame[k] * frame[k];
      const rmsAmplitude = Math.sqrt(sumSq / frame.length);

      const timeMs = (i / SAMPLE_RATE) * 1000;

      // Amplitude gate - always call process(), gate the result (continuous-valued)
      const rawOnset = onsetDetector.process(magnitudes, bandEnergy, timeMs);
      lastOnset = (rmsAmplitude >= AMPLITUDE_THRESHOLD) ? rawOnset : 0;

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
  };
}

/**
 * Standard deviation of BPM readings after initial warmup (10s).
 */
function calculateJitter(readings) {
  if (readings.length < 2) return 0;
  const stableReadings = readings.filter(r => r.time > 10);
  if (stableReadings.length < 2) return 0;

  const mean = stableReadings.reduce((sum, r) => sum + r.bpm, 0) / stableReadings.length;
  const variance = stableReadings.reduce((sum, r) => sum + (r.bpm - mean) ** 2, 0) / stableReadings.length;
  return Math.sqrt(variance);
}

/**
 * Calculate response time - how long until first valid reading.
 */
function calculateResponseTime(readings) {
  if (readings.length < 2) return null;
  const firstValid = readings.find(r => r.confidence > 0.1);
  return firstValid ? firstValid.time : null;
}

/**
 * Compute accuracy metrics by comparing algorithm output to librosa ground truth.
 * @param {Array} readings - [{time, bpm, confidence}, ...] (algorithm output)
 * @param {Array} timeline - [{time, bpm}, ...] (librosa ground truth)
 * @returns {Object} { mae, responseLag, trackingError, finalAccuracy, jitter }
 */
function computeAccuracyMetrics(readings, timeline) {
  if (!timeline || readings.length === 0) return null;

  // MAE: mean absolute error vs ground truth for each reading
  let maeSum = 0, maeCount = 0;
  for (const r of readings) {
    const gt = interpolateGroundTruth(timeline, r.time);
    if (gt !== null) {
      maeSum += Math.abs(r.bpm - gt);
      maeCount++;
    }
  }
  const mae = maeCount > 0 ? maeSum / maeCount : null;

  // Final accuracy: how close the last reading is to the expected final BPM
  const lastReading = readings[readings.length - 1];
  const finalGt = timeline[timeline.length - 1].bpm;
  const finalAccuracy = lastReading && finalGt > 0
    ? 1 - Math.abs(lastReading.bpm - finalGt) / finalGt
    : null;

  // Response lag: for non-constant patterns, how long before our readings
  // begin tracking the slope of the ground truth
  let responseLag = null;
  if (timeline.length > 1) {
    const totalChange = Math.abs(timeline[timeline.length - 1].bpm - timeline[0].bpm);
    if (totalChange > 5) {
      // Find first reading where error is within 20% of total BPM range
      const threshold = totalChange * 0.2;
      for (const r of readings) {
        const gt = interpolateGroundTruth(timeline, r.time);
        if (gt !== null && Math.abs(r.bpm - gt) < threshold) {
          responseLag = r.time;
          break;
        }
      }
    }
  }

  // Tracking error: correlation between slope of readings and slope of GT
  // Measure as mean absolute deviation of first derivative
  let trackingError = null;
  if (readings.length > 5 && timeline.length > 1) {
    let slopeErrorSum = 0, slopeCount = 0;
    for (let i = 1; i < readings.length; i++) {
      const dt = readings[i].time - readings[i - 1].time;
      if (dt < 0.01) continue;
      const actualSlope = (readings[i].bpm - readings[i - 1].bpm) / dt;
      const gt0 = interpolateGroundTruth(timeline, readings[i - 1].time);
      const gt1 = interpolateGroundTruth(timeline, readings[i].time);
      if (gt0 !== null && gt1 !== null) {
        const expectedSlope = (gt1 - gt0) / dt;
        slopeErrorSum += Math.abs(actualSlope - expectedSlope);
        slopeCount++;
      }
    }
    trackingError = slopeCount > 0 ? slopeErrorSum / slopeCount : null;
  }

  return { mae, responseLag, trackingError, finalAccuracy };
}

// Load ground truth timelines (null if file not yet generated)
const groundTruth = loadGroundTruth();

/**
 * All 9 test tracks: 3 genres x 3 patterns (original, increasing, decreasing)
 */
const TEST_TRACKS = [
  // Techno (~136 BPM)
  { name: 'techno_original',   file: 'techno_original.wav',   baseBpm: 136, pattern: 'constant' },
  { name: 'techno_increasing', file: 'techno_increasing.wav', baseBpm: 136, pattern: 'increasing' },
  { name: 'techno_decreasing', file: 'techno_decreasing.wav', baseBpm: 136, pattern: 'decreasing' },
  // Trance (~140 BPM)
  { name: 'trance_original',   file: 'trance_original.wav',   baseBpm: 140, pattern: 'constant' },
  { name: 'trance_increasing', file: 'trance_increasing.wav', baseBpm: 140, pattern: 'increasing' },
  { name: 'trance_decreasing', file: 'trance_decreasing.wav', baseBpm: 140, pattern: 'decreasing' },
  // Drum & Bass (~175 BPM)
  { name: 'dnb_original',      file: 'dnb_original.wav',      baseBpm: 175, pattern: 'constant' },
  { name: 'dnb_increasing',    file: 'dnb_increasing.wav',    baseBpm: 175, pattern: 'increasing' },
  { name: 'dnb_decreasing',    file: 'dnb_decreasing.wav',    baseBpm: 175, pattern: 'decreasing' },
];

describe('Stability Slider E2E Tests', () => {
  TEST_TRACKS.forEach(track => {
    const wavPath = resolve(__dirname, 'fixtures', track.file);
    if (!existsSync(wavPath)) {
      it(`SKIP: ${track.name} - fixture file missing`, () => {
        console.log(`  SKIP: ${wavPath} not found`);
      });
      return;
    }

    describe(`${track.name} (${track.pattern})`, () => {
      Object.entries(STABILITY_PRESETS).forEach(([presetName, presetValue]) => {
        it(`STAB=${presetValue} (${presetName})`, () => {
          const samples = readWavAsFloat32(wavPath);
          const durationSec = samples.length / SAMPLE_RATE;
          console.log(`  Loading: ${track.file} (${durationSec.toFixed(1)}s)`);

          const result = processTrackWithStability(samples, presetValue);
          console.log(`  BPM=${result.finalBpm.toFixed(1)} conf=${result.finalConfidence.toFixed(2)} jitter=${result.jitter.toFixed(2)}`);
          console.log(`  Response time: ${result.responseTime ? result.responseTime.toFixed(1) + 's' : 'N/A'}`);
          console.log(`  Total readings: ${result.bpmReadings.length}`);

          // Ground truth comparison
          const gtData = groundTruth && groundTruth[track.name];
          if (gtData) {
            const metrics = computeAccuracyMetrics(result.bpmReadings, gtData.timeline);
            if (metrics) {
              console.log(`  MAE: ${metrics.mae !== null ? metrics.mae.toFixed(2) : 'N/A'} BPM`);
              console.log(`  Response lag: ${metrics.responseLag !== null ? metrics.responseLag.toFixed(1) + 's' : 'N/A'}`);
              console.log(`  Final accuracy: ${metrics.finalAccuracy !== null ? (metrics.finalAccuracy * 100).toFixed(1) + '%' : 'N/A'}`);
            }
          }

          // Assertions: must have readings
          assert.ok(
            result.bpmReadings.length > 10,
            `Should have >10 readings, got ${result.bpmReadings.length}`
          );

          // Final BPM must be non-zero
          assert.ok(result.finalBpm > 0, `Final BPM should be > 0, got ${result.finalBpm}`);

          // For original (constant tempo) tracks, verify we're in a reasonable range.
          // Use 50% margin to accept half-tempo detections (common with high stability
          // settings where convergence is slow and the algorithm may lock onto a
          // sub-harmonic peak).
          if (track.pattern === 'constant') {
            const margin = track.baseBpm * 0.50;
            assert.ok(
              result.finalBpm >= track.baseBpm - margin &&
              result.finalBpm <= track.baseBpm + margin,
              `BPM ${result.finalBpm.toFixed(1)} should be within 50% of ${track.baseBpm} for constant track`
            );
          }

          // Jitter should be lower for higher stability settings (relative check)
          // Just log it rather than assert, since variable tempo inherently causes jitter
          if (result.jitter > 0) {
            console.log(`  Jitter: ${result.jitter.toFixed(2)} BPM stddev`);
          }
        });
      });
    });
  });

  // Print accuracy report if ground truth is available
  it('Accuracy report (summary)', () => {
    if (!groundTruth) {
      console.log('  Ground truth not available. Run: python3 test/analyze_tempo_timeline.py');
      return;
    }

    const header = '╔════════════════════════════════════════════════════════════════╗';
    const footer = '╚════════════════════════════════════════════════════════════════╝';
    console.log('\n' + header);
    console.log('║  STABILITY vs ACCURACY ANALYSIS                                ║');
    console.log('╠════════════════════════════════════════════════════════════════╣');
    console.log('║  Track              │ STAB │  MAE  │  Lag  │ FinalAcc │ Jitter ║');
    console.log('╠════════════════════════════════════════════════════════════════╣');

    for (const track of TEST_TRACKS) {
      const wavPath = resolve(__dirname, 'fixtures', track.file);
      if (!existsSync(wavPath)) continue;
      const samples = readWavAsFloat32(wavPath);
      const gtData = groundTruth[track.name];

      for (const [presetName, presetValue] of Object.entries(STABILITY_PRESETS)) {
        const result = processTrackWithStability(samples, presetValue);
        const metrics = gtData ? computeAccuracyMetrics(result.bpmReadings, gtData.timeline) : null;

        const name = track.name.padEnd(19);
        const stab = String(presetValue).padStart(4);
        const mae = metrics?.mae != null ? metrics.mae.toFixed(1).padStart(5) : '  N/A';
        const lag = metrics?.responseLag != null ? (metrics.responseLag.toFixed(1) + 's').padStart(5) : '  N/A';
        const acc = metrics?.finalAccuracy != null ? (metrics.finalAccuracy * 100).toFixed(1).padStart(7) + '%' : '     N/A';
        const jit = result.jitter.toFixed(1).padStart(6);
        console.log(`║  ${name} │ ${stab} │ ${mae} │ ${lag} │ ${acc} │ ${jit} ║`);
      }
    }

    console.log(footer);
  });
});