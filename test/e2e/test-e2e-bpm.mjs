/**
 * End-to-end BPM detection test
 * Reads a WAV file, runs it through the full DSP pipeline, verifies BPM output.
 *
 * Run: node --experimental-vm-modules test/e2e/test-e2e-bpm.mjs
 */

import { describe, it } from 'node:test';
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
const AMPLITUDE_THRESHOLD = 0.15;

/**
 * Parse a 16-bit mono PCM WAV file.
 * Skips the 44-byte header and converts Int16LE samples to Float32 in [-1, 1].
 */
function readWavAsFloat32(filePath) {
  const buf = readFileSync(filePath);

  // Skip 44-byte WAV header, read as Int16LE
  const headerSize = 44;
  const sampleCount = (buf.length - headerSize) / 2;
  const float32 = new Float32Array(sampleCount);

  for (let i = 0; i < sampleCount; i++) {
    const int16 = buf.readInt16LE(headerSize + i * 2);
    float32[i] = int16 / 32768.0;
  }

  return float32;
}

describe('E2E BPM detection on Darude - Sandstorm (30s WAV)', () => {
  it('should detect BPM between 130-145 with confidence > 0', () => {
    // --- Load audio ---
    const wavPath = resolve(__dirname, '..', 'fixtures', 'sandstorm_30s_44100.wav');
    const samples = readWavAsFloat32(wavPath);
    const durationSec = samples.length / SAMPLE_RATE;
    console.log(`Loaded WAV: ${samples.length} samples, ${durationSec.toFixed(1)}s`);

    // --- Initialize DSP pipeline ---
    const fftProcessor = new FFTProcessor(FFT_SIZE);
    const bandFilter = new BandFilter();
    const onsetDetector = new OnsetDetector();
    const tempoEstimator = new TempoEstimator();

    // Wide BPM range
    tempoEstimator.bpmRangeMin = 60;
    tempoEstimator.bpmRangeMax = 240;

    // --- Ring buffer state (mirrors AudioCapture) ---
    const ringBuffer = new Float32Array(FFT_SIZE);
    let ringWritePos = 0;
    let samplesUntilEmit = FFT_SIZE; // first frame needs FFT_SIZE samples

    // --- ODF timing state (mirrors main.js) ---
    let odfSamplesProcessed = 0;
    let odfAccumulator = 0;
    let lastOnset = false;

    // --- Result tracking ---
    let bpm = 0;
    let confidence = 0;
    let totalFrames = 0;
    let onsetCount = 0;
    let nextProgressSample = 5 * SAMPLE_RATE; // print every 5s

    // --- Process all samples through ring buffer ---
    for (let i = 0; i < samples.length; i++) {
      ringBuffer[ringWritePos] = samples[i];
      ringWritePos = (ringWritePos + 1) % FFT_SIZE;
      samplesUntilEmit--;

      if (samplesUntilEmit <= 0) {
        // Emit a frame: read FFT_SIZE samples starting from current write position
        const frame = new Float32Array(FFT_SIZE);
        for (let j = 0; j < FFT_SIZE; j++) {
          frame[j] = ringBuffer[(ringWritePos + j) % FFT_SIZE];
        }

        // --- Process frame (same as main.js processFrame) ---
        const magnitudes = fftProcessor.process(frame);
        const bandEnergy = bandFilter.filter(magnitudes);

        // RMS amplitude from time-domain frame
        let sumSq = 0;
        for (let k = 0; k < frame.length; k++) {
          sumSq += frame[k] * frame[k];
        }
        const rmsAmplitude = Math.sqrt(sumSq / frame.length);

        // Time in ms based on sample position
        const timeMs = (i / SAMPLE_RATE) * 1000;

        // Amplitude gate + onset detection
        if (rmsAmplitude < AMPLITUDE_THRESHOLD) {
          lastOnset = false;
        } else {
          lastOnset = onsetDetector.process(magnitudes, bandEnergy, timeMs);
        }

        if (lastOnset) onsetCount++;
        totalFrames++;

        // Feed ODF at 100 Hz (same logic as main.js)
        odfSamplesProcessed += HOP_SIZE;
        while (odfAccumulator + ODF_INTERVAL <= odfSamplesProcessed) {
          odfAccumulator += ODF_INTERVAL;
          const result = tempoEstimator.addOnsetSample(lastOnset);
          if (result) {
            bpm = result.bpm;
            confidence = result.confidence;
          }
        }

        // Next emission after HOP_SIZE samples
        samplesUntilEmit = HOP_SIZE;
      }

      // Progress logging
      if (i >= nextProgressSample) {
        const sec = (i / SAMPLE_RATE).toFixed(0);
        console.log(`  ${sec}s processed | frames=${totalFrames} onsets=${onsetCount} BPM=${bpm.toFixed(1)} conf=${confidence.toFixed(2)}`);
        nextProgressSample += 5 * SAMPLE_RATE;
      }
    }

    // Final status
    console.log(`\nDone: ${totalFrames} frames, ${onsetCount} onsets`);
    console.log(`Final BPM: ${bpm.toFixed(2)}, Confidence: ${confidence.toFixed(3)}`);

    // --- Assertions ---
    assert.ok(bpm >= 130, `BPM ${bpm.toFixed(2)} should be >= 130`);
    assert.ok(bpm <= 145, `BPM ${bpm.toFixed(2)} should be <= 145`);
    assert.ok(confidence > 0, `Confidence ${confidence} should be > 0`);

    console.log('PASS: BPM is in expected range [130, 145] and confidence > 0');
  });
});
