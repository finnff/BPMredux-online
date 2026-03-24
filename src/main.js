/**
 * BPMredux-online — Main Orchestration
 * Wires audio DSP pipeline + canvas UI renderers + DOM controls.
 */

import { AudioCapture, SAMPLE_RATE, FFT_SIZE } from './audio/audio-capture.js';
import { FFTProcessor } from './audio/fft-processor.js';
import { BandFilter, BANDS } from './audio/band-filter.js';
import { OnsetDetector } from './audio/onset-detector.js';
import { TempoEstimator } from './audio/tempo-estimator.js';
import { TapProcessor } from './audio/tap-processor.js';
import { BpmGauge } from './ui/bpm-display.js';
import { Spectrogram } from './ui/spectrogram.js';

// ===== DOM References =====
const $ = id => document.getElementById(id);
const dom = {
  app:            $('app'),
  gaugeCanvas:    $('bpm-gauge'),
  spectroCanvas:  $('spectrogram'),
  tapArea:        $('tap-area'),
  tapText:        $('tap-text'),
  tapBpm:         $('tap-bpm'),
  bandSub:        $('band-sub'),
  bandMid:        $('band-mid'),
  bandHi:         $('band-hi'),
  rangeMin:       $('bpm-range-min'),
  rangeMax:       $('bpm-range-max'),
  rangeMinLabel:  $('range-min-label'),
  rangeMaxLabel:  $('range-max-label'),
  rangeTrack:     $('range-track'),
  threshold:      $('threshold-slider'),
  thresholdValue: $('threshold-value'),
  btnStart:       $('btn-start'),
  statFreq:       $('stat-freq'),
  statSig:        $('stat-sig'),
  statConf:       $('stat-conf'),
};

// ===== DSP Pipeline =====
const fftProcessor    = new FFTProcessor(FFT_SIZE);
const bandFilter      = new BandFilter(SAMPLE_RATE, FFT_SIZE);
const onsetDetector   = new OnsetDetector();
const tempoEstimator  = new TempoEstimator();
const tapProcessor    = new TapProcessor();

let audioCapture = null;

// ===== Canvas Renderers =====
const bpmGauge    = new BpmGauge(dom.gaugeCanvas);
const spectrogram = new Spectrogram(dom.spectroCanvas);

// ===== App State =====
const state = {
  running: false,
  bpm: 0,
  confidence: 0,
  isAtRangeLimit: false,
  limitSide: null,
  peakFreq: 0,
  tapBpm: 0,
  tapConfidence: 0,
  tapCount: 0,
  activeBands: new Set([BANDS.SUB, BANDS.MID, BANDS.HI]),
  rangeMin: 120,
  rangeMax: 180,
  amplitudeThreshold: 0.15,  // 0..0.5 range, 30/200 = 0.15 default
  rmsAmplitude: 0,
  magnitudes: null,
  // ODF timing: we sample the onset detection function at 100 Hz
  odfAccumulator: 0,
  odfInterval: SAMPLE_RATE / 100, // samples between ODF ticks (441)
  odfSamplesProcessed: 0,
  lastOnset: false,
};

// ===== Audio Frame Processing (called ~43x/sec with 4096 samples, hop 1024) =====
function processFrame(samples) {
  const now = performance.now();

  // FFT
  const magnitudes = fftProcessor.process(samples);
  state.magnitudes = magnitudes;

  // Band energy
  const bandEnergy = bandFilter.filter(magnitudes);

  // RMS amplitude (from time domain)
  let sumSq = 0;
  for (let i = 0; i < samples.length; i++) {
    sumSq += samples[i] * samples[i];
  }
  state.rmsAmplitude = Math.sqrt(sumSq / samples.length);

  // Amplitude gate
  if (state.rmsAmplitude < state.amplitudeThreshold) {
    // Below threshold - feed silence to onset detector
    state.lastOnset = false;
  } else {
    // Onset detection
    state.lastOnset = onsetDetector.process(magnitudes, bandEnergy, now);
  }

  // Peak frequency (for display)
  let maxMag = 0, maxBin = 0;
  for (let i = 1; i < magnitudes.length; i++) {
    if (magnitudes[i] > maxMag) {
      maxMag = magnitudes[i];
      maxBin = i;
    }
  }
  state.peakFreq = maxBin * SAMPLE_RATE / FFT_SIZE;

  // Feed ODF at ~100 Hz
  // Each frame processes HOP_SIZE=1024 new samples
  state.odfSamplesProcessed += 1024;
  while (state.odfAccumulator + state.odfInterval <= state.odfSamplesProcessed) {
    state.odfAccumulator += state.odfInterval;
    const result = tempoEstimator.addOnsetSample(state.lastOnset);
    if (result) {
      state.bpm = result.bpm;
      state.confidence = result.confidence;
      state.isAtRangeLimit = result.isAtRangeLimit;
      state.limitSide = result.limitSide;
    }
  }

  // Blend with tap tempo
  const tapConf = tapProcessor.getConfidenceAt(now);
  if (tapConf > 0 && tapProcessor.lastBpm > 0) {
    const blended = tempoEstimator.blendWithTap(tapProcessor.lastBpm, tapConf);
    state.bpm = blended.bpm;
    state.confidence = blended.confidence;
    state.isAtRangeLimit = blended.isAtRangeLimit;
    state.limitSide = blended.limitSide;
  }

  // Feed spectrogram
  spectrogram.addColumn(magnitudes);
}

// ===== Render Loop (60fps) =====
let rafId = null;
function renderLoop() {
  // Update stats readout
  if (state.running && state.peakFreq > 0) {
    dom.statFreq.textContent = Math.round(state.peakFreq) + 'HZ';
    const sig = Math.min(1, state.rmsAmplitude * 4) * 100;
    dom.statSig.textContent = Math.round(sig) + '%';
    dom.statConf.textContent = Math.round(state.confidence * 100) + '%';
  }

  // Render gauge
  bpmGauge.render({
    bpm: state.bpm,
    confidence: state.confidence,
    rangeMin: state.rangeMin,
    rangeMax: state.rangeMax,
    isAtRangeLimit: state.isAtRangeLimit,
    limitSide: state.limitSide,
    peakFreq: state.peakFreq,
  });

  // Render spectrogram
  spectrogram.render();

  rafId = requestAnimationFrame(renderLoop);
}

// ===== Band Toggles =====
function initBands() {
  const map = { sub: BANDS.SUB, mid: BANDS.MID, hi: BANDS.HI };
  ['sub', 'mid', 'hi'].forEach(key => {
    const btn = dom[`band${key.charAt(0).toUpperCase() + key.slice(1)}`];
    btn.addEventListener('click', () => {
      const band = map[key];
      // Prevent deselecting all bands
      if (state.activeBands.has(band) && state.activeBands.size <= 1) return;

      if (state.activeBands.has(band)) {
        state.activeBands.delete(band);
        btn.classList.remove('active');
      } else {
        state.activeBands.add(band);
        btn.classList.add('active');
      }
      onsetDetector.activeBands = new Set(state.activeBands);
    });
  });
}

// ===== Dual Range Slider =====
function updateRangeTrack() {
  const min = parseInt(dom.rangeMin.value);
  const max = parseInt(dom.rangeMax.value);
  const pctMin = ((min - 60) / 180) * 100;
  const pctMax = ((max - 60) / 180) * 100;
  dom.rangeTrack.style.background = `linear-gradient(to right,
    var(--text-dim) ${pctMin}%,
    var(--accent-dim) ${pctMin}%,
    var(--accent-dim) ${pctMax}%,
    var(--text-dim) ${pctMax}%)`;
  dom.rangeMinLabel.textContent = min;
  dom.rangeMaxLabel.textContent = max;
  state.rangeMin = min;
  state.rangeMax = max;
  tempoEstimator.bpmRangeMin = min;
  tempoEstimator.bpmRangeMax = max;
}

function initRange() {
  dom.rangeMin.addEventListener('input', () => {
    if (parseInt(dom.rangeMin.value) > parseInt(dom.rangeMax.value) - 5) {
      dom.rangeMin.value = parseInt(dom.rangeMax.value) - 5;
    }
    updateRangeTrack();
  });
  dom.rangeMax.addEventListener('input', () => {
    if (parseInt(dom.rangeMax.value) < parseInt(dom.rangeMin.value) + 5) {
      dom.rangeMax.value = parseInt(dom.rangeMin.value) + 5;
    }
    updateRangeTrack();
  });
  updateRangeTrack();
}

// ===== Threshold Slider =====
function initThreshold() {
  dom.threshold.addEventListener('input', () => {
    const val = parseInt(dom.threshold.value);
    dom.thresholdValue.textContent = val;
    // Map 0-100 slider to 0-0.5 amplitude threshold
    state.amplitudeThreshold = val / 200;
  });
}

// ===== Tap Tempo =====
function initTap() {
  const handleTap = () => {
    const now = performance.now();

    // Flash animation
    dom.tapArea.classList.remove('flash');
    void dom.tapArea.offsetWidth;
    dom.tapArea.classList.add('flash');

    const result = tapProcessor.tap(now);
    state.tapCount = result.tapCount;

    if (result.bpm > 0) {
      dom.tapBpm.textContent = `${result.bpm.toFixed(1)} BPM  ×${result.tapCount}`;
    } else if (result.tapCount > 0) {
      dom.tapBpm.textContent = `×${result.tapCount}`;
    } else {
      dom.tapBpm.textContent = '';
    }
  };

  dom.tapArea.addEventListener('pointerdown', e => {
    e.preventDefault();
    handleTap();
  });
  document.addEventListener('keydown', e => {
    if (e.code === 'Space' && document.activeElement !== dom.rangeMin && document.activeElement !== dom.rangeMax) {
      e.preventDefault();
      handleTap();
    }
  });
}

// ===== Start / Stop =====
async function startAudio() {
  if (state.running) return;

  try {
    audioCapture = new AudioCapture(processFrame);
    await audioCapture.start();
    state.running = true;
    dom.btnStart.classList.add('active');
    dom.btnStart.textContent = '■ STOP MIC';
    dom.app.classList.add('listening');
    console.log('[BPMredux] Audio started');
  } catch (err) {
    console.error('[BPMredux] Failed to start audio:', err);
    alert('Could not access microphone. Please allow microphone access and try again.');
  }
}

function stopAudio() {
  if (!state.running) return;

  if (audioCapture) {
    audioCapture.stop();
    audioCapture = null;
  }
  state.running = false;
  state.bpm = 0;
  state.confidence = 0;
  state.odfSamplesProcessed = 0;
  state.odfAccumulator = 0;
  state.lastOnset = false;

  // Reset DSP
  onsetDetector.reset();
  tempoEstimator.reset();

  dom.btnStart.classList.remove('active');
  dom.btnStart.textContent = '▶ START MIC';
  dom.app.classList.remove('listening');
  console.log('[BPMredux] Audio stopped');
}

function initStart() {
  dom.btnStart.addEventListener('click', () => {
    if (state.running) {
      stopAudio();
    } else {
      startAudio();
    }
  });
}

// ===== Init =====
function init() {
  initBands();
  initRange();
  initThreshold();
  initTap();
  initStart();

  // Set initial tempo estimator range
  tempoEstimator.bpmRangeMin = state.rangeMin;
  tempoEstimator.bpmRangeMax = state.rangeMax;

  // Start render loop
  renderLoop();

  console.log('[BPMredux] Initialized — tap START MIC or press Space to tap tempo');
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
