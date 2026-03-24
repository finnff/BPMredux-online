/**
 * OnsetDetector - Adaptive spectral flux onset detection
 * Ported from Kotlin BPMredux
 */

import { BANDS } from './band-filter.js';

const HISTORY_SIZE_DEFAULT = 22;
const MIN_ONSET_INTERVAL_MS = 100;

// Hardcoded bin ranges matching Kotlin version
const BIN_RANGES = {
  SUB: { from: 4, to: 14 },
  MID: { from: 14, to: 186 },
  HI:  { from: 186, to: 929 }
};

// Adaptive threshold targets
const TARGET_ONSETS_PER_SEC_MIN = 1.0;
const TARGET_ONSETS_PER_SEC_MAX = 8.0;

export class OnsetDetector {
  constructor() {
    /** @type {Set<string>} Active bands for onset detection */
    this.activeBands = new Set([BANDS.SUB, BANDS.MID, BANDS.HI]);

    // Stability-adjustable history size (default = 22)
    this.historySize = HISTORY_SIZE_DEFAULT;

    // Previous magnitude spectrum for spectral flux
    this.prevMagnitudes = null;

    // Flux history for adaptive thresholding (circular buffer)
    this.fluxHistory = new Float32Array(this.historySize);
    this.fluxHistoryIdx = 0;
    this.fluxHistoryCount = 0;

    // Timing
    this.lastOnsetTimeMs = -Infinity;

    // Adaptive sensitivity
    this.sensitivity = 1.5;
    this.onsetTimestamps = [];  // recent onset times for rate estimation
  }

  /**
   * Set stability level (0-1).
   * 0 = most responsive (smaller history, faster adaptation)
   * 1 = most stable (larger history, slower adaptation)
   * @param {number} level - 0.0 to 1.0
   */
  setStability(level) {
    // HISTORY_SIZE: 10 (responsive) to 40 (stable), default 22 at level 0.5
    const newSize = Math.round(10 + level * 30);

    if (newSize !== this.historySize) {
      // Reallocate buffer if size changed (preserve existing data if possible)
      const newBuffer = new Float32Array(newSize);
      const copyCount = Math.min(this.fluxHistoryCount, newSize);
      for (let i = 0; i < copyCount; i++) {
        const srcIdx = (this.fluxHistoryIdx - this.fluxHistoryCount + i + this.historySize) % this.historySize;
        newBuffer[i] = this.fluxHistory[srcIdx];
      }
      this.fluxHistory = newBuffer;
      this.historySize = newSize;
      this.fluxHistoryIdx = copyCount % newSize;
      this.fluxHistoryCount = Math.min(this.fluxHistoryCount, newSize);
    }
  }

  /**
   * Half-wave rectified spectral flux between two spectra for a bin range.
   * @param {Float32Array} prev - previous magnitude spectrum
   * @param {Float32Array} curr - current magnitude spectrum
   * @param {number} from - start bin (inclusive)
   * @param {number} to - end bin (exclusive)
   * @returns {number} spectral flux (positive differences only)
   */
  bandFlux(prev, curr, from, to) {
    let flux = 0.0;
    const maxBin = Math.min(to, curr.length, prev.length);
    for (let i = from; i < maxBin; i++) {
      const diff = curr[i] - prev[i];
      if (diff > 0) {
        flux += diff;
      }
    }
    return flux;
  }

  /**
   * Compute combined flux across active bands (normalized per bin count).
   * Each band contributes equally regardless of bin count.
   */
  computeFlux(prevMag, currMag) {
    let totalFlux = 0.0;
    if (this.activeBands.has(BANDS.SUB)) {
      const binCount = BIN_RANGES.SUB.to - BIN_RANGES.SUB.from;
      totalFlux += this.bandFlux(prevMag, currMag, BIN_RANGES.SUB.from, BIN_RANGES.SUB.to) / binCount;
    }
    if (this.activeBands.has(BANDS.MID)) {
      const binCount = BIN_RANGES.MID.to - BIN_RANGES.MID.from;
      totalFlux += this.bandFlux(prevMag, currMag, BIN_RANGES.MID.from, BIN_RANGES.MID.to) / binCount;
    }
    if (this.activeBands.has(BANDS.HI)) {
      const binCount = BIN_RANGES.HI.to - BIN_RANGES.HI.from;
      totalFlux += this.bandFlux(prevMag, currMag, BIN_RANGES.HI.from, BIN_RANGES.HI.to) / binCount;
    }
    return totalFlux;
  }

  /**
   * Adapt sensitivity based on recent onset rate.
   */
  adaptSensitivity(timeMs) {
    // Remove old timestamps (keep last 2 seconds)
    const cutoff = timeMs - 2000;
    this.onsetTimestamps = this.onsetTimestamps.filter(t => t > cutoff);

    const onsetsPerSec = this.onsetTimestamps.length / 2.0;

    if (onsetsPerSec > TARGET_ONSETS_PER_SEC_MAX) {
      // Too many onsets, increase threshold (less sensitive)
      this.sensitivity = Math.min(this.sensitivity + 0.05, 4.0);
    } else if (onsetsPerSec < TARGET_ONSETS_PER_SEC_MIN) {
      // Too few onsets, decrease threshold (more sensitive)
      this.sensitivity = Math.max(this.sensitivity - 0.05, 0.5);
    }
  }

  /**
   * Process a frame and detect onset.
   * @param {Float32Array} magnitudes - current magnitude spectrum
   * @param {{sub: number, mid: number, hi: number}} bandEnergy - band energies (unused for flux but available)
   * @param {number} timeMs - current time in milliseconds
   * @returns {number} flux value (positive number) if onset, 0 otherwise
   */
  process(magnitudes, bandEnergy, timeMs) {
    if (this.prevMagnitudes === null) {
      this.prevMagnitudes = new Float32Array(magnitudes);
      return 0;
    }

    // Enforce minimum onset interval
    if (timeMs - this.lastOnsetTimeMs < MIN_ONSET_INTERVAL_MS) {
      this.prevMagnitudes = new Float32Array(magnitudes);
      return 0;
    }

    // Compute spectral flux
    const flux = this.computeFlux(this.prevMagnitudes, magnitudes);
    this.prevMagnitudes = new Float32Array(magnitudes);

    // Add to history
    this.fluxHistory[this.fluxHistoryIdx] = flux;
    this.fluxHistoryIdx = (this.fluxHistoryIdx + 1) % this.historySize;
    if (this.fluxHistoryCount < this.historySize) {
      this.fluxHistoryCount++;
    }

    // Need enough history for threshold
    if (this.fluxHistoryCount < 3) {
      return 0;
    }

    // Compute adaptive threshold: mean + sensitivity * stddev
    let sum = 0.0;
    for (let i = 0; i < this.fluxHistoryCount; i++) {
      sum += this.fluxHistory[i];
    }
    const mean = sum / this.fluxHistoryCount;

    let variance = 0.0;
    for (let i = 0; i < this.fluxHistoryCount; i++) {
      const diff = this.fluxHistory[i] - mean;
      variance += diff * diff;
    }
    variance /= this.fluxHistoryCount;
    const stddev = Math.sqrt(variance);

    const threshold = mean + this.sensitivity * stddev;

    const isOnset = flux > threshold;

    if (isOnset) {
      this.lastOnsetTimeMs = timeMs;
      this.onsetTimestamps.push(timeMs);
    }

    // Adapt sensitivity
    this.adaptSensitivity(timeMs);

    return isOnset ? flux : 0;
  }

  /**
   * Reset detector state.
   */
  reset() {
    this.prevMagnitudes = null;
    this.fluxHistory.fill(0);
    this.fluxHistoryIdx = 0;
    this.fluxHistoryCount = 0;
    this.lastOnsetTimeMs = -Infinity;
    this.sensitivity = 1.5;
    this.onsetTimestamps = [];
  }
}
