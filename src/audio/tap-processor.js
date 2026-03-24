/**
 * TapProcessor - Tap tempo with outlier rejection and confidence decay
 * Ported from Kotlin BPMredux
 */

const MAX_TAPS = 8;
const MIN_TAPS = 3;
const DECAY_START = 5000;       // ms before confidence starts decaying
const RESET_AFTER = 10000;      // ms after last tap to auto-reset

export class TapProcessor {
  constructor() {
    /** @type {number[]} Tap timestamps in ms */
    this.taps = [];
    this.lastTapTime = 0;
    this.lastBpm = 0;
    this.lastConfidence = 0;
  }

  /**
   * Register a tap event.
   * @param {number} timeMs - current time in milliseconds
   * @returns {{bpm: number, confidence: number, tapCount: number}|null}
   */
  tap(timeMs) {
    // Auto-reset if too long since last tap
    if (this.taps.length > 0 && (timeMs - this.lastTapTime) > RESET_AFTER) {
      this.reset();
    }

    this.taps.push(timeMs);
    this.lastTapTime = timeMs;

    // Keep only last MAX_TAPS
    if (this.taps.length > MAX_TAPS) {
      this.taps = this.taps.slice(this.taps.length - MAX_TAPS);
    }

    // Need minimum taps
    if (this.taps.length < MIN_TAPS) {
      return { bpm: 0, confidence: 0, tapCount: this.taps.length };
    }

    // Compute intervals
    const intervals = [];
    for (let i = 1; i < this.taps.length; i++) {
      intervals.push(this.taps[i] - this.taps[i - 1]);
    }

    // Outlier rejection: remove intervals > 2x or < 0.5x median
    const filtered = this.rejectOutliers(intervals);

    if (filtered.length === 0) {
      return { bpm: 0, confidence: 0, tapCount: this.taps.length };
    }

    // Average interval
    let sum = 0;
    for (let i = 0; i < filtered.length; i++) {
      sum += filtered[i];
    }
    const avgInterval = sum / filtered.length;

    if (avgInterval <= 0) {
      return { bpm: 0, confidence: 0, tapCount: this.taps.length };
    }

    // BPM from average interval
    const bpm = 60000.0 / avgInterval;

    // Confidence = 1 - stddev/mean (coefficient of variation)
    let variance = 0;
    for (let i = 0; i < filtered.length; i++) {
      const diff = filtered[i] - avgInterval;
      variance += diff * diff;
    }
    variance /= filtered.length;
    const stddev = Math.sqrt(variance);
    const confidence = Math.max(0, Math.min(1, 1.0 - stddev / avgInterval));

    this.lastBpm = bpm;
    this.lastConfidence = confidence;

    return { bpm, confidence, tapCount: this.taps.length };
  }

  /**
   * Remove outlier intervals (> 2x or < 0.5x median).
   * @param {number[]} intervals
   * @returns {number[]} filtered intervals
   */
  rejectOutliers(intervals) {
    if (intervals.length < 2) return intervals.slice();

    // Compute median
    const sorted = intervals.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const median = sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2.0
      : sorted[mid];

    // Filter
    return intervals.filter(iv => iv >= median * 0.5 && iv <= median * 2.0);
  }

  /**
   * Get confidence at a given time, with decay.
   * @param {number} timeMs - current time in milliseconds
   * @returns {number} confidence (0-1)
   */
  getConfidenceAt(timeMs) {
    if (this.taps.length < MIN_TAPS) return 0;

    const elapsed = timeMs - this.lastTapTime;

    if (elapsed > RESET_AFTER) {
      return 0;
    }

    if (elapsed <= DECAY_START) {
      return this.lastConfidence;
    }

    // Linear decay from DECAY_START to RESET_AFTER
    const decayDuration = RESET_AFTER - DECAY_START;
    const decayElapsed = elapsed - DECAY_START;
    const decayFactor = 1.0 - (decayElapsed / decayDuration);

    return this.lastConfidence * Math.max(0, decayFactor);
  }

  /**
   * Reset tap state.
   */
  reset() {
    this.taps = [];
    this.lastTapTime = 0;
    this.lastBpm = 0;
    this.lastConfidence = 0;
  }
}
