/**
 * TempoEstimator — Autocorrelation-based BPM estimation
 * Ported from Kotlin BPMredux (exact same algorithm)
 */

const ODF_SAMPLE_RATE = 100;      // Hz — onset detection function sample rate
const ODF_BUFFER_SIZE = 400;      // 4 seconds at 100 Hz
const UPDATE_INTERVAL_SAMPLES = 20; // ~0.2s between updates
const EMA_ALPHA = 0.3;
const EXTENDED_BPM_MIN = 40;
const EXTENDED_BPM_MAX = 250;
const PEGGING_THRESHOLD = 0.3;    // out-of-range peak must be >30% stronger

export class TempoEstimator {
  constructor() {
    this.odfBuffer = new Float32Array(ODF_BUFFER_SIZE);
    this.odfWritePos = 0;
    this.odfCount = 0;
    this.samplesSinceUpdate = 0;
    this.lastSmoothedBpm = 0;
    this.lastResult = { bpm: 0, confidence: 0, isAtRangeLimit: false, limitSide: null };

    this.bpmRangeMin = 120;
    this.bpmRangeMax = 180;

    // Public for tap blending
    this.lastBpm = 0;
  }

  /**
   * @param {boolean} isOnset
   * @returns {null|{bpm: number, confidence: number, isAtRangeLimit: boolean, limitSide: string|null}}
   */
  addOnsetSample(isOnset) {
    this.odfBuffer[this.odfWritePos % ODF_BUFFER_SIZE] = isOnset ? 1.0 : 0.0;
    this.odfWritePos++;
    this.odfCount = Math.min(this.odfCount + 1, ODF_BUFFER_SIZE);
    this.samplesSinceUpdate++;

    if (this.samplesSinceUpdate < UPDATE_INTERVAL_SAMPLES || this.odfCount < ODF_SAMPLE_RATE * 2) {
      return null;
    }
    this.samplesSinceUpdate = 0;

    // In-range lags
    const lagMin = Math.floor(60.0 * ODF_SAMPLE_RATE / this.bpmRangeMax);
    const lagMax = Math.floor(60.0 * ODF_SAMPLE_RATE / this.bpmRangeMin);

    // Extended lags for pegging detection
    const extLagMin = Math.floor(60.0 * ODF_SAMPLE_RATE / EXTENDED_BPM_MAX);
    const extLagMax = Math.floor(60.0 * ODF_SAMPLE_RATE / EXTENDED_BPM_MIN);

    const effectiveLagMax = Math.max(lagMax, extLagMax);
    if (lagMax <= lagMin || effectiveLagMax >= this.odfCount) return null;

    // Autocorrelation over full extended range
    const acf = new Float32Array(effectiveLagMax + 1);
    const bufLen = this.odfCount;
    const start = this.odfWritePos > bufLen ? this.odfWritePos - bufLen : 0;

    for (let lag = extLagMin; lag <= effectiveLagMax; lag++) {
      let sum = 0;
      const n = bufLen - lag;
      for (let i = 0; i < n; i++) {
        const idx1 = (start + i) % ODF_BUFFER_SIZE;
        const idx2 = (start + i + lag) % ODF_BUFFER_SIZE;
        sum += this.odfBuffer[idx1] * this.odfBuffer[idx2];
      }
      acf[lag] = sum;
    }

    // Find best in-range peak
    let bestLag = lagMin;
    let bestVal = acf[lagMin];
    let secondBest = 0;
    for (let lag = lagMin + 1; lag <= lagMax; lag++) {
      if (acf[lag] > bestVal) {
        secondBest = bestVal;
        bestVal = acf[lag];
        bestLag = lag;
      } else if (acf[lag] > secondBest) {
        secondBest = acf[lag];
      }
    }

    if (bestVal <= 0) return null;

    // Sub-harmonic check: prefer 2x lag if within range and strong
    const doubleLag = bestLag * 2;
    if (doubleLag >= lagMin && doubleLag <= lagMax) {
      const doubleVal = acf[doubleLag];
      if (doubleVal > bestVal * 0.8) {
        bestLag = doubleLag;
        bestVal = doubleVal;
      }
    }

    // Half lag check
    const halfLag = Math.floor(bestLag / 2);
    if (halfLag >= lagMin && halfLag <= lagMax) {
      const halfVal = acf[halfLag];
      if (halfVal > bestVal * 0.9) {
        bestLag = halfLag;
        bestVal = halfVal;
      }
    }

    const rawBpm = 60.0 * ODF_SAMPLE_RATE / bestLag;
    const confidence = secondBest > 0
      ? Math.min(1, (bestVal / secondBest - 1) * 2)
      : 1;

    // Range-limit pegging detection
    let isAtRangeLimit = false;
    let limitSide = null;

    // Scan out-of-range for stronger peaks
    let bestOutOfRangeLag = -1;
    let bestOutOfRangeVal = 0;
    for (let lag = extLagMin; lag <= effectiveLagMax; lag++) {
      if (lag >= lagMin && lag <= lagMax) continue; // skip in-range
      if (acf[lag] > bestOutOfRangeVal) {
        bestOutOfRangeVal = acf[lag];
        bestOutOfRangeLag = lag;
      }
    }

    if (bestOutOfRangeLag > 0 && bestOutOfRangeVal > bestVal * (1 + PEGGING_THRESHOLD)) {
      const outBpm = 60.0 * ODF_SAMPLE_RATE / bestOutOfRangeLag;
      const doubleBpm = outBpm * 2;
      const halfBpm = outBpm / 2;
      const doubleInRange = doubleBpm >= this.bpmRangeMin && doubleBpm <= this.bpmRangeMax;
      const halfInRange = halfBpm >= this.bpmRangeMin && halfBpm <= this.bpmRangeMax;

      if (!doubleInRange && !halfInRange) {
        isAtRangeLimit = true;
        limitSide = outBpm > this.bpmRangeMax ? 'MAX' : 'MIN';
      }
    }

    // EMA smoothing
    let smoothedBpm;
    if (this.lastSmoothedBpm === 0) {
      smoothedBpm = rawBpm;
    } else {
      smoothedBpm = EMA_ALPHA * rawBpm + (1 - EMA_ALPHA) * this.lastSmoothedBpm;
    }
    this.lastSmoothedBpm = smoothedBpm;

    this.lastResult = { bpm: smoothedBpm, confidence, isAtRangeLimit, limitSide };
    return this.lastResult;
  }

  /**
   * Blend with tap tempo — taps weighted MORE when algo confidence is LOW
   */
  blendWithTap(tapBpm, tapConfidence) {
    if (this.lastResult.bpm <= 0) {
      return { bpm: tapBpm, confidence: tapConfidence, isAtRangeLimit: false, limitSide: null };
    }
    if (tapBpm <= 0) return this.lastResult;

    const algoWeight = this.lastResult.confidence;
    const tapWeight = tapConfidence * (2 - this.lastResult.confidence);
    const totalWeight = algoWeight + tapWeight;

    const blended = (this.lastResult.bpm * algoWeight + tapBpm * tapWeight) / totalWeight;
    const blendedConfidence = Math.max(this.lastResult.confidence, tapConfidence);

    return {
      bpm: blended,
      confidence: blendedConfidence,
      isAtRangeLimit: this.lastResult.isAtRangeLimit,
      limitSide: this.lastResult.limitSide,
    };
  }

  reset() {
    this.odfBuffer.fill(0);
    this.odfWritePos = 0;
    this.odfCount = 0;
    this.samplesSinceUpdate = 0;
    this.lastSmoothedBpm = 0;
    this.lastResult = { bpm: 0, confidence: 0, isAtRangeLimit: false, limitSide: null };
  }
}

export { ODF_SAMPLE_RATE };
