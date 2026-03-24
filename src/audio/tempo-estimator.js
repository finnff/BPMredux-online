/**
 * TempoEstimator — Autocorrelation-based BPM estimation
 * Ported from Kotlin BPMredux (exact same algorithm)
 */

const ODF_SAMPLE_RATE = 100;      // Hz — onset detection function sample rate
const ODF_BUFFER_SIZE_DEFAULT = 400;      // 4 seconds at 100 Hz
const UPDATE_INTERVAL_SAMPLES_DEFAULT = 20; // ~0.2s between updates
const EMA_ALPHA_DEFAULT = 0.3;
const EXTENDED_BPM_MIN = 40;
const EXTENDED_BPM_MAX = 250;
const PEGGING_THRESHOLD = 0.3;    // out-of-range peak must be >30% stronger

export class TempoEstimator {
  constructor() {
    // Stability-adjustable parameters (default = middle/stable)
    this.emaAlpha = EMA_ALPHA_DEFAULT;
    this.odfBufferSize = ODF_BUFFER_SIZE_DEFAULT;
    this.updateIntervalSamples = UPDATE_INTERVAL_SAMPLES_DEFAULT;

    // ODF buffer initialized to default size
    this.odfBuffer = new Float32Array(this.odfBufferSize);
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
   * Set stability level (0-1).
   * 0 = most responsive (less smoothing, smaller buffer, more frequent updates)
   * 1 = most stable (more smoothing, larger buffer, less frequent updates)
   * @param {number} level - 0.0 to 1.0
   */
  setStability(level) {
    // EMA_ALPHA: 0.05 (stable) to 0.6 (responsive), default 0.3 at level 0.5
    // Lower alpha = more smoothing = more stable
    this.emaAlpha = 0.6 - level * 0.55; // 0.6 at 0, 0.05 at 1

    // ODF_BUFFER_SIZE: 200 (responsive) to 600 (stable), default 400 at level 0.5
    this.odfBufferSize = Math.round(200 + level * 400);

    // UPDATE_INTERVAL_SAMPLES: 10 (responsive) to 40 (stable), default 20 at level 0.5
    this.updateIntervalSamples = Math.round(10 + level * 30);

    // Reallocate buffer if size changed (preserve existing data if possible)
    if (this.odfBuffer.length !== this.odfBufferSize) {
      const newBuffer = new Float32Array(this.odfBufferSize);
      // Copy as much existing data as will fit
      const copyCount = Math.min(this.odfCount, this.odfBufferSize);
      for (let i = 0; i < copyCount; i++) {
        const srcIdx = (this.odfWritePos - this.odfCount + i) % this.odfBuffer.length;
        const dstIdx = i;
        newBuffer[dstIdx] = this.odfBuffer[srcIdx];
      }
      this.odfBuffer = newBuffer;
      this.odfWritePos = copyCount;
      this.odfCount = copyCount;
    }
  }

  /**
   * @param {number} onsetValue - flux value (positive number) if onset, 0 otherwise
   * @returns {null|{bpm: number, confidence: number, isAtRangeLimit: boolean, limitSide: string|null}}
   */
  addOnsetSample(onsetValue) {
    this.odfBuffer[this.odfWritePos % this.odfBufferSize] = onsetValue;
    this.odfWritePos++;
    this.odfCount = Math.min(this.odfCount + 1, this.odfBufferSize);
    this.samplesSinceUpdate++;

    if (this.samplesSinceUpdate < this.updateIntervalSamples || this.odfCount < ODF_SAMPLE_RATE * 2) {
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
        const idx1 = (start + i) % this.odfBufferSize;
        const idx2 = (start + i + lag) % this.odfBufferSize;
        sum += this.odfBuffer[idx1] * this.odfBuffer[idx2];
      }
      acf[lag] = sum;
    }

    // Find best in-range peak using local maxima detection
    const peaks = [];
    for (let lag = lagMin + 1; lag < lagMax; lag++) {
      if (acf[lag] > acf[lag - 1] && acf[lag] >= acf[lag + 1]) {
        peaks.push({ lag, val: acf[lag] });
      }
    }
    // Boundary checks
    if (acf[lagMin] >= acf[lagMin + 1]) peaks.push({ lag: lagMin, val: acf[lagMin] });
    if (acf[lagMax] >= acf[lagMax - 1]) peaks.push({ lag: lagMax, val: acf[lagMax] });

    peaks.sort((a, b) => b.val - a.val);

    let bestLag, bestVal, secondBestPeakVal;
    if (peaks.length >= 1) {
      bestLag = peaks[0].lag;
      bestVal = peaks[0].val;
      secondBestPeakVal = peaks.length >= 2 ? peaks[1].val : 0;
    } else {
      bestLag = acf[lagMin] >= acf[lagMax] ? lagMin : lagMax;
      bestVal = acf[bestLag];
      secondBestPeakVal = 0;
    }

    if (bestVal <= 0) return null;

    // Sub-harmonic check: prefer 2x lag if within range and strong
    const doubleLag = bestLag * 2;
    if (doubleLag >= lagMin && doubleLag <= lagMax) {
      const doubleVal = acf[doubleLag];
      if (doubleVal > bestVal * 0.8) {
        bestLag = doubleLag;
        bestVal = doubleVal;
        // Find second best peak again (excluding the new best)
        secondBestPeakVal = peaks.length >= 2 ? peaks[1].val : 0;
      }
    }

    // Half lag check
    const halfLag = Math.floor(bestLag / 2);
    if (halfLag >= lagMin && halfLag <= lagMax) {
      const halfVal = acf[halfLag];
      if (halfVal > bestVal * 0.9) {
        bestLag = halfLag;
        bestVal = halfVal;
        // Find second best peak again (excluding the new best)
        secondBestPeakVal = peaks.length >= 2 ? peaks[1].val : 0;
      }
    }

    // Parabolic interpolation for sub-sample precision
    let refinedLag = bestLag;
    if (bestLag > extLagMin && bestLag < effectiveLagMax) {
      const y0 = acf[bestLag - 1];
      const y1 = acf[bestLag];
      const y2 = acf[bestLag + 1];
      const denom = 2 * (2 * y1 - y0 - y2);
      if (denom > 0) {
        refinedLag = bestLag + (y0 - y2) / denom;
      }
    }
    const rawBpm = 60.0 * ODF_SAMPLE_RATE / refinedLag;
    const confidence = secondBestPeakVal > 0
      ? Math.min(1, (bestVal / secondBestPeakVal - 1) * 2)
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
      smoothedBpm = this.emaAlpha * rawBpm + (1 - this.emaAlpha) * this.lastSmoothedBpm;
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
