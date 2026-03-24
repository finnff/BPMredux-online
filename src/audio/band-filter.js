/**
 * BandFilter - Frequency band energy extraction
 * Ported from Kotlin BPMredux
 */

export const BANDS = {
  SUB: 'SUB',
  MID: 'MID',
  HI: 'HI'
};

const SAMPLE_RATE = 44100;
const FFT_SIZE = 4096;

// Frequency to FFT bin: bin = freq * FFT_SIZE / SAMPLE_RATE
function freqToBin(freq) {
  return Math.round(freq * FFT_SIZE / SAMPLE_RATE);
}

// Band frequency ranges
const BAND_RANGES = {
  SUB: { low: 40, high: 150 },
  MID: { low: 150, high: 2000 },
  HI:  { low: 2000, high: 10000 }
};

// Pre-compute bin ranges
const BAND_BINS = {
  SUB: { from: freqToBin(BAND_RANGES.SUB.low), to: freqToBin(BAND_RANGES.SUB.high) },
  MID: { from: freqToBin(BAND_RANGES.MID.low), to: freqToBin(BAND_RANGES.MID.high) },
  HI:  { from: freqToBin(BAND_RANGES.HI.low),  to: freqToBin(BAND_RANGES.HI.high) }
};

export class BandFilter {
  constructor() {
    this.sampleRate = SAMPLE_RATE;
    this.fftSize = FFT_SIZE;
    this.bandBins = BAND_BINS;
  }

  /**
   * Compute RMS energy for a frequency band from magnitude spectrum.
   * @param {Float32Array} magnitudes - magnitude spectrum
   * @param {number} fromBin - start bin (inclusive)
   * @param {number} toBin - end bin (exclusive)
   * @returns {number} RMS energy
   */
  rmsEnergy(magnitudes, fromBin, toBin) {
    let sum = 0.0;
    let count = 0;
    const maxBin = Math.min(toBin, magnitudes.length);
    for (let i = fromBin; i < maxBin; i++) {
      sum += magnitudes[i] * magnitudes[i];
      count++;
    }
    if (count === 0) return 0.0;
    return Math.sqrt(sum / count);
  }

  /**
   * Filter magnitude spectrum into band energies.
   * @param {Float32Array} magnitudes - magnitude spectrum from FFT
   * @returns {{sub: number, mid: number, hi: number}} RMS energies per band
   */
  filter(magnitudes) {
    return {
      sub: this.rmsEnergy(magnitudes, BAND_BINS.SUB.from, BAND_BINS.SUB.to),
      mid: this.rmsEnergy(magnitudes, BAND_BINS.MID.from, BAND_BINS.MID.to),
      hi:  this.rmsEnergy(magnitudes, BAND_BINS.HI.from,  BAND_BINS.HI.to)
    };
  }
}
