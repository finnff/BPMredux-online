/**
 * FFTProcessor - Radix-2 Cooley-Tukey DIT FFT
 * Ported from Kotlin BPMredux
 */
export class FFTProcessor {
  constructor(size = 4096) {
    this.size = size;
    this.logN = Math.round(Math.log2(size));

    if ((1 << this.logN) !== size) {
      throw new Error(`FFT size must be a power of 2, got ${size}`);
    }

    // Pre-compute Hann window
    this.hannWindow = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      this.hannWindow[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / (size - 1)));
    }

    // Pre-compute bit-reversal table
    this.bitReversal = new Uint32Array(size);
    for (let i = 0; i < size; i++) {
      let reversed = 0;
      let val = i;
      for (let j = 0; j < this.logN; j++) {
        reversed = (reversed << 1) | (val & 1);
        val >>= 1;
      }
      this.bitReversal[i] = reversed;
    }
  }

  /**
   * Apply Hann window to samples.
   * Web Audio gives -1..1 floats, so no /32768 normalization needed.
   */
  applyWindow(float32Array) {
    const windowed = new Float32Array(this.size);
    const len = Math.min(float32Array.length, this.size);
    for (let i = 0; i < len; i++) {
      windowed[i] = float32Array[i] * this.hannWindow[i];
    }
    return windowed;
  }

  /**
   * In-place radix-2 Cooley-Tukey decimation-in-time FFT.
   * @param {Float32Array} real - real part (modified in-place)
   * @param {Float32Array} imag - imaginary part (modified in-place)
   */
  fft(real, imag) {
    const n = this.size;

    // Bit-reversal permutation
    for (let i = 0; i < n; i++) {
      const j = this.bitReversal[i];
      if (j > i) {
        let tmp = real[i];
        real[i] = real[j];
        real[j] = tmp;
        tmp = imag[i];
        imag[i] = imag[j];
        imag[j] = tmp;
      }
    }

    // Cooley-Tukey butterfly
    for (let stage = 1; stage <= this.logN; stage++) {
      const m = 1 << stage;        // butterfly group size
      const halfM = m >> 1;        // half group size
      const wReal = Math.cos(-2.0 * Math.PI / m);
      const wImag = Math.sin(-2.0 * Math.PI / m);

      for (let k = 0; k < n; k += m) {
        let curReal = 1.0;
        let curImag = 0.0;

        for (let j = 0; j < halfM; j++) {
          const evenIdx = k + j;
          const oddIdx = k + j + halfM;

          const tReal = curReal * real[oddIdx] - curImag * imag[oddIdx];
          const tImag = curReal * imag[oddIdx] + curImag * real[oddIdx];

          real[oddIdx] = real[evenIdx] - tReal;
          imag[oddIdx] = imag[evenIdx] - tImag;
          real[evenIdx] = real[evenIdx] + tReal;
          imag[evenIdx] = imag[evenIdx] + tImag;

          const newCurReal = curReal * wReal - curImag * wImag;
          const newCurImag = curReal * wImag + curImag * wReal;
          curReal = newCurReal;
          curImag = newCurImag;
        }
      }
    }
  }

  /**
   * Compute magnitude spectrum from FFT output.
   * @returns {Float32Array} of size N/2 + 1
   */
  magnitudeSpectrum(real, imag) {
    const halfN = (this.size >> 1) + 1;
    const magnitudes = new Float32Array(halfN);
    for (let i = 0; i < halfN; i++) {
      magnitudes[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
    }
    return magnitudes;
  }

  /**
   * Full pipeline: window -> FFT -> magnitude spectrum
   * @param {Float32Array} samples - time-domain samples
   * @returns {Float32Array} magnitude spectrum
   */
  process(samples) {
    const windowed = this.applyWindow(samples);
    const real = new Float32Array(windowed);
    const imag = new Float32Array(this.size);
    this.fft(real, imag);
    return this.magnitudeSpectrum(real, imag);
  }
}
