// spectrogram.js - Mel-scale spectrogram with CRT teal aesthetic
import { Black, AccentDim, AccentSubtle, TextSecondary, TextDim, rgba, buildTealLut } from './colors.js';

const FREQ_BINS = 128;
const TIME_ROWS = 256;
const MAX_FREQ = 10000;
const SAMPLE_RATE = 44100;
const DB_MIN = -80;
const DB_MAX = -10;
const SCANLINE_INTERVAL = 3;
const SCANLINE_OPACITY = 0.12;

function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel) {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

/**
 * Build mel-scale mapping: for each mel bin (0..FREQ_BINS-1),
 * store the corresponding FFT bin index range.
 */
function buildMelMapping(fftSize) {
  const melMin = hzToMel(0);
  const melMax = hzToMel(MAX_FREQ);
  const mapping = new Array(FREQ_BINS);
  const binHz = SAMPLE_RATE / fftSize;

  for (let i = 0; i < FREQ_BINS; i++) {
    const melLo = melMin + (melMax - melMin) * (i / FREQ_BINS);
    const melHi = melMin + (melMax - melMin) * ((i + 1) / FREQ_BINS);
    const hzLo = melToHz(melLo);
    const hzHi = melToHz(melHi);
    const binLo = Math.max(0, Math.floor(hzLo / binHz));
    const binHi = Math.min(Math.ceil(hzHi / binHz), fftSize / 2);
    mapping[i] = { binLo, binHi, hzCenter: melToHz((melLo + melHi) / 2) };
  }
  return mapping;
}

function getMelBinForHz(hz) {
  const melMin = hzToMel(0);
  const melMax = hzToMel(MAX_FREQ);
  const mel = hzToMel(hz);
  return Math.round(((mel - melMin) / (melMax - melMin)) * FREQ_BINS);
}

export class Spectrogram {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.lut = buildTealLut();
    this.melMapping = null;
    this._currentRow = 0;

    // Offscreen bitmap canvas
    this._offCanvas = document.createElement('canvas');
    this._offCanvas.width = FREQ_BINS;
    this._offCanvas.height = TIME_ROWS;
    this._offCtx = this._offCanvas.getContext('2d');
    this._imageData = this._offCtx.createImageData(FREQ_BINS, 1);

    // Band boundaries (mel bin indices)
    this._subMidBin = getMelBinForHz(150);
    this._midHiBin = getMelBinForHz(2000);
  }

  /**
   * Add one column (time row) of magnitude data.
   * @param {Float32Array} magnitudes - raw FFT magnitude data (linear scale or dB)
   */
  addColumn(magnitudes) {
    // Lazy-init mel mapping based on incoming data size
    if (!this.melMapping || this._lastFftSize !== magnitudes.length * 2) {
      this._lastFftSize = magnitudes.length * 2;
      this.melMapping = buildMelMapping(this._lastFftSize);
    }

    const pixels = this._imageData.data;
    const lut = this.lut;

    for (let i = 0; i < FREQ_BINS; i++) {
      const { binLo, binHi } = this.melMapping[i];
      // Average magnitudes across the mel bin range
      let sum = 0;
      let count = 0;
      for (let b = binLo; b < binHi && b < magnitudes.length; b++) {
        sum += magnitudes[b];
        count++;
      }
      const avg = count > 0 ? sum / count : 0;

      // Convert to dB
      const db = avg > 0 ? 20 * Math.log10(avg) : DB_MIN;
      const norm = Math.max(0, Math.min(1, (db - DB_MIN) / (DB_MAX - DB_MIN)));
      const lutIdx = Math.round(norm * 255);

      // Frequency bins go bottom-to-top, pixel x goes left-to-right
      const px = i;
      pixels[px * 4] = lut[lutIdx * 3];
      pixels[px * 4 + 1] = lut[lutIdx * 3 + 1];
      pixels[px * 4 + 2] = lut[lutIdx * 3 + 2];
      pixels[px * 4 + 3] = 255;
    }

    this._offCtx.putImageData(this._imageData, 0, this._currentRow);
    this._currentRow = (this._currentRow + 1) % TIME_ROWS;
  }

  /** Render the spectrogram to the visible canvas. */
  render() {
    const { w, h } = this._setupDpi();
    const ctx = this.ctx;

    // Clear
    ctx.fillStyle = Black;
    ctx.fillRect(0, 0, w, h);

    // Draw offscreen bitmap with vertical scroll wrap.
    // Newest row at bottom: the current row pointer is the next write position,
    // so rows from _currentRow..TIME_ROWS-1 are oldest, 0.._currentRow-1 are newest.
    // We rotate the bitmap so frequency is on y-axis (low=bottom) and time on x-axis.

    ctx.save();
    ctx.imageSmoothingEnabled = false;

    // We draw the offscreen canvas rotated: each "row" in offscreen becomes a column on screen.
    // Offscreen: x=freq bin, y=time row
    // Screen: x=time (left=old, right=new), y=freq (bottom=low, top=high)

    const oldRows = TIME_ROWS - this._currentRow;
    const newRows = this._currentRow;

    // Older portion (from _currentRow to end)
    if (oldRows > 0) {
      // Source: sx=0, sy=_currentRow, sw=FREQ_BINS, sh=oldRows
      // Dest: draw so time goes left-right, freq bottom-to-top
      // We need to rotate 90 degrees CW and flip freq axis
      this._drawRotatedSection(ctx, this._currentRow, oldRows, 0, w, h);
    }

    // Newer portion (from 0 to _currentRow)
    if (newRows > 0) {
      this._drawRotatedSection(ctx, 0, newRows, oldRows, w, h);
    }

    ctx.restore();

    // --- Noise grain overlay ---
    this._drawNoise(ctx, w, h);

    // --- Edge fades ---
    this._drawEdgeFades(ctx, w, h);

    // --- Band boundary lines and labels ---
    this._drawBandBoundaries(ctx, w, h);

    // --- Scanline overlay ---
    this._drawScanlines(ctx, w, h);
  }

  /**
   * Draw a section of the offscreen canvas rotated 90 degrees.
   * Offscreen row = time, offscreen x = freq bin (0=low freq).
   * On screen: x = time, y = freq (0=high freq at top, FREQ_BINS=low freq at bottom).
   */
  _drawRotatedSection(ctx, srcRow, srcRows, destTimeOffset, canvasW, canvasH) {
    const colW = canvasW / TIME_ROWS;
    const destX = destTimeOffset * colW;
    const destW = srcRows * colW;

    // We draw the offscreen portion transposed:
    // offscreen (x=freq, y=time) -> screen (x=time, y=freq flipped)
    ctx.save();
    // Translate to destination, then rotate
    ctx.translate(destX, canvasH);
    ctx.scale(colW, -canvasH / FREQ_BINS);
    // Now drawing in offscreen coordinates where x maps to time-offset, y maps to freq bin
    // drawImage source: freq on x (0..FREQ_BINS), time slice on y
    ctx.drawImage(
      this._offCanvas,
      0, srcRow, FREQ_BINS, srcRows, // source rect
      0, 0, srcRows, FREQ_BINS       // dest rect (will be scaled)
    );
    ctx.restore();
  }

  _drawNoise(ctx, w, h) {
    // Cache noise pattern on first call
    if (!this._noisePattern) {
      const noiseCanvas = document.createElement('canvas');
      noiseCanvas.width = 64;
      noiseCanvas.height = 64;
      const nCtx = noiseCanvas.getContext('2d');
      const nData = nCtx.createImageData(64, 64);
      for (let i = 0; i < nData.data.length; i += 4) {
        const v = Math.random() * 20;
        nData.data[i] = v;
        nData.data[i + 1] = v;
        nData.data[i + 2] = v;
        nData.data[i + 3] = 12;
      }
      nCtx.putImageData(nData, 0, 0);
      this._noisePattern = ctx.createPattern(noiseCanvas, 'repeat');
    }
    ctx.save();
    ctx.globalAlpha = 0.3;
    ctx.fillStyle = this._noisePattern;
    ctx.fillRect(0, 0, w, h);
    ctx.restore();
  }

  _drawEdgeFades(ctx, w, h) {
    const fadeV = h * 0.08;
    const fadeH = w * 0.05;

    // Top fade
    const topGrad = ctx.createLinearGradient(0, 0, 0, fadeV);
    topGrad.addColorStop(0, rgba(Black, 0.9));
    topGrad.addColorStop(1, rgba(Black, 0));
    ctx.fillStyle = topGrad;
    ctx.fillRect(0, 0, w, fadeV);

    // Bottom fade
    const botGrad = ctx.createLinearGradient(0, h - fadeV, 0, h);
    botGrad.addColorStop(0, rgba(Black, 0));
    botGrad.addColorStop(1, rgba(Black, 0.9));
    ctx.fillStyle = botGrad;
    ctx.fillRect(0, h - fadeV, w, fadeV);

    // Left fade
    const leftGrad = ctx.createLinearGradient(0, 0, fadeH, 0);
    leftGrad.addColorStop(0, rgba(Black, 0.9));
    leftGrad.addColorStop(1, rgba(Black, 0));
    ctx.fillStyle = leftGrad;
    ctx.fillRect(0, 0, fadeH, h);

    // Right fade
    const rightGrad = ctx.createLinearGradient(w - fadeH, 0, w, 0);
    rightGrad.addColorStop(0, rgba(Black, 0));
    rightGrad.addColorStop(1, rgba(Black, 0.9));
    ctx.fillStyle = rightGrad;
    ctx.fillRect(w - fadeH, 0, fadeH, h);
  }

  _drawBandBoundaries(ctx, w, h) {
    // Freq bin 0 = low freq at bottom of screen
    // y position for a freq bin: y = h - (bin / FREQ_BINS) * h
    const subMidY = h - (this._subMidBin / FREQ_BINS) * h;
    const midHiY = h - (this._midHiBin / FREQ_BINS) * h;

    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;

    // Sub/Mid boundary
    ctx.beginPath();
    ctx.moveTo(0, subMidY);
    ctx.lineTo(w, subMidY);
    ctx.strokeStyle = rgba(TextDim, 0.5);
    ctx.stroke();

    // Mid/Hi boundary
    ctx.beginPath();
    ctx.moveTo(0, midHiY);
    ctx.lineTo(w, midHiY);
    ctx.strokeStyle = rgba(TextDim, 0.5);
    ctx.stroke();

    ctx.setLineDash([]);

    // Labels
    const fontSize = Math.max(9, Math.min(12, w * 0.03));
    ctx.font = `${fontSize}px monospace`;
    ctx.fillStyle = rgba(TextSecondary, 0.5);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';

    const labelX = 6;
    // SUB label below subMid line
    ctx.fillText('SUB', labelX, subMidY - 3);
    // MID label below midHi line
    ctx.fillText('MID', labelX, midHiY - 3);
    // HI label near top
    ctx.textBaseline = 'top';
    ctx.fillText('HI', labelX, midHiY + 3);
  }

  _drawScanlines(ctx, w, h) {
    ctx.fillStyle = rgba(Black, SCANLINE_OPACITY);
    for (let y = 0; y < h; y += SCANLINE_INTERVAL) {
      ctx.fillRect(0, y, w, 1);
    }
  }

  _setupDpi() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    if (this.canvas.width !== w * dpr || this.canvas.height !== h * dpr) {
      this.canvas.width = w * dpr;
      this.canvas.height = h * dpr;
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    return { w, h };
  }
}
