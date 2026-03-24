/**
 * Spectrogram — Mel-scale waterfall display with CRT teal aesthetic
 * Matches Android BPMredux: frequency on X-axis (left=low, right=high),
 * time scrolling vertically (newest at bottom).
 */
import { Black, AccentDim, TextSecondary, TextDim, rgba, buildTealLut } from './colors.js';

const FREQ_BINS = 128;
const TIME_ROWS = 256;
const MAX_FREQ = 10000;
const SAMPLE_RATE = 44100;
const DB_MIN = -80;
const DB_MAX = -10;

function hzToMel(hz) { return 2595 * Math.log10(1 + hz / 700); }
function melToHz(mel) { return 700 * (Math.pow(10, mel / 2595) - 1); }

function buildMelMapping(fftSize) {
  const melMax = hzToMel(MAX_FREQ);
  const binHz = SAMPLE_RATE / fftSize;
  const mapping = new Array(FREQ_BINS);
  for (let i = 0; i < FREQ_BINS; i++) {
    const melLo = melMax * i / FREQ_BINS;
    const melHi = melMax * (i + 1) / FREQ_BINS;
    const hzLo = melToHz(melLo);
    const hzHi = melToHz(melHi);
    mapping[i] = {
      binLo: Math.max(0, Math.floor(hzLo / binHz)),
      binHi: Math.min(Math.ceil(hzHi / binHz), fftSize / 2),
    };
  }
  return mapping;
}

function getMelBinForHz(hz) {
  const melMax = hzToMel(MAX_FREQ);
  return Math.round((hzToMel(hz) / melMax) * FREQ_BINS);
}

export class Spectrogram {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.lut = buildTealLut();
    this.melMapping = null;
    this._writeRow = 0;
    this._totalRows = 0;

    // Offscreen bitmap: x = freq bin, y = time row
    this._off = document.createElement('canvas');
    this._off.width = FREQ_BINS;
    this._off.height = TIME_ROWS;
    this._offCtx = this._off.getContext('2d');
    this._rowData = this._offCtx.createImageData(FREQ_BINS, 1);

    // Band boundaries in mel-bin space
    this._subMidBin = getMelBinForHz(150);
    this._midHiBin = getMelBinForHz(2000);

    // Cached noise pattern
    this._noisePattern = null;
  }

  /**
   * Add one time row of magnitude data.
   * Layout: x=freq bin (0=low freq), y=time (scrolling down)
   */
  addColumn(magnitudes) {
    if (!this.melMapping || this._lastFftHalf !== magnitudes.length) {
      this._lastFftHalf = magnitudes.length;
      this.melMapping = buildMelMapping(magnitudes.length * 2);
    }

    const px = this._rowData.data;
    const lut = this.lut;

    for (let i = 0; i < FREQ_BINS; i++) {
      const { binLo, binHi } = this.melMapping[i];
      let sum = 0, count = 0;
      for (let b = binLo; b < binHi && b < magnitudes.length; b++) {
        sum += magnitudes[b];
        count++;
      }
      const avg = count > 0 ? sum / count : 0;
      const db = avg > 0 ? 20 * Math.log10(avg) : DB_MIN;
      const norm = Math.max(0, Math.min(1, (db - DB_MIN) / (DB_MAX - DB_MIN)));
      const li = Math.round(norm * 255);

      px[i * 4]     = lut[li * 3];
      px[i * 4 + 1] = lut[li * 3 + 1];
      px[i * 4 + 2] = lut[li * 3 + 2];
      px[i * 4 + 3] = 255;
    }

    this._offCtx.putImageData(this._rowData, 0, this._writeRow);
    this._writeRow = (this._writeRow + 1) % TIME_ROWS;
    this._totalRows++;
  }

  render() {
    const { w, h } = this._setupDpi();
    const ctx = this.ctx;

    ctx.fillStyle = Black;
    ctx.fillRect(0, 0, w, h);
    ctx.imageSmoothingEnabled = false;

    // Draw the waterfall: newest row at bottom of screen.
    // The offscreen buffer is a circular buffer with _writeRow pointing to the next write slot.
    // Rows from _writeRow..TIME_ROWS-1 are oldest, 0.._writeRow-1 are newest.
    // We draw them top-to-bottom (old at top, new at bottom).
    // Freq is on X-axis (left=low, right=high).

    const filled = Math.min(this._totalRows, TIME_ROWS);
    if (filled === 0) return;

    const rowH = h / TIME_ROWS;

    if (this._totalRows >= TIME_ROWS) {
      // Full buffer: draw old portion (from _writeRow to end) at top
      const oldRows = TIME_ROWS - this._writeRow;
      if (oldRows > 0) {
        ctx.drawImage(this._off,
          0, this._writeRow, FREQ_BINS, oldRows,  // src
          0, 0, w, oldRows * rowH                  // dest
        );
      }
      // Then new portion (from 0 to _writeRow) at bottom
      if (this._writeRow > 0) {
        ctx.drawImage(this._off,
          0, 0, FREQ_BINS, this._writeRow,                        // src
          0, oldRows * rowH, w, this._writeRow * rowH             // dest
        );
      }
    } else {
      // Partial fill: draw available rows at bottom of screen
      const startY = h - filled * rowH;
      ctx.drawImage(this._off,
        0, 0, FREQ_BINS, filled,         // src
        0, startY, w, filled * rowH      // dest
      );
    }

    // ── Overlays ──
    this._drawEdgeFades(ctx, w, h);
    this._drawBandBoundaries(ctx, w, h);
    this._drawNoise(ctx, w, h);
    this._drawScanlines(ctx, w, h);

    // Border
    ctx.strokeStyle = rgba(TextDim, 0.4);
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, w, h);
  }

  _drawNoise(ctx, w, h) {
    if (!this._noisePattern) {
      const nc = document.createElement('canvas');
      nc.width = 64; nc.height = 64;
      const nctx = nc.getContext('2d');
      const nd = nctx.createImageData(64, 64);
      for (let i = 0; i < nd.data.length; i += 4) {
        const v = Math.random() * 18;
        nd.data[i] = 0; nd.data[i+1] = v; nd.data[i+2] = v; nd.data[i+3] = 10;
      }
      nctx.putImageData(nd, 0, 0);
      this._noisePattern = ctx.createPattern(nc, 'repeat');
    }
    ctx.save();
    ctx.globalAlpha = 0.4;
    ctx.fillStyle = this._noisePattern;
    ctx.fillRect(0, 0, w, h);
    ctx.restore();
  }

  _drawEdgeFades(ctx, w, h) {
    const fV = h * 0.08, fH = w * 0.05;

    // Top
    const tg = ctx.createLinearGradient(0, 0, 0, fV);
    tg.addColorStop(0, rgba(Black, 0.85));
    tg.addColorStop(1, rgba(Black, 0));
    ctx.fillStyle = tg;
    ctx.fillRect(0, 0, w, fV);

    // Bottom
    const bg = ctx.createLinearGradient(0, h - fV, 0, h);
    bg.addColorStop(0, rgba(Black, 0));
    bg.addColorStop(1, rgba(Black, 0.85));
    ctx.fillStyle = bg;
    ctx.fillRect(0, h - fV, w, fV);

    // Left
    const lg = ctx.createLinearGradient(0, 0, fH, 0);
    lg.addColorStop(0, rgba(Black, 0.8));
    lg.addColorStop(1, rgba(Black, 0));
    ctx.fillStyle = lg;
    ctx.fillRect(0, 0, fH, h);

    // Right
    const rg = ctx.createLinearGradient(w - fH, 0, w, 0);
    rg.addColorStop(0, rgba(Black, 0));
    rg.addColorStop(1, rgba(Black, 0.8));
    ctx.fillStyle = rg;
    ctx.fillRect(w - fH, 0, fH, h);
  }

  _drawBandBoundaries(ctx, w, h) {
    // Band boundaries as vertical lines (freq on X-axis)
    const subMidX = (this._subMidBin / FREQ_BINS) * w;
    const midHiX = (this._midHiBin / FREQ_BINS) * w;

    ctx.save();
    ctx.setLineDash([]);
    ctx.globalAlpha = 0.15;
    ctx.strokeStyle = AccentDim;
    ctx.lineWidth = 1;

    ctx.beginPath();
    ctx.moveTo(subMidX, 0); ctx.lineTo(subMidX, h);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(midHiX, 0); ctx.lineTo(midHiX, h);
    ctx.stroke();

    ctx.restore();

    // Labels at bottom
    const fs = Math.max(8, Math.min(11, w * 0.03));
    ctx.font = `${fs}px 'Share Tech Mono', monospace`;
    ctx.fillStyle = rgba(AccentDim, 0.35);
    ctx.textBaseline = 'bottom';
    const ly = h - 4;

    ctx.textAlign = 'center';
    ctx.fillText('SUB', subMidX / 2, ly);
    ctx.fillText('MID', (subMidX + midHiX) / 2, ly);
    ctx.fillText('HI', (midHiX + w) / 2, ly);
  }

  _drawScanlines(ctx, w, h) {
    ctx.fillStyle = rgba(Black, 0.1);
    for (let y = 0; y < h; y += 3) {
      ctx.fillRect(0, y, w, 1);
    }
  }

  _setupDpi() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    return { w, h };
  }
}
