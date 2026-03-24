// bpm-display.js - BpmGauge: 270-degree arc gauge with CRT phosphor aesthetic
import { Black, Surface, Accent, AccentDim, AccentSubtle, TextPrimary, TextSecondary, TextDim, rgba } from './colors.js';

const BPM_MIN = 60;
const BPM_MAX = 240;
const ARC_DEG = 270;
const ARC_START = (90 + (360 - ARC_DEG) / 2) * Math.PI / 180; // bottom-left
const ARC_END = ARC_START + ARC_DEG * Math.PI / 180;

function bpmToAngle(bpm) {
  const t = Math.max(0, Math.min(1, (bpm - BPM_MIN) / (BPM_MAX - BPM_MIN)));
  return ARC_START + t * (ARC_END - ARC_START);
}

export class BpmGauge {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this._animBpm = 0;
    this._lastTime = 0;
  }

  /** Set up canvas for high-DPI */
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

  /**
   * @param {object} state - {bpm, confidence, rangeMin, rangeMax, isAtRangeLimit, limitSide, peakFreq}
   */
  render(state) {
    const { w, h } = this._setupDpi();
    const ctx = this.ctx;
    const { bpm = 0, confidence = 0, rangeMin = 60, rangeMax = 240, isAtRangeLimit = false, limitSide, peakFreq } = state;

    // Animate BPM smoothly
    const now = performance.now();
    const dt = Math.min((now - (this._lastTime || now)) / 1000, 0.1);
    this._lastTime = now;
    const lerpSpeed = 8;
    this._animBpm += (bpm - this._animBpm) * Math.min(1, lerpSpeed * dt);
    const displayBpm = this._animBpm;

    // Clear
    ctx.fillStyle = Surface;
    ctx.fillRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h * 0.48;
    const radius = Math.min(w, h) * 0.38;
    const lineW = Math.max(2, radius * 0.04);

    // --- Range highlight arc ---
    const rangeStartAngle = bpmToAngle(rangeMin);
    const rangeEndAngle = bpmToAngle(rangeMax);
    ctx.beginPath();
    ctx.arc(cx, cy, radius, rangeStartAngle, rangeEndAngle);
    ctx.strokeStyle = rgba(AccentSubtle, 0.6);
    ctx.lineWidth = lineW * 3;
    ctx.lineCap = 'butt';
    ctx.stroke();

    // --- Background arc ---
    ctx.beginPath();
    ctx.arc(cx, cy, radius, ARC_START, ARC_END);
    ctx.strokeStyle = rgba(TextDim, 0.4);
    ctx.lineWidth = lineW;
    ctx.lineCap = 'round';
    ctx.stroke();

    // --- Tick marks ---
    for (let bpmTick = BPM_MIN; bpmTick <= BPM_MAX; bpmTick += 20) {
      const angle = bpmToAngle(bpmTick);
      const isMajor = (bpmTick % 60 === 0);
      const innerR = radius - (isMajor ? lineW * 5 : lineW * 3);
      const outerR = radius + (isMajor ? lineW * 2 : lineW);

      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(angle) * innerR, cy + Math.sin(angle) * innerR);
      ctx.lineTo(cx + Math.cos(angle) * outerR, cy + Math.sin(angle) * outerR);
      ctx.strokeStyle = isMajor ? rgba(TextPrimary, 0.7) : rgba(TextSecondary, 0.4);
      ctx.lineWidth = isMajor ? 2 : 1;
      ctx.stroke();

      // Labels for major ticks
      if (isMajor) {
        const labelR = radius - lineW * 8;
        const lx = cx + Math.cos(angle) * labelR;
        const ly = cy + Math.sin(angle) * labelR;
        ctx.fillStyle = rgba(TextSecondary, 0.7);
        ctx.font = `${Math.max(10, radius * 0.1)}px monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(bpmTick), lx, ly);
      }
    }

    // --- Indicator dot with phosphor bloom ---
    if (bpm > 0) {
      const dotAngle = bpmToAngle(displayBpm);
      const dotX = cx + Math.cos(dotAngle) * radius;
      const dotY = cy + Math.sin(dotAngle) * radius;
      const dotR = lineW * 2;

      // Outer bloom
      const bloom3 = ctx.createRadialGradient(dotX, dotY, 0, dotX, dotY, dotR * 6);
      bloom3.addColorStop(0, rgba(Accent, 0.15));
      bloom3.addColorStop(1, rgba(Accent, 0));
      ctx.beginPath();
      ctx.arc(dotX, dotY, dotR * 6, 0, Math.PI * 2);
      ctx.fillStyle = bloom3;
      ctx.fill();

      // Mid bloom
      const bloom2 = ctx.createRadialGradient(dotX, dotY, 0, dotX, dotY, dotR * 3);
      bloom2.addColorStop(0, rgba(Accent, 0.35));
      bloom2.addColorStop(1, rgba(Accent, 0));
      ctx.beginPath();
      ctx.arc(dotX, dotY, dotR * 3, 0, Math.PI * 2);
      ctx.fillStyle = bloom2;
      ctx.fill();

      // Core dot
      ctx.beginPath();
      ctx.arc(dotX, dotY, dotR, 0, Math.PI * 2);
      ctx.fillStyle = Accent;
      ctx.fill();
    }

    // --- Large BPM number ---
    const textAlpha = 0.3 + confidence * 0.7;
    const bpmText = bpm > 0 ? Math.round(displayBpm).toString() : '--';
    const suffix = isAtRangeLimit ? '+' : '';

    ctx.fillStyle = rgba(Accent, textAlpha);
    ctx.font = `bold ${Math.max(24, radius * 0.55)}px monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(bpmText + suffix, cx, cy);

    // "BPM" label
    ctx.fillStyle = rgba(TextSecondary, textAlpha * 0.7);
    ctx.font = `${Math.max(10, radius * 0.13)}px monospace`;
    ctx.fillText('BPM', cx, cy + radius * 0.25);

    // --- Peak frequency label ---
    if (peakFreq != null && peakFreq > 0) {
      const freqStr = peakFreq >= 1000
        ? (peakFreq / 1000).toFixed(1) + ' kHz'
        : Math.round(peakFreq) + ' Hz';
      ctx.fillStyle = rgba(TextDim, 0.7);
      ctx.font = `${Math.max(9, radius * 0.1)}px monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText('peak: ' + freqStr, cx, h - 8);
    }

    // --- CRT scanline overlay ---
    this._drawScanlines(ctx, w, h);
  }

  _drawScanlines(ctx, w, h) {
    ctx.fillStyle = rgba(Black, 0.08);
    for (let y = 0; y < h; y += 3) {
      ctx.fillRect(0, y, w, 1);
    }
  }
}
