/**
 * BpmGauge — 270-degree arc gauge with CRT phosphor aesthetic
 * Matches the Android BPMredux Compose Canvas gauge.
 */
import { Black, Surface, Accent, AccentDim, AccentSubtle, TextPrimary, TextSecondary, TextDim, rgba } from './colors.js';

const BPM_MIN = 60;
const BPM_MAX = 240;
const ARC_DEG = 270;
// Arc starts at bottom-left (225°), sweeps 270° clockwise to bottom-right (135°)
const ARC_START = (90 + (360 - ARC_DEG) / 2) * Math.PI / 180;
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

  render(state) {
    const { w, h } = this._setupDpi();
    const ctx = this.ctx;
    const { bpm = 0, confidence = 0, rangeMin = 120, rangeMax = 180,
            isAtRangeLimit = false, limitSide, peakFreq } = state;

    // Smooth BPM animation
    const now = performance.now();
    const dt = Math.min((now - (this._lastTime || now)) / 1000, 0.1);
    this._lastTime = now;
    this._animBpm += (bpm - this._animBpm) * Math.min(1, 8 * dt);
    const displayBpm = this._animBpm;

    // Clear
    ctx.fillStyle = Black;
    ctx.fillRect(0, 0, w, h);

    // Gauge geometry — center the arc, use most of available space
    const cx = w / 2;
    const cy = h * 0.46;
    const radius = Math.min(w * 0.42, h * 0.38);
    const lineW = Math.max(2, radius * 0.03);

    // ── Range highlight arc (thick, subtle) ──
    ctx.beginPath();
    ctx.arc(cx, cy, radius, bpmToAngle(rangeMin), bpmToAngle(rangeMax));
    ctx.strokeStyle = rgba(AccentSubtle, 0.7);
    ctx.lineWidth = lineW * 4;
    ctx.lineCap = 'butt';
    ctx.stroke();

    // ── Background arc (thin) ──
    ctx.beginPath();
    ctx.arc(cx, cy, radius, ARC_START, ARC_END);
    ctx.strokeStyle = rgba(TextDim, 0.5);
    ctx.lineWidth = lineW;
    ctx.lineCap = 'round';
    ctx.stroke();

    // ── Tick marks every 20 BPM ──
    for (let bpmTick = BPM_MIN; bpmTick <= BPM_MAX; bpmTick += 20) {
      const angle = bpmToAngle(bpmTick);
      const isMajor = (bpmTick % 60 === 0);
      const tickInner = radius - (isMajor ? lineW * 6 : lineW * 3);
      const tickOuter = radius + (isMajor ? lineW * 2.5 : lineW * 1.5);

      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(angle) * tickInner, cy + Math.sin(angle) * tickInner);
      ctx.lineTo(cx + Math.cos(angle) * tickOuter, cy + Math.sin(angle) * tickOuter);
      ctx.strokeStyle = isMajor ? rgba(TextPrimary, 0.8) : rgba(TextSecondary, 0.5);
      ctx.lineWidth = isMajor ? 2 : 1;
      ctx.stroke();

      // Numeric label at major ticks
      if (isMajor) {
        const labelR = radius - lineW * 10;
        const lx = cx + Math.cos(angle) * labelR;
        const ly = cy + Math.sin(angle) * labelR;
        ctx.fillStyle = rgba(TextSecondary, 0.8);
        ctx.font = `${Math.max(10, radius * 0.11)}px 'Share Tech Mono', monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(bpmTick), lx, ly);
      }
    }

    // ── Indicator dot with phosphor bloom ──
    if (bpm > 0) {
      const dotAngle = bpmToAngle(displayBpm);
      const dotX = cx + Math.cos(dotAngle) * radius;
      const dotY = cy + Math.sin(dotAngle) * radius;
      const dotR = lineW * 2.5;

      // Large bloom
      const bloom3 = ctx.createRadialGradient(dotX, dotY, 0, dotX, dotY, dotR * 7);
      bloom3.addColorStop(0, rgba(Accent, 0.18));
      bloom3.addColorStop(1, rgba(Accent, 0));
      ctx.beginPath();
      ctx.arc(dotX, dotY, dotR * 7, 0, Math.PI * 2);
      ctx.fillStyle = bloom3;
      ctx.fill();

      // Mid bloom
      const bloom2 = ctx.createRadialGradient(dotX, dotY, 0, dotX, dotY, dotR * 3.5);
      bloom2.addColorStop(0, rgba(Accent, 0.4));
      bloom2.addColorStop(1, rgba(Accent, 0));
      ctx.beginPath();
      ctx.arc(dotX, dotY, dotR * 3.5, 0, Math.PI * 2);
      ctx.fillStyle = bloom2;
      ctx.fill();

      // Core bright dot
      ctx.beginPath();
      ctx.arc(dotX, dotY, dotR, 0, Math.PI * 2);
      ctx.fillStyle = Accent;
      ctx.fill();
    }

    // ── Large BPM number (centered in arc) ──
    const textAlpha = bpm > 0 ? (0.3 + confidence * 0.7) : 0.2;
    const bpmText = bpm > 0 ? displayBpm.toFixed(1) : '---';
    const suffix = isAtRangeLimit ? '+' : '';

    // Main BPM value — large, inside arc center
    const bpmFontSize = Math.max(28, radius * 0.55);
    ctx.fillStyle = rgba(Accent, textAlpha);
    ctx.font = `${bpmFontSize}px 'Share Tech Mono', monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(bpmText + suffix, cx, cy + bpmFontSize * 0.15);

    // "BPM" label below number
    const labelSize = Math.max(10, radius * 0.12);
    ctx.fillStyle = rgba(TextSecondary, textAlpha * 0.6);
    ctx.font = `${labelSize}px 'Share Tech Mono', monospace`;
    ctx.textBaseline = 'top';
    ctx.fillText('BPM', cx, cy + bpmFontSize * 0.22);

    // ── Confidence dot below BPM label ──
    if (bpm > 0) {
      const dotY = cy + bpmFontSize * 0.22 + labelSize + 6;
      ctx.beginPath();
      ctx.arc(cx, dotY, 3, 0, Math.PI * 2);
      ctx.fillStyle = rgba(Accent, confidence);
      ctx.fill();
    }

    // ── Bottom stats row: FREQ | confidence % ──
    const statsY = h - 10;
    const statsFont = Math.max(9, Math.min(12, h * 0.04));
    ctx.font = `${statsFont}px 'Share Tech Mono', monospace`;
    ctx.textBaseline = 'bottom';

    // Peak freq (left)
    if (peakFreq != null && peakFreq > 0) {
      const freqStr = peakFreq >= 1000
        ? (peakFreq / 1000).toFixed(1) + 'kHz'
        : Math.round(peakFreq) + 'Hz';
      ctx.fillStyle = rgba(AccentDim, 0.7);
      ctx.textAlign = 'left';
      ctx.fillText('FREQ ' + freqStr, 10, statsY);
    }

    // Confidence (right)
    const confPct = Math.round(confidence * 100);
    ctx.fillStyle = rgba(AccentDim, 0.7);
    ctx.textAlign = 'right';
    ctx.fillText(confPct + '%', w - 10, statsY);

    // ── Scanline overlay ──
    ctx.fillStyle = rgba(Black, 0.07);
    for (let y = 0; y < h; y += 3) {
      ctx.fillRect(0, y, w, 1);
    }
  }
}
