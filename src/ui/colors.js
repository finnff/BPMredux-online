// colors.js - Color constants and helpers for BPMredux CRT aesthetic

export const Black = '#000000';
export const Surface = '#030808';
export const Accent = '#00E5FF';
export const AccentDim = '#00889A';
export const AccentSubtle = '#0D2A2D';
export const TextPrimary = '#7FCFD6';
export const TextSecondary = '#2A6B6E';
export const TextDim = '#163538';

/**
 * Convert hex color + alpha to rgba() string.
 * @param {string} hex - e.g. '#00E5FF'
 * @param {number} alpha - 0..1
 * @returns {string} rgba(r,g,b,a)
 */
export function rgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

/**
 * Build a 256-entry teal colormap LUT for spectrogram rendering.
 * Maps 0..255 -> [r, g, b] using gamma curve t*t.
 * black -> teal (#00889A) -> cyan (#00E5FF)
 */
export function buildTealLut() {
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255);
    const g = t * t; // gamma curve
    let r, gr, b;
    if (g < 0.5) {
      // black -> teal (#00889A)
      const s = g / 0.5;
      r = 0;
      gr = Math.round(0x88 * s);
      b = Math.round(0x9A * s);
    } else {
      // teal (#00889A) -> cyan (#00E5FF)
      const s = (g - 0.5) / 0.5;
      r = 0;
      gr = Math.round(0x88 + (0xE5 - 0x88) * s);
      b = Math.round(0x9A + (0xFF - 0x9A) * s);
    }
    lut[i * 3] = r;
    lut[i * 3 + 1] = gr;
    lut[i * 3 + 2] = b;
  }
  return lut;
}
