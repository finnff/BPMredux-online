# BPMredux-online

Web port of [BPMredux](https://github.com/finnff/BPMredux) — real-time BPM detection from your microphone with a CRT terminal aesthetic.

## Run

```bash
docker compose up -d
# → http://localhost:8080
```

Or serve statically (needs HTTPS for mic on mobile):
```bash
npx serve .
```

## What it does

Listens to your mic, runs FFT → spectral flux onset detection → autocorrelation tempo estimation, and shows BPM on a gauge with a live spectrogram. Also supports tap-tempo that blends with the algorithm.

Verified against librosa: detects Darude - Sandstorm at **136.4 BPM** (librosa: 136.0, Δ 0.4).

## Tests

```bash
node --experimental-vm-modules test/unit/test-dsp.mjs      # 38 unit tests
node --experimental-vm-modules test/e2e/test-e2e-bpm.mjs   # E2E (needs WAV fixture)
python3.12 test/verify_bpm_librosa.py                       # librosa verification
```

## Stack

Vanilla JS · Web Audio API · Canvas 2D · nginx · Docker
