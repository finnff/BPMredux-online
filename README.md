# BPMredux-online

Real-time BPM detection web app ported from Android BPMredux. CRT terminal aesthetic.

## Features

- Mic-based BPM detection
- FFT + spectral flux onset detection
- Autocorrelation tempo estimation
- Tap-tempo blending
- Live mel-scale spectrogram
- Mobile-first responsive design
- CRT scanline/vignette effects

## Quick Start

```
docker compose up -d
```

Visit http://localhost:8080

## Manual Setup

Serve `index.html` with any HTTP server. HTTPS is required for microphone access on mobile.

## Tech

Vanilla JS (no frameworks), Web Audio API, Canvas 2D, nginx + Docker.

## Note

`getUserMedia` requires HTTPS on mobile browsers. Use a reverse proxy with TLS or a tunnel (e.g. ngrok) for mobile testing.
