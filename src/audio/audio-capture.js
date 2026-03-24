/**
 * AudioCapture - Web Audio API microphone capture with ring buffer
 * Ported from Kotlin BPMredux (Android AudioRecord equivalent)
 */

const SAMPLE_RATE = 44100;
const FFT_SIZE = 4096;
const HOP_SIZE = 1024;

export class AudioCapture {
  /**
   * @param {function(Float32Array): void} onFrame - callback receiving 4096-sample frames
   */
  constructor(onFrame) {
    this.onFrame = onFrame;
    this.audioContext = null;
    this.mediaStream = null;
    this.sourceNode = null;
    this.processorNode = null;

    // Ring buffer: accumulate samples, emit FFT_SIZE frames every HOP_SIZE samples
    this.ringBuffer = new Float32Array(FFT_SIZE);
    this.ringWritePos = 0;       // next write position
    this.samplesUntilEmit = FFT_SIZE; // first frame needs full buffer
    this.isRunning = false;
  }

  /**
   * Start audio capture from microphone.
   */
  async start() {
    if (this.isRunning) return;

    try {
      // Request microphone access
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          sampleRate: SAMPLE_RATE
        }
      });

      // Create AudioContext
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: SAMPLE_RATE
      });

      // Source from mic
      this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);

      // ScriptProcessorNode for sample-level access
      // Buffer size of 1024 to match HOP_SIZE for low latency
      this.processorNode = this.audioContext.createScriptProcessor(HOP_SIZE, 1, 1);

      this.processorNode.onaudioprocess = (event) => {
        if (!this.isRunning) return;

        const inputData = event.inputBuffer.getChannelData(0);
        this.feedSamples(inputData);
      };

      // Connect: mic -> processor -> destination (required for ScriptProcessorNode to fire)
      this.sourceNode.connect(this.processorNode);
      this.processorNode.connect(this.audioContext.destination);

      this.isRunning = true;
    } catch (err) {
      console.error('AudioCapture: Failed to start', err);
      this.stop();
      throw err;
    }
  }

  /**
   * Feed samples into ring buffer and emit frames.
   * @param {Float32Array} samples - incoming audio samples
   */
  feedSamples(samples) {
    for (let i = 0; i < samples.length; i++) {
      this.ringBuffer[this.ringWritePos] = samples[i];
      this.ringWritePos = (this.ringWritePos + 1) % FFT_SIZE;
      this.samplesUntilEmit--;

      if (this.samplesUntilEmit <= 0) {
        // Emit a frame: read FFT_SIZE samples starting from current position
        const frame = new Float32Array(FFT_SIZE);
        for (let j = 0; j < FFT_SIZE; j++) {
          frame[j] = this.ringBuffer[(this.ringWritePos + j) % FFT_SIZE];
        }
        this.onFrame(frame);

        // Next emission after HOP_SIZE samples
        this.samplesUntilEmit = HOP_SIZE;
      }
    }
  }

  /**
   * Stop audio capture and release resources.
   */
  stop() {
    this.isRunning = false;

    if (this.processorNode) {
      this.processorNode.disconnect();
      this.processorNode.onaudioprocess = null;
      this.processorNode = null;
    }

    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }

    if (this.audioContext) {
      this.audioContext.close().catch(() => {});
      this.audioContext = null;
    }

    // Reset ring buffer
    this.ringBuffer.fill(0);
    this.ringWritePos = 0;
    this.samplesUntilEmit = FFT_SIZE;
  }
}

export { SAMPLE_RATE, FFT_SIZE, HOP_SIZE };
