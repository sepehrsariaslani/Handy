/** Capture mono microphone audio, resample to 16k PCM16, split at voice pauses or 5s.

 * AudioWorklet is preferred; deprecated ScriptProcessor is a compatibility fallback.
 * The WebView must grant microphone permissions separately from Handy's Rust recorder.
 */
export class LiveAudioCapture {
  private stream: MediaStream | null = null;
  private context: AudioContext | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private processor: AudioNode | null = null;
  private workletUrl: string | null = null;
  private total: number[] = [];
  private sum = 0;
  private weight = 0;
  private silenceSamples = 0;
  private readonly sampleRate = 16000;
  private readonly maxSamples = 16000 * 5;

  constructor(
    private readonly onChunk: (bytes: ArrayBuffer) => void,
    private readonly onLevel?: (level: number) => void,
  ) {}

  private emitChunk(): void {
    if (this.total.length < 1600) return;
    const pcm = new Int16Array(this.total);
    this.total = [];
    this.silenceSamples = 0;
    this.onChunk(pcm.buffer);
  }

  private receiveSamples(input: Float32Array): void {
    const context = this.context;
    if (!context) return;
    const samplesPerTarget = context.sampleRate / this.sampleRate;
    // Weighted averaging is a simple low-pass resampler, using continuous
    // fractional position across AudioWorklet frames (e.g. 44.1k -> 16k).
    let energy = 0;
    let produced = 0;
    for (let i = 0; i < input.length; i++) {
      const sample = Math.max(-1, Math.min(1, input[i]));
      let remaining = 1;
      while (remaining > 1e-8) {
        const take = Math.min(remaining, samplesPerTarget - this.weight);
        this.sum += sample * take;
        this.weight += take;
        remaining -= take;
        if (this.weight + 1e-8 >= samplesPerTarget) {
          const normalized = this.sum / this.weight;
          energy += normalized * normalized;
          produced++;
          this.total.push(
            normalized < 0
              ? Math.round(normalized * 32768)
              : Math.round(normalized * 32767),
          );
          this.weight = 0;
          this.sum = 0;
        }
      }
    }
    const rms = produced ? Math.sqrt(energy / produced) : 0;
    this.onLevel?.(Math.min(1, rms));
    this.silenceSamples = rms < 0.007 ? this.silenceSamples + produced : 0;
    if (
      this.total.length >= this.maxSamples ||
      (this.total.length >= 16000 * 1.2 && this.silenceSamples >= 16000 * 0.65)
    ) {
      this.emitChunk();
    }
  }

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    try {
      this.context = new AudioContext();
      this.source = this.context.createMediaStreamSource(this.stream);
      const silent = this.context.createGain();
      silent.gain.value = 0;
      silent.connect(this.context.destination);

      if (this.context.audioWorklet) {
        const workletCode = [
          "class HandyCaptureProcessor extends AudioWorkletProcessor {",
          " constructor() { super(); this.frames = new Float32Array(4096); this.offset = 0; }",
          " process(inputs) {",
          "  const channels = inputs[0]; const input = channels && channels[0];",
          "  if (input) for (let i=0;i<input.length;i++) {",
          "    this.frames[this.offset++] = input[i];",
          "    if(this.offset === this.frames.length) {",
          "      this.port.postMessage(this.frames, [this.frames.buffer]);",
          "      this.frames = new Float32Array(4096); this.offset=0;",
          "    }",
          "  } return true;",
          " }",
          "}",
          "registerProcessor('handy-live-capture', HandyCaptureProcessor);",
        ].join("\n");
        this.workletUrl = URL.createObjectURL(
          new Blob([workletCode], { type: "text/javascript" }),
        );
        await this.context.audioWorklet.addModule(this.workletUrl);
        const processor = new AudioWorkletNode(
          this.context,
          "handy-live-capture",
        );
        processor.port.onmessage = (event: MessageEvent<Float32Array>) => {
          if (event.data instanceof Float32Array)
            this.receiveSamples(event.data);
        };
        this.processor = processor;
      } else {
        const processor = this.context.createScriptProcessor(4096, 1, 1);
        processor.onaudioprocess = (event) => {
          this.receiveSamples(event.inputBuffer.getChannelData(0));
        };
        this.processor = processor;
      }
      this.source.connect(this.processor);
      this.processor.connect(silent);
      await this.context.resume();
    } catch (error) {
      await this.stop(false);
      throw error;
    }
  }

  async stop(flush = true): Promise<void> {
    this.source?.disconnect();
    this.processor?.disconnect();
    this.stream?.getTracks().forEach((track) => track.stop());
    await this.context?.close().catch(() => undefined);
    if (flush) this.emitChunk();
    this.stream = null;
    this.source = null;
    this.processor = null;
    this.context = null;
    if (this.workletUrl) URL.revokeObjectURL(this.workletUrl);
    this.workletUrl = null;
    this.total = [];
    this.sum = 0;
    this.weight = 0;
  }
}
