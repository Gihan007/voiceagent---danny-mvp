class PcmStreamerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    this.inputSampleRate = sampleRate;
    this.ratio = this.inputSampleRate / this.targetSampleRate;
    this.position = 0;
    this.pending = [];
    this.frame = [];
    this.frameSize = 320;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) return true;

    const channel = input[0];
    let sum = 0;
    for (let i = 0; i < channel.length; i += 1) {
      sum += channel[i] * channel[i];
    }
    this.port.postMessage({ type: "level", level: Math.sqrt(sum / channel.length) });

    for (let i = this.position; i < channel.length; i += this.ratio) {
      const sample = Math.max(-1, Math.min(1, channel[Math.floor(i)] || 0));
      this.frame.push(sample < 0 ? sample * 0x8000 : sample * 0x7fff);

      if (this.frame.length >= this.frameSize) {
        const pcm = new Int16Array(this.frame.length);
        for (let j = 0; j < this.frame.length; j += 1) {
          pcm[j] = this.frame[j];
        }
        this.port.postMessage({ type: "audio", buffer: pcm.buffer }, [pcm.buffer]);
        this.frame = [];
      }
    }

    this.position = (this.position + channel.length) % this.ratio;
    return true;
  }
}

registerProcessor("pcm-streamer", PcmStreamerProcessor);
