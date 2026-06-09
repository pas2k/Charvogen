/**
 * Web Audio API helpers for playback of base64-encoded WAV.
 */

let _audioCtx = null;
let _currentSource = null;

function getCtx() {
  if (!_audioCtx) _audioCtx = new AudioContext();
  return _audioCtx;
}

/**
 * Play a base64-encoded WAV string.
 * Returns a promise that resolves when playback finishes.
 */
export async function playB64Audio(b64) {
  stop();
  const ctx = getCtx();
  const binary = atob(b64);
  const buf = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) buf[i] = binary.charCodeAt(i);

  const audioBuffer = await ctx.decodeAudioData(buf.buffer);
  const source = ctx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(ctx.destination);
  _currentSource = source;

  return new Promise((resolve) => {
    source.onended = () => {
      _currentSource = null;
      resolve();
    };
    source.start();
  });
}

export function stop() {
  if (_currentSource) {
    try { _currentSource.stop(); } catch {}
    _currentSource = null;
  }
}

/**
 * Encode interleaved-mono Float32 PCM chunks into a 16-bit PCM WAV ArrayBuffer.
 */
function encodeWav(chunks, sampleRate) {
  let length = 0;
  for (const c of chunks) length += c.length;
  const samples = new Float32Array(length);
  let pos = 0;
  for (const c of chunks) { samples.set(c, pos); pos += c.length; }

  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);              // fmt chunk size
  view.setUint16(20, 1, true);               // PCM
  view.setUint16(22, 1, true);               // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);  // byte rate
  view.setUint16(32, 2, true);               // block align
  view.setUint16(34, 16, true);              // bits per sample
  writeStr(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let o = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    o += 2;
  }
  return buffer;
}

/**
 * Microphone recorder that captures raw PCM via Web Audio and encodes a WAV
 * File on stop. Produces audio the backend can decode with soundfile alone —
 * no ffmpeg needed (unlike MediaRecorder's webm/opus output).
 *
 * Usage:
 *   const rec = createWavRecorder();
 *   await rec.start();   // throws if mic denied/unavailable
 *   const file = await rec.stop();  // File("recording.wav", "audio/wav")
 */
export function createWavRecorder() {
  let ctx = null, source = null, processor = null, sink = null, stream = null;
  let chunks = [];
  let sampleRate = 44100;

  async function start() {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    ctx = new AudioContext();
    sampleRate = ctx.sampleRate;
    chunks = [];
    source = ctx.createMediaStreamSource(stream);
    processor = ctx.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (e) => {
      // copy: the underlying buffer is reused across callbacks
      chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    };
    // Route through a muted sink so the graph keeps pulling without echoing
    // the mic to the speakers.
    sink = ctx.createGain();
    sink.gain.value = 0;
    source.connect(processor);
    processor.connect(sink);
    sink.connect(ctx.destination);
  }

  async function stop() {
    if (processor) { processor.onaudioprocess = null; processor.disconnect(); }
    if (source) source.disconnect();
    if (sink) sink.disconnect();
    if (stream) stream.getTracks().forEach((t) => t.stop());
    if (ctx) { try { await ctx.close(); } catch {} }
    const wav = encodeWav(chunks, sampleRate);
    processor = source = sink = stream = ctx = null;
    chunks = [];
    return new File([wav], "recording.wav", { type: "audio/wav" });
  }

  return { start, stop };
}

/**
 * Download base64 WAV as a file.
 */
export function downloadB64Audio(b64, filename = "charvogen.wav") {
  const binary = atob(b64);
  const buf = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) buf[i] = binary.charCodeAt(i);

  const blob = new Blob([buf], { type: "audio/wav" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
