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
