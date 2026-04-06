<script>
  import { onMount } from "svelte";

  let { mel = null, maxFrames = 0 } = $props();

  let canvas = $state(null);
  let width = $state(512);
  let height = $state(160);

  // --- ITU-R BS.1770-4 K-weighting for LUFS ---
  // Stage 1: Pre-filter (head-effect high shelf), 48 kHz coefficients
  const K1_B = [1.53512485958697, -2.69169618940638, 1.19839281085285];
  const K1_A = [1.0, -1.69065929318241, 0.73248077421585];
  // Stage 2: RLB high-pass
  const K2_B = [1.0, -2.0, 1.0];
  const K2_A = [1.0, -1.99004745483398, 0.99007225036621];

  function biquadMagSq(b, a, f, fs) {
    const w = 2 * Math.PI * f / fs;
    const cosw = Math.cos(w);
    const cos2w = Math.cos(2 * w);
    const sinw = Math.sin(w);
    const sin2w = Math.sin(2 * w);
    const nRe = b[0] + b[1] * cosw + b[2] * cos2w;
    const nIm = -(b[1] * sinw + b[2] * sin2w);
    const dRe = a[0] + a[1] * cosw + a[2] * cos2w;
    const dIm = -(a[1] * sinw + a[2] * sin2w);
    return (nRe * nRe + nIm * nIm) / (dRe * dRe + dIm * dIm);
  }

  // Precompute K-weighting as linear power multiplier per mel bin
  const K_FS = 48000;
  const K_LINEAR = new Float64Array(80);
  for (let m = 0; m < 80; m++) {
    const f = (m + 0.5) * 100; // center freq of bin (0–8 kHz in 80 bins)
    const magSq = biquadMagSq(K1_B, K1_A, f, K_FS) * biquadMagSq(K2_B, K2_A, f, K_FS);
    K_LINEAR[m] = magSq;
  }

  function computeLUFS(data) {
    if (!data || !data[0]) return null;
    const nMels = data.length;
    const T = data[0].length;

    // Per-frame K-weighted power (mel is log domain, power ∝ exp(2*mel))
    const framePower = new Float64Array(T);
    for (let t = 0; t < T; t++) {
      let sum = 0;
      for (let m = 0; m < nMels; m++) {
        sum += Math.exp(2 * data[m][t]) * K_LINEAR[m];
      }
      framePower[t] = sum;
    }

    // Ungated mean
    let ungatedSum = 0;
    for (let t = 0; t < T; t++) ungatedSum += framePower[t];
    const ungatedMean = ungatedSum / T;
    if (ungatedMean <= 0) return null;

    // Relative gating: keep frames ≥ ungatedMean − 10 dB
    const relThreshold = ungatedMean * 0.1;
    let gatedSum = 0;
    let gatedCount = 0;
    for (let t = 0; t < T; t++) {
      if (framePower[t] >= relThreshold) {
        gatedSum += framePower[t];
        gatedCount++;
      }
    }
    if (gatedCount === 0) return null;
    return -0.691 + 10 * Math.log10(gatedSum / gatedCount);
  }

  let lufs = $derived(computeLUFS(mel));

  $effect(() => {
    if (mel && canvas) drawMel(mel);
  });

  function drawMel(data) {
    const ctx = canvas.getContext("2d");
    const nMels = data.length;       // 80
    const totalFrames = data[0].length;
    const nFrames = maxFrames > 0 ? Math.min(totalFrames, maxFrames) : totalFrames;

    // Resize canvas
    canvas.width = nFrames;
    canvas.height = nMels;

    // Find min/max for normalization
    let vmin = Infinity, vmax = -Infinity;
    for (let m = 0; m < nMels; m++) {
      for (let t = 0; t < nFrames; t++) {
        const v = data[m][t];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
      }
    }
    const range = vmax - vmin || 1;

    const imgData = ctx.createImageData(nFrames, nMels);
    for (let m = 0; m < nMels; m++) {
      for (let t = 0; t < nFrames; t++) {
        const norm = (data[m][t] - vmin) / range;
        // Mel bins are low-freq at bottom, but canvas Y=0 is top
        // So flip: row (nMels - 1 - m)
        const row = nMels - 1 - m;
        const idx = (row * nFrames + t) * 4;
        // Inferno-ish colormap
        const r = Math.min(255, norm * 400);
        const g = Math.min(255, Math.max(0, (norm - 0.3) * 500));
        const b = Math.min(255, Math.max(0, (norm - 0.6) * 600));
        imgData.data[idx] = r;
        imgData.data[idx + 1] = g;
        imgData.data[idx + 2] = b;
        imgData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }
</script>

<div class="mel-heatmap">
  {#if mel}
    <canvas
      bind:this={canvas}
      style="width: {width}px; height: {height}px; image-rendering: pixelated;"
    ></canvas>
    <div class="mel-info">{mel.length} mels x {maxFrames > 0 && (mel[0]?.length ?? 0) > maxFrames ? `${maxFrames} / ${mel[0].length}` : (mel[0]?.length ?? 0)} frames{#if lufs != null}, {lufs.toFixed(1)} LUFS{/if}</div>
  {:else}
    <div class="placeholder">No mel spectrogram</div>
  {/if}
</div>

<style>
  .mel-heatmap {
    background: var(--bg);
    border-radius: 4px;
    padding: 4px;
  }
  canvas {
    display: block;
    border-radius: 2px;
  }
  .mel-info {
    font-size: 10px;
    color: var(--fg-dim);
    margin-top: 2px;
  }
  .placeholder {
    height: 80px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--fg-dim);
    font-size: 12px;
  }
</style>
