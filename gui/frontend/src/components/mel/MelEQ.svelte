<script>
  let { mel = $bindable(null), onchange = null } = $props();
  let open = $state(false);
  let volume = $state(0);
  let bands = $state([0, 0, 0, 0, 0, 0, 0, 0]);
  let dragging = false;
  let canvasEl;

  const BAND_LABELS = ["1k", "2k", "3k", "4k", "5k", "6k", "7k", "8k"];
  const DB_TO_LOG = Math.log(10) / 20; // ≈ 0.1151

  function bandFromX(clientX) {
    const rect = canvasEl.getBoundingClientRect();
    const col = Math.floor((clientX - rect.left) / (rect.width / 8));
    return Math.max(0, Math.min(7, col));
  }

  function dbFromY(clientY) {
    const rect = canvasEl.getBoundingClientRect();
    const ratio = (clientY - rect.top) / rect.height;
    const db = 12 - ratio * 24;
    return Math.max(-12, Math.min(12, Math.round(db * 2) / 2));
  }

  function handlePointerDown(e) {
    dragging = true;
    canvasEl.setPointerCapture(e.pointerId);
    bands[bandFromX(e.clientX)] = dbFromY(e.clientY);
  }

  function handlePointerMove(e) {
    if (!dragging) return;
    bands[bandFromX(e.clientX)] = dbFromY(e.clientY);
  }

  function handlePointerUp() {
    dragging = false;
  }

  function drawCanvas() {
    if (!canvasEl) return;
    const ctx = canvasEl.getContext("2d");
    const w = canvasEl.width;
    const h = canvasEl.height;
    const colW = w / 8;
    const centerY = h / 2;

    ctx.clearRect(0, 0, w, h);

    // 0 dB center line
    ctx.strokeStyle = "rgba(128,128,128,0.4)";
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(w, centerY);
    ctx.stroke();
    ctx.setLineDash([]);

    // dB scale markers
    ctx.fillStyle = "rgba(128,128,128,0.5)";
    ctx.font = "9px monospace";
    ctx.textAlign = "left";
    ctx.fillText("+12", 2, 10);
    ctx.fillText("0", 2, centerY - 2);
    ctx.fillText("-12", 2, h - 3);

    // Bands
    const pad = 3;
    for (let i = 0; i < 8; i++) {
      const db = bands[i];
      const x = i * colW + pad;
      const barW = colW - pad * 2;
      const barH = (db / 12) * centerY;

      if (Math.abs(db) > 0.01) {
        ctx.fillStyle = db > 0 ? "rgba(99,179,237,0.7)" : "rgba(237,137,99,0.5)";
        if (db > 0) {
          ctx.fillRect(x, centerY - barH, barW, barH);
        } else {
          ctx.fillRect(x, centerY, barW, -barH);
        }
      }

      // Band label
      ctx.fillStyle = "rgba(180,180,180,0.8)";
      ctx.font = "9px monospace";
      ctx.textAlign = "center";
      ctx.fillText(BAND_LABELS[i], i * colW + colW / 2, h - 2);

      // dB value above/below bar
      if (Math.abs(db) > 0.01) {
        ctx.fillStyle = "var(--fg, #ccc)";
        ctx.fillText(db.toFixed(1), i * colW + colW / 2, db > 0 ? centerY - barH - 3 : centerY - barH + 12);
      }
    }
  }

  $effect(() => {
    // Touch bands array to create dependency
    bands.forEach(() => {});
    drawCanvas();
  });

  function applyEQ() {
    if (!mel) return;
    const nMels = mel.length;
    const T = mel[0].length;
    for (let m = 0; m < nMels; m++) {
      const bandIdx = Math.min(Math.floor(m / 10), 7);
      const totalDB = volume + bands[bandIdx];
      if (Math.abs(totalDB) < 0.01) continue;
      const offset = totalDB * DB_TO_LOG;
      for (let t = 0; t < T; t++) {
        mel[m][t] += offset;
      }
    }
    onchange?.();
    volume = 0;
    bands = [0, 0, 0, 0, 0, 0, 0, 0];
  }

  function resetAll() {
    volume = 0;
    bands = [0, 0, 0, 0, 0, 0, 0, 0];
  }
</script>

<div class="eq-wrap">
  <button class="btn-sm" class:active={open} disabled={!mel}
          onclick={() => open = !open}>
    EQ
  </button>
  {#if open}
    <div class="eq-popup">
      <div class="eq-row">
        <span class="eq-label">Vol</span>
        <input type="range" min={-12} max={12} step={0.5} bind:value={volume} />
        <span class="eq-val">{volume > 0 ? "+" : ""}{volume.toFixed(1)}</span>
      </div>
      <canvas
        bind:this={canvasEl}
        width={200} height={120}
        onpointerdown={handlePointerDown}
        onpointermove={handlePointerMove}
        onpointerup={handlePointerUp}
        class="eq-canvas"
      ></canvas>
      <div class="eq-buttons">
        <button class="btn-sm" onclick={resetAll}>Reset</button>
        <button class="btn-sm btn-apply" onclick={applyEQ}>Apply</button>
      </div>
    </div>
  {/if}
</div>

<style>
  .eq-wrap {
    position: relative;
    display: inline-block;
  }
  .btn-sm {
    padding: 2px 8px;
    font-size: 11px;
    background: var(--bg-input);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 3px;
    cursor: pointer;
  }
  .btn-sm:hover { background: var(--accent-dim); }
  .btn-sm:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-sm.active { border-color: var(--accent); }
  .eq-popup {
    position: absolute;
    z-index: 10;
    top: 100%;
    right: 0;
    margin-top: 4px;
    padding: 8px;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    width: 220px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  }
  .eq-row {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .eq-label {
    font-size: 10px;
    width: 24px;
    color: var(--fg);
    opacity: 0.7;
  }
  .eq-row input[type="range"] {
    flex: 1;
    height: 14px;
    accent-color: var(--accent);
  }
  .eq-val {
    font-size: 10px;
    width: 36px;
    text-align: right;
    font-family: monospace;
    color: var(--fg);
  }
  .eq-canvas {
    cursor: crosshair;
    border-radius: 3px;
    background: var(--bg);
    border: 1px solid var(--border);
    width: 100%;
    height: 120px;
  }
  .eq-buttons {
    display: flex;
    gap: 6px;
    justify-content: flex-end;
  }
  .btn-apply {
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }
  .btn-apply:hover {
    background: var(--accent-dim);
  }
</style>
