<script>
  import { t3PromptState, referenceState, uploadT3Prompt, applyRefT3, resetT3ToSeed } from "../../lib/stores.svelte.js";

  let fileInputEl;

  function handleUpload(e) {
    const file = e.target.files?.[0];
    if (file) uploadT3Prompt(file);
    e.target.value = "";
  }

  let hasPrompt = $derived(!!t3PromptState.tokens || !!t3PromptState.audioId);

  let sourceLabel = $derived(
    !hasPrompt ? "None (required)"
    : t3PromptState.source === "stash" ? `Stash${t3PromptState.fileName ? ` (${t3PromptState.fileName})` : ""}`
    : t3PromptState.source === "reference" ? "Reference audio"
    : t3PromptState.fileName || "Custom upload"
  );

  let tokenCount = $derived(t3PromptState.tokens ? t3PromptState.tokens.length : 0);

  // --- Microphone recording ---
  let canRecord = $derived(typeof navigator !== "undefined" && !!navigator.mediaDevices);
  let recording = $state(false);
  let elapsedSec = $state(0);
  let mediaRecorder = null;
  let recordedChunks = [];
  let recStream = null;
  let timerInterval = null;

  function formatTime(sec) {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m}:${String(s).padStart(2, "0")}`;
  }

  async function startRecording() {
    try {
      recStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      return; // user denied or no mic
    }
    recordedChunks = [];
    // prefer webm, fall back to whatever the browser supports
    const mimeType = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "";
    mediaRecorder = new MediaRecorder(recStream, mimeType ? { mimeType } : undefined);
    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      const ext = mediaRecorder.mimeType?.includes("webm") ? "webm" : "ogg";
      const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || "audio/webm" });
      const file = new File([blob], `recording.${ext}`, { type: blob.type });
      uploadT3Prompt(file);
    };
    mediaRecorder.start();
    recording = true;
    elapsedSec = 0;
    timerInterval = setInterval(() => { elapsedSec++; }, 1000);
  }

  function stopRecording() {
    clearInterval(timerInterval);
    timerInterval = null;
    if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
    if (recStream) { recStream.getTracks().forEach(t => t.stop()); recStream = null; }
    recording = false;
  }
</script>

<div class="panel t3-panel">
  <div class="panel-header">
    <h3>T3 Speech Prompt</h3>
  </div>

  <div class="panel-body">
    <div class="source-row">
      <span class="source-label">Source:</span>
      <span class="source-value" class:is-missing={!hasPrompt}>{sourceLabel}</span>
      {#if tokenCount > 0}
        <span class="token-count">({tokenCount} tokens)</span>
      {/if}
    </div>

    <div class="actions">
      <button class="btn-sm" onclick={() => fileInputEl.click()} disabled={t3PromptState.uploading || recording}>
        {t3PromptState.uploading ? "Uploading..." : "Upload"}
      </button>
      {#if canRecord}
        {#if recording}
          <button class="btn-sm btn-rec-stop" onclick={stopRecording}>Stop {formatTime(elapsedSec)}</button>
        {:else}
          <button class="btn-sm btn-rec" onclick={startRecording} disabled={t3PromptState.uploading}>Record</button>
        {/if}
      {/if}
      {#if referenceState.loaded}
        <button class="btn-sm btn-ref" onclick={applyRefT3}>Use Reference</button>
      {/if}
      {#if hasPrompt}
        <button class="btn-sm" onclick={resetT3ToSeed}>Clear</button>
      {/if}
    </div>

    <input bind:this={fileInputEl} type="file" accept="audio/*" onchange={handleUpload} style="display:none" />

    <p class="hint">
      Required. Speech tokens from this audio condition the T3 model's prosody and rhythm.
      Upload any speech sample or apply from a loaded reference.
    </p>
  </div>
</div>

<style>
  .panel {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .panel-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    background: var(--bg);
  }
  .panel-header h3 {
    font-size: 13px;
    font-weight: 600;
    flex: 1;
  }
  .panel-body {
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .source-row {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
  }
  .source-label {
    color: var(--fg-dim);
  }
  .source-value {
    font-weight: 600;
    color: var(--accent);
  }
  .token-count {
    font-size: 10px;
    color: var(--fg-dim);
  }
  .source-value.is-missing {
    color: #e74c3c;
    font-weight: 400;
  }
  .actions {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
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
  .btn-ref { color: var(--success, #27ae60); border-color: var(--success, #27ae60); }
  .btn-rec { color: #e74c3c; border-color: #e74c3c; }
  .btn-rec:hover { background: rgba(231, 76, 60, 0.15); }
  .btn-rec-stop { color: #fff; background: #e74c3c; border-color: #e74c3c; animation: rec-pulse 1s ease-in-out infinite; }
  .btn-rec-stop:hover { background: #c0392b; }
  @keyframes rec-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
  .hint {
    font-size: 10px;
    color: var(--fg-dim);
    line-height: 1.4;
    margin: 0;
  }
</style>
