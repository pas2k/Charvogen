<script>
  import MelHeatmap from "./MelHeatmap.svelte";
  import MelEQ from "./MelEQ.svelte";
  import Slider from "../common/Slider.svelte";
  import { melState, generateMel, vocodeMel, referenceState, applyRefMel, ttsState } from "../../lib/stores.svelte.js";
  import { playB64Audio, stop, downloadB64Audio } from "../../lib/audio.js";

  let playing = $state(false);

  async function handlePlay() {
    if (!melState.audioB64) return;
    playing = true;
    try {
      await playB64Audio(melState.audioB64);
    } finally {
      playing = false;
    }
  }

  function handleStop() {
    stop();
    playing = false;
  }
</script>

<div class="panel mel-panel">
  <div class="panel-header">
    <h3>Mel Spectrogram</h3>
    {#if referenceState.loaded}
      <button class="btn-sm btn-ref" onclick={applyRefMel}>Apply Ref</button>
    {/if}
    <MelEQ bind:mel={melState.mel} onchange={() => { melState.audioB64 = null; ttsState.conditionalsId = null; }} />
    <button class="btn-primary" onclick={generateMel} disabled={melState.generating}>
      {melState.generating ? "Generating..." : "Generate"}
    </button>
  </div>

  <div class="panel-body">
    <div class="mel-params">
      <Slider label="Frames" bind:value={melState.nFrames} min={64} max={512} step={16} />
      <Slider label="DDIM Steps" bind:value={melState.ddimSteps} min={10} max={100} step={5} />
      <Slider label="Guidance" bind:value={melState.guidanceScale} min={0} max={10} step={0.1} />
      <Slider label="Seed strength" bind:value={melState.seedStrength} min={0} max={0.3} step={0.005} />
    </div>

    <MelHeatmap mel={melState.mel} maxFrames={melState.nFrames} />

    {#if melState.mel && !melState.audioB64}
      <div class="audio-controls">
        <button class="btn-sm" onclick={vocodeMel} disabled={melState.generating}>
          {melState.generating ? "Vocoding..." : "Vocode"}
        </button>
      </div>
    {/if}
    {#if melState.audioB64}
      <div class="audio-controls">
        {#if playing}
          <button class="btn-sm" onclick={handleStop}>Stop</button>
        {:else}
          <button class="btn-sm" onclick={handlePlay}>Play</button>
        {/if}
        <button class="btn-sm" onclick={() => downloadB64Audio(melState.audioB64, "mel_preview.wav")}>Download</button>
      </div>
    {/if}
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
  .btn-primary {
    padding: 4px 12px;
    font-size: 12px;
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
  }
  .btn-primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .btn-primary:hover:not(:disabled) {
    background: var(--accent-dim);
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
  .btn-ref { color: var(--success, #27ae60); border-color: var(--success, #27ae60); }
  .panel-body { padding: 8px; }
  .mel-params {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-bottom: 8px;
  }
  .audio-controls {
    display: flex;
    gap: 6px;
    margin-top: 8px;
  }
</style>
