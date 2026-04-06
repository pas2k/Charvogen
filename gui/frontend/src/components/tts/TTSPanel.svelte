<script>
  import Slider from "../common/Slider.svelte";
  import QueuePanel from "./QueuePanel.svelte";
  import { ttsState, embeddingsState, enqueueTTS } from "../../lib/stores.svelte.js";

  let showAdvanced = $state(false);

  let canGenerate = $derived(
    !!embeddingsState.ve && !!ttsState.text.trim() && !ttsState.preparing
  );
</script>

<div class="panel tts-panel">
  <div class="panel-header">
    <h3>TTS Generation</h3>
    {#if ttsState.preparing}
      <span class="status-badge preparing">Preparing voice...</span>
    {/if}
  </div>

  <div class="panel-body">
    <div class="text-input">
      <textarea
        bind:value={ttsState.text}
        placeholder="Enter text to synthesize..."
        rows="3"
      ></textarea>
    </div>

    <div class="tts-params">
      <Slider label="Exaggeration" bind:value={ttsState.exaggeration} min={0} max={2} step={0.05} />
      <Slider label="CFG Weight" bind:value={ttsState.cfgWeight} min={0} max={2} step={0.05} />
      <Slider label="Temperature" bind:value={ttsState.temperature} min={0.1} max={2} step={0.05} />

      <button class="btn-sm" onclick={() => showAdvanced = !showAdvanced}>
        {showAdvanced ? "Hide" : "Show"} Advanced
      </button>

      {#if showAdvanced}
        <Slider label="Rep. Penalty" bind:value={ttsState.repetitionPenalty} min={1} max={2} step={0.05} />
        <Slider label="Min P" bind:value={ttsState.minP} min={0} max={0.5} step={0.01} />
        <Slider label="Top P" bind:value={ttsState.topP} min={0.1} max={1} step={0.05} />
      {/if}
    </div>

    <button
      class="btn-primary generate-btn"
      onclick={enqueueTTS}
      disabled={!canGenerate}
    >
      {ttsState.preparing ? "Preparing..." : "Generate Speech"}
    </button>

    <div class="queue-section">
      <h4>Queue</h4>
      <QueuePanel />
    </div>
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
  .status-badge {
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 3px;
  }
  .status-badge.preparing {
    color: var(--accent);
    background: rgba(52, 152, 219, 0.15);
    animation: pulse 0.8s infinite;
  }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
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
  .panel-body {
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  textarea {
    width: 100%;
    background: var(--bg-input);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px;
    font-family: inherit;
    font-size: 12px;
    resize: vertical;
  }
  textarea:focus {
    outline: 1px solid var(--accent);
  }
  .tts-params {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .generate-btn {
    width: 100%;
    padding: 8px;
    font-size: 13px;
  }
  .queue-section {
    margin-top: 8px;
  }
  .queue-section h4 {
    font-size: 12px;
    color: var(--fg-dim);
    margin-bottom: 4px;
  }
</style>
