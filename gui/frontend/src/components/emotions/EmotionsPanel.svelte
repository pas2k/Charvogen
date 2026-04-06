<script>
  import EmotionSlider from "./EmotionSlider.svelte";
  import { emotionsState, setEmotion, toggleEmotionMask, referenceState, applyRefEmotions } from "../../lib/stores.svelte.js";

  function enableAll() {
    for (const name of [...emotionsState.columns, ...emotionsState.metrics]) {
      emotionsState.mask[name] = 1;
    }
  }

  function disableAll() {
    for (const name of [...emotionsState.columns, ...emotionsState.metrics]) {
      emotionsState.mask[name] = 0;
    }
  }

  function resetValues() {
    for (const name of [...emotionsState.columns, ...emotionsState.metrics]) {
      emotionsState.values[name] = 0.5;
    }
  }

  // Expand metric abbreviations
  const metricLabels = {
    sisdr: "SI-SDR (signal quality)",
    pesq: "PESQ (perceptual quality)",
    stoi: "STOI (intelligibility)",
    snr: "SNR (signal-to-noise)",
    reverb: "Reverb",
    speaking_rate: "Speaking Rate",
  };
</script>

<div class="panel emotions-panel">
  <div class="panel-header">
    <h3>Emotions & Metrics</h3>
    {#if referenceState.loaded}
      <button class="btn-sm btn-ref" onclick={applyRefEmotions}>Apply Ref</button>
    {/if}
    <button class="btn-sm" onclick={enableAll}>Enable All</button>
    <button class="btn-sm" onclick={disableAll}>Disable All</button>
    <button class="btn-sm" onclick={resetValues}>Reset</button>
  </div>

  <div class="panel-body">
    <div class="section-label">Emotions ({emotionsState.columns.length})</div>
    {#each emotionsState.columns as name}
      <EmotionSlider
        {name}
        value={emotionsState.values[name] ?? 0.5}
        active={!!emotionsState.mask[name]}
        onchange={setEmotion}
        ontoggle={toggleEmotionMask}
      />
    {/each}

    <div class="section-label">Metrics ({emotionsState.metrics.length})</div>
    {#each emotionsState.metrics as name}
      <EmotionSlider
        {name}
        displayName={metricLabels[name] || name}
        value={emotionsState.values[name] ?? 0.5}
        active={!!emotionsState.mask[name]}
        onchange={setEmotion}
        ontoggle={toggleEmotionMask}
      />
    {/each}
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
    flex-wrap: wrap;
  }
  .panel-header h3 {
    font-size: 13px;
    font-weight: 600;
    flex: 1;
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
  .panel-body {
    overflow-y: auto;
    max-height: 600px;
    padding: 4px 8px;
  }
  .section-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--fg-dim);
    padding: 6px 0 2px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 4px;
  }
</style>
