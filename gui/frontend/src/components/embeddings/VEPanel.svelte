<script>
  import Toggle from "../common/Toggle.svelte";
  import { embeddingsState, referenceState, applyRefVE } from "../../lib/stores.svelte.js";

  let showAll = $state(false);

  let displayVec = $derived(
    embeddingsState.ve
      ? (showAll ? embeddingsState.ve : embeddingsState.ve.slice(0, 16))
      : null
  );

  let norm = $derived(
    embeddingsState.ve
      ? Math.sqrt(embeddingsState.ve.reduce((s, v) => s + v * v, 0))
      : 0
  );
</script>

<div class="panel embed-panel">
  <div class="panel-header">
    <h3>VE (256d)</h3>
    <Toggle bind:checked={embeddingsState.autoInfer} label="Auto-infer" />
    {#if referenceState.loaded}
      <button class="btn-sm btn-ref" onclick={applyRefVE}>Apply Ref</button>
    {/if}
    {#if embeddingsState.inferring}
      <span class="spinner">...</span>
    {/if}
  </div>

  <div class="panel-body">
    {#if displayVec}
      <div class="vec-info">
        <span>norm: {norm.toFixed(4)}</span>
        <span>min: {Math.min(...embeddingsState.ve).toFixed(4)}</span>
        <span>max: {Math.max(...embeddingsState.ve).toFixed(4)}</span>
      </div>
      <div class="vec-grid">
        {#each displayVec as v, i}
          <span class="vec-val" style="background: rgba(233,69,96,{Math.min(v * 2, 1)})" title="[{i}] = {v.toFixed(4)}">
            {v.toFixed(2)}
          </span>
        {/each}
        {#if !showAll && embeddingsState.ve.length > 16}
          <button class="show-more" onclick={() => showAll = true}>+{embeddingsState.ve.length - 16} more</button>
        {/if}
      </div>
    {:else}
      <p class="muted">No VE embedding yet. Adjust features and run inference.</p>
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
  .spinner {
    color: var(--accent);
    font-size: 11px;
    animation: pulse 0.8s infinite;
  }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
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
  .vec-info {
    display: flex;
    gap: 12px;
    font-size: 10px;
    color: var(--fg-dim);
    margin-bottom: 6px;
  }
  .vec-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 2px;
  }
  .vec-val {
    display: inline-block;
    padding: 1px 3px;
    font-size: 9px;
    font-variant-numeric: tabular-nums;
    border-radius: 2px;
    color: var(--fg);
    min-width: 28px;
    text-align: center;
  }
  .show-more {
    background: var(--bg-input);
    border: 1px solid var(--border);
    color: var(--fg-dim);
    font-size: 9px;
    padding: 1px 6px;
    border-radius: 2px;
    cursor: pointer;
  }
  .muted {
    color: var(--fg-dim);
    font-size: 12px;
  }
</style>
