<script>
  import FeatureGroup from "./FeatureGroup.svelte";
  import StashList from "../stash/StashList.svelte";
  import { saeState, annotationsState, setFeature, scheduleInfer, referenceState, applyRefSAEVE, applyRefSAECP, reloadAnnotations } from "../../lib/stores.svelte.js";
  import { apiPut } from "../../lib/api.js";

  let { subspace = "ve", onshiftrecall = null } = $props();

  const isVE = subspace === "ve";
  const label = isVE ? "VE" : "CP";
  let activations = $derived(isVE ? saeState.veActivations : saeState.cpActivations);
  let annotations = $derived(isVE ? annotationsState.ve : annotationsState.cp);
  let hierarchy = $derived(isVE ? saeState.hierarchy?.ve : saeState.hierarchy?.cp);
  let featuresMeta = $derived(
    Object.fromEntries(((isVE ? saeState.features.ve : saeState.features.cp) || []).map(f => [f.index, f]))
  );
  let activeCount = $derived(Object.keys(activations).length);
  let alivePlanes = $derived(isVE ? saeState.config?.ve?.alive_planes : saeState.config?.cp?.alive_planes);

  function handleFeatureChange(index, value) {
    setFeature(subspace, index, value);
  }

  async function handleAnnotate(key, text) {
    annotations[key] = text;
    try {
      await apiPut(`/api/annotations/${subspace}/${encodeURIComponent(key)}`, { description: text });
    } catch (e) {
      console.error(`Failed to save ${label} annotation:`, e);
    }
  }

  function reset() {
    for (const key of Object.keys(activations)) delete activations[key];
  }

  // Quantization
  let qMin = $state(-1);
  let qMax = $state(1);
  let qSteps = $state(3);

  function quantize() {
    const steps = Math.max(1, Math.round(qSteps));
    const stepSize = steps > 1 ? (qMax - qMin) / (steps - 1) : 0;
    for (const [key, val] of Object.entries(activations)) {
      if (steps === 1) {
        if (qMin === 0) {
          delete activations[key];
        } else {
          activations[key] = qMin;
        }
      } else {
        const snapped = Math.round((val - qMin) / stepSize) * stepSize + qMin;
        const clamped = Math.max(qMin, Math.min(qMax, snapped));
        const rounded = Math.round(clamped * 1000) / 1000;
        if (rounded === 0) {
          delete activations[key];
        } else {
          activations[key] = rounded;
        }
      }
    }
    scheduleInfer();
  }

  function applyRef() {
    if (isVE) applyRefSAEVE(); else applyRefSAECP();
  }

  // Stash helpers
  const stashKey = `sae_${subspace}`;
  function getStashData() {
    return { activations: { ...activations } };
  }
  function getStashName(data) {
    const entries = Object.entries(data.activations || {})
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 3);
    return entries.map(([k, v]) => `#${k}=${v.toFixed(1)}`).join(" ") || "Empty";
  }
  function recallStash(data) {
    for (const key of Object.keys(activations)) delete activations[key];
    for (const [k, v] of Object.entries(data.activations || {})) activations[k] = v;
  }
</script>

<div class="panel sae-panel">
  <div class="panel-header">
    <h3>{label} Planes</h3>
    {#if saeState.config}
      <span class="badge">{activeCount}/{alivePlanes ?? "?"}</span>
    {/if}
    {#if referenceState.loaded}
      <button class="btn-sm btn-ref" onclick={applyRef}>Apply Ref</button>
    {/if}
    <button class="btn-sm" onclick={reset}>Reset</button>
    <button class="btn-sm" onclick={reloadAnnotations} title="Reload labels from server">Reload Labels</button>
  </div>

  <div class="quantize-bar">
    <span class="q-label">Quantize</span>
    <label class="q-field">
      <span>min</span>
      <input type="number" bind:value={qMin} step="0.1" />
    </label>
    <label class="q-field">
      <span>max</span>
      <input type="number" bind:value={qMax} step="0.1" />
    </label>
    <label class="q-field">
      <span>steps</span>
      <input type="number" bind:value={qSteps} step="1" min="1" />
    </label>
    <button class="btn-sm" onclick={quantize}>Apply</button>
  </div>

  <div class="panel-body">
    {#if hierarchy?.root}
      <FeatureGroup
        group={hierarchy.root}
        {activations}
        {annotations}
        {featuresMeta}
        onfeaturechange={handleFeatureChange}
        onannotate={handleAnnotate}
        depth={0}
        {subspace}
      />
    {:else}
      <p class="muted">Loading {label} hierarchy...</p>
    {/if}
  </div>

  <StashList
    storageKey={stashKey}
    label="{label} Planes Stash"
    getCurrentData={getStashData}
    getEntryName={getStashName}
    onrecall={recallStash}
    {onshiftrecall}
  />
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
    gap: 6px;
    padding: 6px 10px;
    border-bottom: 1px solid var(--border);
    background: var(--bg);
    flex-wrap: wrap;
  }
  .panel-header h3 {
    font-size: 13px;
    font-weight: 600;
    flex: 1;
    min-width: 60px;
  }
  .badge {
    font-size: 10px;
    color: var(--fg-dim);
    background: var(--bg-input);
    padding: 2px 6px;
    border-radius: 3px;
  }
  .btn-sm {
    padding: 2px 6px;
    font-size: 10px;
    background: var(--bg-input);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 3px;
    cursor: pointer;
    white-space: nowrap;
  }
  .btn-sm:hover { background: var(--accent-dim); }
  .btn-ref { color: var(--success, #27ae60); border-color: var(--success, #27ae60); }
  .btn-link { text-decoration: none; color: var(--accent, #3b82f6); border-color: var(--accent, #3b82f6); }
  .quantize-bar {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-bottom: 1px solid var(--border);
    background: var(--bg);
    font-size: 10px;
  }
  .q-label {
    color: var(--fg-dim);
    font-weight: 600;
  }
  .q-field {
    display: flex;
    align-items: center;
    gap: 2px;
    color: var(--fg-dim);
  }
  .q-field input {
    width: 38px;
    padding: 1px 3px;
    font-size: 10px;
    background: var(--bg-input);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 2px;
    text-align: center;
  }
  .panel-body {
    overflow-y: auto;
    max-height: 600px;
    padding: 4px;
  }
  .muted {
    color: var(--fg-dim);
    padding: 12px;
    font-size: 12px;
  }
</style>
