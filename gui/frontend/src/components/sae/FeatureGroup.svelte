<script>
  import FeatureSlider from "./FeatureSlider.svelte";
  import FeatureGroup from "./FeatureGroup.svelte";

  let { group, activations = {}, annotations = {}, featuresMeta = {}, onfeaturechange = null, onannotate = null, depth = 0, subspace = "ve" } = $props();

  let collapsed = $state(undefined);
  let isCollapsed = $derived(collapsed ?? false);

  let editing = $state(false);
  let editText = $state("");

  function toggle() {
    collapsed = !isCollapsed;
  }

  function getMeta(index) {
    return featuresMeta[index] || {};
  }

  // Support both "planes" and "features" field names
  function getItems(g) {
    return g.planes ?? g.features ?? [];
  }

  function getCount(g) {
    return g.n_planes ?? g.n_features ?? 0;
  }

  // Count active planes/features in this group (leaf items + recursive children)
  function countActive(g) {
    let count = 0;
    for (const item of getItems(g)) {
      if (activations[String(item.index)]) count++;
    }
    if (g.children) {
      for (const child of g.children) {
        count += countActive(child);
      }
    }
    return count;
  }

  let nTotal = $derived(getCount(group));
  let items = $derived(getItems(group));
  let activeInGroup = $derived(countActive(group));

  // Group annotation key: "g:<id>" or fall back to label
  let annotKey = $derived(group.id ? `g:${group.id}` : null);
  let groupAnnotation = $derived(annotKey ? (annotations[annotKey] || "") : "");

  function startEdit(e) {
    e.stopPropagation();
    editText = groupAnnotation || group.label;
    editing = true;
  }

  function commitEdit() {
    editing = false;
    const trimmed = editText.trim();
    if (annotKey && trimmed !== groupAnnotation) {
      onannotate?.(annotKey, trimmed);
    }
  }

  function onKeydown(e) {
    if (e.key === "Enter") commitEdit();
    if (e.key === "Escape") { editing = false; }
  }
</script>

<div class="feature-group" style="margin-left: {depth * 8}px">
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="group-header" onclick={toggle}>
    <span class="chevron">{isCollapsed ? "\u25b6" : "\u25bc"}</span>
    {#if editing}
      <!-- svelte-ignore a11y_autofocus -->
      <input
        class="label-input"
        type="text"
        bind:value={editText}
        onblur={commitEdit}
        onkeydown={onKeydown}
        onclick={(e) => e.stopPropagation()}
        autofocus
      />
    {:else}
      <span class="group-label" ondblclick={startEdit}>{groupAnnotation || group.label}</span>
    {/if}
    <span class="group-count" class:has-active={activeInGroup > 0}>{activeInGroup}/{nTotal}</span>
  </div>

  {#if !isCollapsed}
    <div class="group-body">
      {#if group.children}
        {#each group.children as child}
          <FeatureGroup
            group={child}
            {activations}
            {annotations}
            {featuresMeta}
            {onfeaturechange}
            {onannotate}
            depth={depth + 1}
            {subspace}
          />
        {/each}
      {/if}

      {#if items.length > 0}
        {#each items as feat}
          {@const meta = getMeta(feat.index)}
          <FeatureSlider
            index={feat.index}
            angleVariance={feat.angle_variance ?? meta.angle_variance ?? 0}
            angleDistribution={meta.angle_distribution ?? null}
            subspace={feat.subspace ?? subspace.toUpperCase()}
            annotation={annotations[String(feat.index)] || ""}
            value={activations[String(feat.index)] || 0}
            onchange={onfeaturechange}
            {onannotate}
          />
        {/each}
      {/if}
    </div>
  {/if}
</div>

<style>
  .feature-group {
    border-left: 1px solid var(--border);
  }
  .group-header {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 8px;
    cursor: pointer;
    font-size: 12px;
  }
  .group-header:hover {
    background: var(--bg-input);
  }
  .chevron {
    font-size: 9px;
    color: var(--fg-dim);
    width: 10px;
  }
  .group-label {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    cursor: text;
  }
  .label-input {
    flex: 1;
    min-width: 0;
    font-size: 12px;
    padding: 0 3px;
    background: var(--bg-input);
    color: var(--fg);
    border: 1px solid var(--accent);
    border-radius: 2px;
    outline: none;
  }
  .group-count {
    color: var(--fg-dim);
    font-size: 10px;
    font-variant-numeric: tabular-nums;
  }
  .group-count.has-active {
    color: var(--accent);
    font-weight: 600;
  }
  .group-body {
    padding-left: 4px;
  }
</style>
