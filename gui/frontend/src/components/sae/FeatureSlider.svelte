<script>
  import Slider from "../common/Slider.svelte";

  let { index, angleVariance = 0, subspace = "", annotation = "", angleDistribution = null, value = $bindable(0), onchange = null, onannotate = null } = $props();

  // Compute asymmetric scale factors from distribution:
  // posScale = weighted avg of positive angles, negScale = weighted avg of |negative angles|
  // Slider ±1 corresponds to these averages.
  let scales = $derived.by(() => {
    if (!angleDistribution || Object.keys(angleDistribution).length === 0) {
      return { neg: 1, pos: 1 };
    }
    let posSum = 0, posCount = 0, negSum = 0, negCount = 0;
    for (const [key, count] of Object.entries(angleDistribution)) {
      const angle = Number(key);
      if (angle > 0) { posSum += angle * count; posCount += count; }
      else if (angle < 0) { negSum += -angle * count; negCount += count; }
    }
    return {
      neg: negCount > 0 ? negSum / negCount : 1,
      pos: posCount > 0 ? posSum / posCount : 1,
    };
  });

  // Raw angle → normalized slider value
  function toNorm(raw) {
    if (raw >= 0) return scales.pos > 0 ? raw / scales.pos : raw;
    return scales.neg > 0 ? -((-raw) / scales.neg) : raw;
  }

  // Normalized slider value → raw angle
  function toRaw(norm) {
    if (norm >= 0) return norm * scales.pos;
    return -((-norm) * scales.neg);
  }

  let normValue = $derived(toNorm(value));

  // Slider range: convert raw distribution extents to normalized space, with padding
  let sliderMin = $derived.by(() => {
    if (!angleDistribution || Object.keys(angleDistribution).length === 0) return -3;
    const keys = Object.keys(angleDistribution).map(Number);
    return toNorm(Math.min(...keys)) - 0.5;
  });
  let sliderMax = $derived.by(() => {
    if (!angleDistribution || Object.keys(angleDistribution).length === 0) return 3;
    const keys = Object.keys(angleDistribution).map(Number);
    return toNorm(Math.max(...keys)) + 0.5;
  });

  let editing = $state(false);
  let editText = $state("");

  function handleChange(v) {
    const raw = toRaw(v);
    value = raw;
    onchange?.(index, raw);
  }

  function startEdit() {
    editText = annotation;
    editing = true;
  }

  function commitEdit() {
    editing = false;
    const trimmed = editText.trim();
    if (trimmed !== annotation) {
      onannotate?.(String(index), trimmed);
    }
  }

  function onKeydown(e) {
    if (e.key === "Enter") commitEdit();
    if (e.key === "Escape") { editing = false; }
  }
</script>

<div class="feature-slider">
  <div class="feature-header">
    <span class="feature-idx">#{index}</span>
    <span class="feature-imp" title="angle variance">{angleVariance.toFixed(3)}</span>
    <span class="feature-sub" title="subspace">{subspace}</span>
    {#if editing}
      <!-- svelte-ignore a11y_autofocus -->
      <input
        class="annot-input"
        type="text"
        bind:value={editText}
        onblur={commitEdit}
        onkeydown={onKeydown}
        autofocus
      />
    {:else}
      <span
        class="feature-annot"
        class:has-text={!!annotation}
        ondblclick={startEdit}
        title={annotation || "Double-click to annotate"}
      >{annotation || "\u00b7"}</span>
    {/if}
  </div>
  <Slider value={normValue} min={sliderMin} max={sliderMax} step={0.01} notch={0} onchange={handleChange} />
</div>

<style>
  .feature-slider {
    padding: 2px 0;
  }
  .feature-header {
    display: flex;
    gap: 6px;
    align-items: center;
    font-size: 10px;
    color: var(--fg-dim);
    margin-bottom: 1px;
  }
  .feature-idx {
    color: var(--accent);
    font-weight: 600;
    min-width: 36px;
  }
  .feature-imp, .feature-sub {
    font-variant-numeric: tabular-nums;
  }
  .feature-annot {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--fg-dim);
    opacity: 0.3;
    cursor: text;
    font-size: 10px;
  }
  .feature-annot.has-text {
    color: var(--success);
    opacity: 1;
  }
  .annot-input {
    flex: 1;
    min-width: 0;
    font-size: 10px;
    padding: 0 3px;
    background: var(--bg-input);
    color: var(--fg);
    border: 1px solid var(--accent);
    border-radius: 2px;
    outline: none;
  }
</style>
