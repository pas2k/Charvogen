<script>
  let { value = $bindable(0), min = 0, max = 1, step = 0.01, label = "", disabled = false, notch = null, onchange = null } = $props();

  let notchPct = $derived(notch != null && min < notch && notch < max
    ? ((notch - min) / (max - min)) * 100
    : null);

  function handleInput(e) {
    value = parseFloat(e.target.value);
    onchange?.(value);
  }
</script>

<div class="slider-row">
  {#if label}
    <span class="slider-label" title={label}>{label}</span>
  {/if}
  <div class="slider-track-wrap">
    {#if notchPct != null}
      <div class="notch" style="left: {notchPct}%"></div>
    {/if}
    <input
      type="range"
      {min} {max} {step}
      value={value}
      {disabled}
      oninput={handleInput}
    />
  </div>
  <span class="slider-value">{value.toFixed(2)}</span>
</div>

<style>
  .slider-row {
    display: flex;
    align-items: center;
    gap: 6px;
    min-height: 24px;
  }
  .slider-label {
    flex: 0 0 auto;
    max-width: 160px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 11px;
    color: var(--fg-dim);
  }
  .slider-track-wrap {
    flex: 1;
    position: relative;
    display: flex;
    align-items: center;
  }
  .notch {
    position: absolute;
    width: 1px;
    height: 10px;
    background: var(--fg-dim);
    opacity: 0.5;
    pointer-events: none;
    z-index: 1;
    translate: -0.5px 0;
  }
  input[type="range"] {
    width: 100%;
    height: 4px;
    -webkit-appearance: none;
    appearance: none;
    background: var(--slider-track);
    border-radius: 2px;
    outline: none;
    cursor: pointer;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--accent);
    cursor: pointer;
  }
  input[type="range"]:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  .slider-value {
    flex: 0 0 40px;
    text-align: right;
    font-size: 11px;
    font-variant-numeric: tabular-nums;
    color: var(--fg-dim);
  }
</style>
