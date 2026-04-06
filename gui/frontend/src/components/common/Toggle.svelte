<script>
  let { checked = $bindable(false), label = "", onchange = null } = $props();

  function handleChange(e) {
    checked = e.target.checked;
    onchange?.(checked);
  }
</script>

<label class="toggle">
  <input type="checkbox" checked={checked} onchange={handleChange} />
  <span class="track"><span class="thumb"></span></span>
  {#if label}<span class="toggle-label">{label}</span>{/if}
</label>

<style>
  .toggle {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    user-select: none;
  }
  input { display: none; }
  .track {
    width: 28px;
    height: 16px;
    border-radius: 8px;
    background: var(--slider-track);
    position: relative;
    transition: background 0.15s;
  }
  input:checked + .track {
    background: var(--accent);
  }
  .thumb {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--fg);
    transition: transform 0.15s;
  }
  input:checked + .track .thumb {
    transform: translateX(12px);
  }
  .toggle-label {
    font-size: 11px;
    color: var(--fg-dim);
  }
</style>
