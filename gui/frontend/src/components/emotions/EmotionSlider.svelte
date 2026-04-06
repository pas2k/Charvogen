<script>
  import Slider from "../common/Slider.svelte";

  let { name, displayName: displayNameProp = null, value = $bindable(0.5), active = $bindable(false), onchange = null, ontoggle = null } = $props();

  function handleValueChange(v) {
    value = v;
    onchange?.(name, v);
  }

  function handleToggle() {
    active = !active;
    ontoggle?.(name);
  }

  // Pretty-print name (use prop if provided, otherwise auto-format)
  let label = $derived(displayNameProp || name.replace(/_/g, " "));
</script>

<div class="emotion-row" class:inactive={!active}>
  <button class="toggle-btn" onclick={handleToggle} title={active ? "Enabled — value overrides the VAE" : "Disabled — VAE ignores this value"}>
    {active ? "\u25c9" : "\u25cb"}
  </button>
  <span class="emotion-name" title={name}>{label}</span>
  <div class="emotion-slider">
    <Slider value={value} min={0} max={1} step={0.01} onchange={handleValueChange} />
  </div>
</div>

<style>
  .emotion-row {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 1px 0;
  }
  .emotion-row.inactive {
    opacity: 0.4;
  }
  .toggle-btn {
    background: none;
    border: none;
    color: var(--accent);
    cursor: pointer;
    font-size: 12px;
    padding: 0 2px;
    line-height: 1;
  }
  .emotion-name {
    flex: 0 0 140px;
    font-size: 11px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--fg-dim);
  }
  .emotion-slider {
    flex: 1;
  }
</style>
