<script>
  let { accept = "audio/*", label = "Upload", onfile = null } = $props();

  let inputEl;

  function handleChange(e) {
    const file = e.target.files?.[0];
    if (file) onfile?.(file);
    e.target.value = "";
  }
</script>

<button class="upload-btn" onclick={() => inputEl.click()}>
  {label}
</button>
<input
  bind:this={inputEl}
  type="file"
  {accept}
  onchange={handleChange}
  style="display:none"
/>

<style>
  .upload-btn {
    padding: 4px 12px;
    background: var(--bg-input);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
  }
  .upload-btn:hover {
    background: var(--accent-dim);
  }
</style>
