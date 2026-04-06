<script>
  let { entry, onrecall = null, ondelete = null, onrename = null } = $props();

  let editing = $state(false);
  let editName = $state("");

  function startEdit() {
    editName = entry.name;
    editing = true;
  }

  function finishEdit() {
    editing = false;
    if (editName.trim() && editName !== entry.name) {
      onrename?.(entry.id, editName.trim());
    }
  }

  function handleKeydown(e) {
    if (e.key === "Enter") finishEdit();
    if (e.key === "Escape") editing = false;
  }

  let timeStr = $derived(
    new Date(entry.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
  );
</script>

<div class="stash-entry">
  {#if editing}
    <input
      class="edit-input"
      bind:value={editName}
      onblur={finishEdit}
      onkeydown={handleKeydown}
    />
  {:else}
    <span class="entry-name" ondblclick={startEdit} role="button" tabindex="0" title="Double-click to rename">{entry.name}</span>
  {/if}
  <span class="entry-time">{timeStr}</span>
  <button class="btn-xs" onclick={(e) => onrecall?.(entry, e.shiftKey)}>Load</button>
  <button class="btn-xs btn-danger" onclick={() => ondelete?.(entry.id)}>X</button>
</div>

<style>
  .stash-entry {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 3px 4px;
    border-radius: 3px;
  }
  .stash-entry:hover {
    background: var(--bg-input);
  }
  .entry-name {
    flex: 1;
    font-size: 11px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    cursor: default;
  }
  .entry-time {
    font-size: 9px;
    color: var(--fg-dim);
  }
  .edit-input {
    flex: 1;
    background: var(--bg-input);
    border: 1px solid var(--accent);
    color: var(--fg);
    font-size: 11px;
    padding: 1px 4px;
    border-radius: 2px;
    outline: none;
  }
  .btn-xs {
    padding: 1px 6px;
    font-size: 10px;
    background: var(--bg-input);
    color: var(--fg-dim);
    border: 1px solid var(--border);
    border-radius: 2px;
    cursor: pointer;
  }
  .btn-xs:hover { background: var(--accent-dim); color: var(--fg); }
  .btn-danger:hover { background: #e74c3c; }
</style>
