<script>
  import StashEntry from "./StashEntry.svelte";
  import { loadStash, addStashEntry, removeStashEntry, renameStashEntry } from "../../lib/stash.js";
  import { uiState, setLastRecalled } from "../../lib/stores.svelte.js";

  let { storageKey, label = "Stash", getCurrentData = null, getEntryName = null, onrecall = null, onshiftrecall = null } = $props();

  let entries = $state([]);

  async function refresh() {
    entries = await loadStash(storageKey);
  }

  $effect(() => {
    uiState.stashVersion;  // dependency tracking — reload when bumped
    refresh();
  });

  async function handleSave() {
    if (!getCurrentData) return;
    const data = await getCurrentData();
    const name = getEntryName ? getEntryName(data) : `Entry ${entries.length + 1}`;
    entries = await addStashEntry(storageKey, name, data);
  }

  async function handleDelete(id) {
    entries = await removeStashEntry(storageKey, id);
  }

  async function handleRename(id, newName) {
    entries = await renameStashEntry(storageKey, id, newName);
  }

  function handleRecall(entry, shiftKey = false) {
    onrecall?.(entry.data);
    setLastRecalled(storageKey, label, entry.name, entry.data);
    if (shiftKey) onshiftrecall?.();
  }
</script>

<div class="stash-list">
  <div class="stash-header">
    <span class="stash-label">{label}</span>
    <button class="btn-xs" onclick={handleSave}>Save</button>
  </div>
  {#if entries.length === 0}
    <p class="muted">No saved entries</p>
  {:else}
    {#each entries as entry (entry.id)}
      <StashEntry
        {entry}
        onrecall={handleRecall}
        ondelete={handleDelete}
        onrename={handleRename}
      />
    {/each}
  {/if}
</div>

<style>
  .stash-list {
    border-top: 1px solid var(--border);
    padding: 4px;
    margin-top: 4px;
  }
  .stash-header {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 4px;
  }
  .stash-label {
    font-size: 10px;
    font-weight: 600;
    color: var(--fg-dim);
    flex: 1;
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
  .muted {
    font-size: 10px;
    color: var(--fg-dim);
    padding: 4px;
  }
</style>
