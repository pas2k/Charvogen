<script>
  import { playB64Audio, stop, downloadB64Audio } from "../../lib/audio.js";

  let { audioB64 = null, filename = "charvogen.wav" } = $props();
  let playing = $state(false);

  async function handlePlay() {
    if (!audioB64) return;
    playing = true;
    try {
      await playB64Audio(audioB64);
    } finally {
      playing = false;
    }
  }

  function handleStop() {
    stop();
    playing = false;
  }
</script>

{#if audioB64}
  <div class="audio-player">
    {#if playing}
      <button class="btn" onclick={handleStop}>Stop</button>
    {:else}
      <button class="btn" onclick={handlePlay}>Play</button>
    {/if}
    <button class="btn" onclick={() => downloadB64Audio(audioB64, filename)}>DL</button>
  </div>
{/if}

<style>
  .audio-player {
    display: inline-flex;
    gap: 4px;
  }
  .btn {
    padding: 2px 8px;
    font-size: 11px;
    background: var(--bg-input);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 3px;
    cursor: pointer;
  }
  .btn:hover { background: var(--accent-dim); }
</style>
