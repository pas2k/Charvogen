<script>
  import AudioPlayer from "./AudioPlayer.svelte";
  import { ttsState, removeJob, clearFinishedJobs } from "../../lib/stores.svelte.js";

  let hasFinished = $derived(
    ttsState.jobs.some(j => j.status === "done" || j.status === "error")
  );

  let editingId = $state(null);
  let editText = $state("");

  function startEdit(job) {
    editingId = job.job_id;
    editText = job.label || "";
  }

  function commitEdit(job) {
    job.label = editText.trim();
    editingId = null;
  }

  function onKeydown(e, job) {
    if (e.key === "Enter") commitEdit(job);
    if (e.key === "Escape") { editingId = null; }
  }
</script>

<div class="queue">
  {#if ttsState.jobs.length === 0}
    <p class="muted">No jobs yet</p>
  {:else}
    {#if hasFinished}
      <button class="btn-sm clear-btn" onclick={clearFinishedJobs}>Clear finished</button>
    {/if}
    {#each ttsState.jobs as job (job.job_id)}
      <div class="job" class:running={job.status === "running"} class:done={job.status === "done"} class:error={job.status === "error"}>
        <span class="job-status">{job.status}</span>
        <div class="job-info">
          {#if editingId === job.job_id}
            <!-- svelte-ignore a11y_autofocus -->
            <input
              class="label-input"
              type="text"
              placeholder="label..."
              bind:value={editText}
              onblur={() => commitEdit(job)}
              onkeydown={(e) => onKeydown(e, job)}
              autofocus
            />
          {:else}
            <!-- svelte-ignore a11y_no_static_element_interactions -->
            <span class="job-label" class:has-label={!!job.label} ondblclick={() => startEdit(job)}
                  title={job.label || job.text}>
              {job.label || job.text.slice(0, 60)}{!job.label && job.text.length > 60 ? "..." : ""}
            </span>
          {/if}
          {#if job.label}
            <span class="job-text-small" title={job.text}>{job.text.slice(0, 40)}{job.text.length > 40 ? "..." : ""}</span>
          {/if}
        </div>
        {#if job.status === "done" && job.audio_b64}
          <AudioPlayer audioB64={job.audio_b64} filename="tts_{job.job_id}.wav" />
          <span class="job-duration">{job.duration_s?.toFixed(1)}s</span>
        {/if}
        {#if job.status === "error"}
          <span class="job-error" title={job.error}>Error</span>
        {/if}
        {#if job.status === "done" || job.status === "error"}
          <button class="btn-dismiss" onclick={() => removeJob(job.job_id)} title="Remove">&times;</button>
        {/if}
      </div>
    {/each}
  {/if}
</div>

<style>
  .queue {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .job {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 8px;
    border-radius: 4px;
    background: var(--bg);
    font-size: 11px;
  }
  .job.running { border-left: 3px solid var(--accent); }
  .job.done { border-left: 3px solid var(--success); }
  .job.error { border-left: 3px solid #e74c3c; }
  .job-status {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    min-width: 50px;
    color: var(--fg-dim);
  }
  .job-info {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 1px;
  }
  .job-label {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    cursor: text;
  }
  .job-label.has-label {
    font-weight: 600;
    color: var(--accent);
  }
  .job-text-small {
    font-size: 9px;
    color: var(--fg-dim);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .label-input {
    width: 100%;
    font-size: 11px;
    padding: 0 3px;
    background: var(--bg-input);
    color: var(--fg);
    border: 1px solid var(--accent);
    border-radius: 2px;
    outline: none;
  }
  .job-duration {
    color: var(--fg-dim);
    font-size: 10px;
    white-space: nowrap;
  }
  .job-error {
    color: #e74c3c;
    font-size: 10px;
  }
  .btn-dismiss {
    background: none;
    border: none;
    color: var(--fg-dim);
    font-size: 14px;
    cursor: pointer;
    padding: 0 4px;
    line-height: 1;
  }
  .btn-dismiss:hover { color: #e74c3c; }
  .btn-sm {
    padding: 2px 8px;
    font-size: 11px;
    background: var(--bg-input);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 3px;
    cursor: pointer;
  }
  .btn-sm:hover { background: var(--accent-dim); }
  .clear-btn { align-self: flex-end; margin-bottom: 2px; }
  .muted {
    color: var(--fg-dim);
    font-size: 12px;
    padding: 8px;
  }
</style>
