<script>
  import { onMount } from "svelte";
  import SAEPanel from "./components/sae/SAEPanel.svelte";
  import EmotionsPanel from "./components/emotions/EmotionsPanel.svelte";
  import VEPanel from "./components/embeddings/VEPanel.svelte";
  import CPPanel from "./components/embeddings/CPPanel.svelte";
  import MelPanel from "./components/mel/MelPanel.svelte";
  import T3PromptPanel from "./components/tts/T3PromptPanel.svelte";
  import TTSPanel from "./components/tts/TTSPanel.svelte";
  import StashList from "./components/stash/StashList.svelte";
  import {
    loadInitialData, runInfer, scheduleInfer, analyzeReference, applyRefAll, enqueueTTS,
    randomK, randomKBatch, randomizeVE, randomizeCP, randomizeEmotions, randomizeMetrics,
    saeState, emotionsState, embeddingsState, melState, ttsState, t3PromptState,
    uiState, referenceState,
    handleWSMessage,
  } from "./lib/stores.svelte.js";
  import { connectTTSWebSocket } from "./lib/api.js";
  import { addStashEntry } from "./lib/stash.js";

  let refInputEl;
  let configInputEl;
  let randomKKappa = $state(100);

  // --- Mel base64 encoding helpers ---

  function melToBase64(mel) {
    if (!mel) return null;
    const rows = mel.length, cols = mel[0].length;
    const buf = new ArrayBuffer(8 + rows * cols * 4);
    const header = new Uint32Array(buf, 0, 2);
    header[0] = rows; header[1] = cols;
    const data = new Float32Array(buf, 8);
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++)
        data[r * cols + c] = mel[r][c];
    const bytes = new Uint8Array(buf);
    let binary = '';
    for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
    return btoa(binary);
  }

  function base64ToMel(b64) {
    if (!b64) return null;
    const binary = atob(b64);
    const buf = new ArrayBuffer(binary.length);
    const bytes = new Uint8Array(buf);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const header = new Uint32Array(buf, 0, 2);
    const rows = header[0], cols = header[1];
    const data = new Float32Array(buf, 8);
    const mel = [];
    for (let r = 0; r < rows; r++)
      mel.push(Array.from(data.subarray(r * cols, (r + 1) * cols)));
    return mel;
  }

  // --- Full config save/load ---

  function getFullConfig() {
    return {
      version: 1,
      savedAt: new Date().toISOString(),
      sae: {
        veActivations: { ...saeState.veActivations },
        cpActivations: { ...saeState.cpActivations },
      },
      emotions: {
        values: { ...emotionsState.values },
        mask: { ...emotionsState.mask },
      },
      mel: {
        melB64: melToBase64(melState.mel),
        nFrames: melState.nFrames,
        ddimSteps: melState.ddimSteps,
        guidanceScale: melState.guidanceScale,
        seedStrength: melState.seedStrength,
      },
      t3: {
        tokens: t3PromptState.tokens ? [...t3PromptState.tokens] : null,
        source: t3PromptState.source,
        fileName: t3PromptState.fileName,
      },
      tts: {
        exaggeration: ttsState.exaggeration,
        cfgWeight: ttsState.cfgWeight,
        temperature: ttsState.temperature,
        repetitionPenalty: ttsState.repetitionPenalty,
        minP: ttsState.minP,
        topP: ttsState.topP,
      },
    };
  }

  function recallFullConfig(config) {
    // SAE activations
    if (config.sae) {
      for (const key of Object.keys(saeState.veActivations)) delete saeState.veActivations[key];
      for (const key of Object.keys(saeState.cpActivations)) delete saeState.cpActivations[key];
      if (config.sae.veActivations) {
        for (const [k, v] of Object.entries(config.sae.veActivations)) saeState.veActivations[k] = v;
      }
      if (config.sae.cpActivations) {
        for (const [k, v] of Object.entries(config.sae.cpActivations)) saeState.cpActivations[k] = v;
      }
    }
    // Emotions
    if (config.emotions) {
      if (config.emotions.values) Object.assign(emotionsState.values, config.emotions.values);
      if (config.emotions.mask) Object.assign(emotionsState.mask, config.emotions.mask);
    }
    // Mel — only overwrite spectrogram if present; never overwrite diffuser config
    if (config.mel && config.mel.melB64 != null) {
      melState.mel = base64ToMel(config.mel.melB64);
      melState.audioB64 = null;
      ttsState.conditionalsId = null;
    }
    // T3 tokens — only overwrite if tokens are present
    if (config.t3 && config.t3.tokens != null) {
      recallT3(config.t3);
    }
    // TTS params — never overwrite generation parameters from loaded config
    // Re-infer embeddings from restored SAE + emotions
    scheduleInfer();
  }

  function getConfigName(data) {
    const veKeys = Object.keys(data.sae?.veActivations || {}).length;
    const cpKeys = Object.keys(data.sae?.cpActivations || {}).length;
    return `${veKeys}V ${cpKeys}C ${new Date().toLocaleTimeString()}`;
  }

  function saveConfigToFile() {
    const config = getFullConfig();
    const json = JSON.stringify(config, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `charvogen_config_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  async function loadConfigFromFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const config = JSON.parse(text);
      recallFullConfig(config);
    } catch (err) {
      uiState.error = `Failed to load config: ${err.message}`;
    }
    e.target.value = "";
  }

  // --- Explore mode helpers ---

  function replaceDict(target, source) {
    for (const k of Object.keys(target)) delete target[k];
    for (const [k, v] of Object.entries(source)) target[k] = v;
  }

  function buildFullConfigFromSample(s) {
    return {
      version: 1,
      savedAt: new Date().toISOString(),
      sae: {
        veActivations: { ...s.ve_activations },
        cpActivations: { ...s.cp_activations },
      },
      emotions: {
        values: { ...s.emotions },
        mask: { ...emotionsState.mask },
      },
      mel: {
        melB64: melToBase64(melState.mel),
        nFrames: melState.nFrames,
        ddimSteps: melState.ddimSteps,
        guidanceScale: melState.guidanceScale,
        seedStrength: melState.seedStrength,
      },
      t3: {
        tokens: t3PromptState.tokens ? [...t3PromptState.tokens] : null,
        source: t3PromptState.source,
        fileName: t3PromptState.fileName,
      },
      tts: {
        exaggeration: ttsState.exaggeration,
        cfgWeight: ttsState.cfgWeight,
        temperature: ttsState.temperature,
        repetitionPenalty: ttsState.repetitionPenalty,
        minP: ttsState.minP,
        topP: ttsState.topP,
      },
    };
  }

  async function handleRandomK() {
    if (!uiState.exploreMode) { randomK(randomKKappa); return; }
    embeddingsState.inferring = true;
    uiState.error = null;
    try {
      const samples = await randomKBatch(randomKKappa, uiState.exploreCount);
      for (let i = 0; i < samples.length; i++) {
        const data = buildFullConfigFromSample(samples[i]);
        await addStashEntry("full_config", `Xplore ${i+1}/${samples.length} κ=${randomKKappa}`, data);
      }
      uiState.stashVersion++;
    } catch (e) {
      uiState.error = e.message;
    } finally {
      embeddingsState.inferring = false;
    }
  }

  async function handleRandomVE() {
    if (!uiState.exploreMode) { randomizeVE(randomKKappa); return; }
    const snap = $state.snapshot(saeState.veActivations);
    for (let i = 0; i < uiState.exploreCount; i++) {
      replaceDict(saeState.veActivations, snap);
      randomizeVE(randomKKappa);
      await addStashEntry("sae_ve", `Xplore ${i+1}/${uiState.exploreCount} κ=${randomKKappa}`,
                    { activations: { ...saeState.veActivations } });
    }
    replaceDict(saeState.veActivations, snap);
    uiState.stashVersion++;
  }

  async function handleRandomCP() {
    if (!uiState.exploreMode) { randomizeCP(randomKKappa); return; }
    const snap = $state.snapshot(saeState.cpActivations);
    for (let i = 0; i < uiState.exploreCount; i++) {
      replaceDict(saeState.cpActivations, snap);
      randomizeCP(randomKKappa);
      await addStashEntry("sae_cp", `Xplore ${i+1}/${uiState.exploreCount} κ=${randomKKappa}`,
                    { activations: { ...saeState.cpActivations } });
    }
    replaceDict(saeState.cpActivations, snap);
    uiState.stashVersion++;
  }

  async function handleRandomEmotions() {
    if (!uiState.exploreMode) { randomizeEmotions(randomKKappa); return; }
    const snap = $state.snapshot(emotionsState.values);
    for (let i = 0; i < uiState.exploreCount; i++) {
      replaceDict(emotionsState.values, snap);
      randomizeEmotions(randomKKappa);
      await addStashEntry("emotions", `Xplore ${i+1}/${uiState.exploreCount} κ=${randomKKappa}`,
                    { values: { ...emotionsState.values }, mask: { ...emotionsState.mask } });
    }
    replaceDict(emotionsState.values, snap);
    uiState.stashVersion++;
  }

  async function handleRandomMetrics() {
    if (!uiState.exploreMode) { randomizeMetrics(randomKKappa); return; }
    const snap = $state.snapshot(emotionsState.values);
    for (let i = 0; i < uiState.exploreCount; i++) {
      replaceDict(emotionsState.values, snap);
      randomizeMetrics(randomKKappa);
      await addStashEntry("emotions", `Xplore ${i+1}/${uiState.exploreCount} κ=${randomKKappa} metrics`,
                    { values: { ...emotionsState.values }, mask: { ...emotionsState.mask } });
    }
    replaceDict(emotionsState.values, snap);
    uiState.stashVersion++;
  }

  onMount(async () => {
    await loadInitialData();
    connectTTSWebSocket(handleWSMessage);
  });

  async function recallAndGenerate() {
    await runInfer();
    await enqueueTTS();
  }

  function handleRefUpload(e) {
    const file = e.target.files?.[0];
    if (file) analyzeReference(file);
    e.target.value = "";
  }

  // --- Microphone recording for reference audio ---
  let canRecord = $derived(typeof navigator !== "undefined" && !!navigator.mediaDevices);
  let refRecording = $state(false);
  let refElapsedSec = $state(0);
  let refMediaRecorder = null;
  let refRecordedChunks = [];
  let refRecStream = null;
  let refTimerInterval = null;

  function formatTime(sec) {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m}:${String(s).padStart(2, "0")}`;
  }

  async function startRefRecording() {
    try {
      refRecStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      return;
    }
    refRecordedChunks = [];
    const mimeType = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "";
    refMediaRecorder = new MediaRecorder(refRecStream, mimeType ? { mimeType } : undefined);
    refMediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) refRecordedChunks.push(e.data); };
    refMediaRecorder.onstop = () => {
      const ext = refMediaRecorder.mimeType?.includes("webm") ? "webm" : "ogg";
      const blob = new Blob(refRecordedChunks, { type: refMediaRecorder.mimeType || "audio/webm" });
      const file = new File([blob], `recording.${ext}`, { type: blob.type });
      analyzeReference(file);
    };
    refMediaRecorder.start();
    refRecording = true;
    refElapsedSec = 0;
    refTimerInterval = setInterval(() => { refElapsedSec++; }, 1000);
  }

  function stopRefRecording() {
    clearInterval(refTimerInterval);
    refTimerInterval = null;
    if (refMediaRecorder && refMediaRecorder.state !== "inactive") refMediaRecorder.stop();
    if (refRecStream) { refRecStream.getTracks().forEach(t => t.stop()); refRecStream = null; }
    refRecording = false;
  }

  function getEmotionsData() {
    return { values: { ...emotionsState.values }, mask: { ...emotionsState.mask } };
  }
  function getEmotionsName(data) {
    const masked = Object.entries(data.mask).filter(([, v]) => v === 1).map(([k]) => k);
    return masked.length > 0 ? masked.slice(0, 3).join(", ") : "All masked";
  }
  function recallEmotions(data) {
    Object.assign(emotionsState.values, data.values);
    Object.assign(emotionsState.mask, data.mask);
  }

  function getVEData() {
    return { ve: embeddingsState.ve ? [...embeddingsState.ve] : null };
  }
  function recallVE(data) {
    if (data.ve) embeddingsState.ve = data.ve;
  }

  function getCPData() {
    return { cp: embeddingsState.cp ? [...embeddingsState.cp] : null };
  }
  function recallCP(data) {
    if (data.cp) embeddingsState.cp = data.cp;
  }

  function getMelData() {
    return {
      mel: melState.mel,
      audioB64: melState.audioB64,
    };
  }
  function getMelName() {
    if (!melState.mel) return "Empty";
    const T = melState.mel[0]?.length ?? 0;
    return `Mel ${T}f ${new Date().toLocaleTimeString()}`;
  }
  function recallMel(data) {
    if (data.mel) melState.mel = data.mel;
    melState.audioB64 = data.audioB64 ?? null;
    ttsState.conditionalsId = null;  // mel changed
  }

  function getT3Data() {
    return {
      tokens: t3PromptState.tokens ? [...t3PromptState.tokens] : null,
      source: t3PromptState.source,
      fileName: t3PromptState.fileName,
    };
  }
  function getT3Name() {
    if (!t3PromptState.tokens) return "Empty";
    const src = t3PromptState.source === "custom" ? t3PromptState.fileName : t3PromptState.source;
    return `T3 ${src} ${new Date().toLocaleTimeString()}`;
  }
  function recallT3(data) {
    t3PromptState.tokens = data.tokens ?? null;
    t3PromptState.source = data.source ?? (data.tokens ? "stash" : "none");
    t3PromptState.fileName = data.fileName ?? null;
    t3PromptState.audioId = null;  // no server-side audio needed
    ttsState.conditionalsId = null;
  }
</script>

<div class="app">
  <header class="app-header">
    <h1>Charvogen</h1>
    <span class="subtitle">Character Voice Generator</span>
    <button class="btn-ref" onclick={() => refInputEl.click()} disabled={uiState.analyzing || refRecording}>
      {uiState.analyzing ? "Analyzing..." : "Load Reference Audio"}
    </button>
    {#if canRecord}
      {#if refRecording}
        <button class="btn-ref btn-rec-stop" onclick={stopRefRecording}>Stop {formatTime(refElapsedSec)}</button>
      {:else}
        <button class="btn-ref btn-rec" onclick={startRefRecording} disabled={uiState.analyzing}>Record</button>
      {/if}
    {/if}
    {#if referenceState.loaded}
      <button class="btn-ref btn-apply" onclick={applyRefAll}>Apply All</button>
    {/if}
    <input bind:this={refInputEl} type="file" accept="audio/*" onchange={handleRefUpload} style="display:none" />
    <div class="random-k-bar">
      <button class="btn-ref" onclick={handleRandomK} disabled={embeddingsState.inferring}>
        Random-K
      </button>
      <input type="range" min="1" max="500" bind:value={randomKKappa}
             title={`κ=${randomKKappa}`} class="kappa-slider" />
      <span class="kappa-label">κ={randomKKappa}</span>
    </div>
    <label class="explore-toggle">
      <input type="checkbox" bind:checked={uiState.exploreMode} /> Explore
    </label>
    {#if uiState.exploreMode}
      <input type="number" bind:value={uiState.exploreCount} min="1" max="20" class="explore-count" />
    {/if}
    {#if uiState.statusMessage}
      <span class="status-msg">{uiState.statusMessage}</span>
    {/if}
    {#if uiState.error}
      <div class="error-bar">{uiState.error}</div>
    {/if}
  </header>

  <main class="app-main">
    <!-- Col 1: VE Planes + VE Embed -->
    <div class="column">
      <SAEPanel subspace="ve" onshiftrecall={recallAndGenerate} />
      <div class="infer-bar">
        <button class="btn-primary" onclick={runInfer} disabled={embeddingsState.inferring}>
          {embeddingsState.inferring ? "Inferring..." : "Infer VE/CP"}
        </button>
        <button class="btn-sm" onclick={handleRandomVE}>
          Random VE
        </button>
      </div>
      <VEPanel />
      <StashList
        storageKey="ve"
        label="VE Embed Stash"
        getCurrentData={getVEData}
        getEntryName={() => `VE ${new Date().toLocaleTimeString()}`}
        onrecall={recallVE}
        onshiftrecall={recallAndGenerate}
      />
    </div>

    <!-- Col 2: CP Planes + CP Embed -->
    <div class="column">
      <SAEPanel subspace="cp" onshiftrecall={recallAndGenerate} />
      <div class="infer-bar">
        <button class="btn-sm" onclick={handleRandomCP}>
          Random CP
        </button>
      </div>
      <CPPanel />
      <StashList
        storageKey="cp"
        label="CP Embed Stash"
        getCurrentData={getCPData}
        getEntryName={() => `CP ${new Date().toLocaleTimeString()}`}
        onrecall={recallCP}
        onshiftrecall={recallAndGenerate}
      />
    </div>

    <!-- Col 3: Emotions & Metrics + T3 Speech Prompt -->
    <div class="column">
      <EmotionsPanel />
      <div class="infer-bar">
        <button class="btn-sm" onclick={handleRandomEmotions}>
          Random Emotions
        </button>
        <button class="btn-sm" onclick={handleRandomMetrics}>
          Random Metrics
        </button>
      </div>
      <StashList
        storageKey="emotions"
        label="Emotions Stash"
        getCurrentData={getEmotionsData}
        getEntryName={getEmotionsName}
        onrecall={recallEmotions}
        onshiftrecall={recallAndGenerate}
      />
      <T3PromptPanel />
      <StashList
        storageKey="t3"
        label="T3 Stash"
        getCurrentData={getT3Data}
        getEntryName={getT3Name}
        onrecall={recallT3}
        onshiftrecall={recallAndGenerate}
      />
      <!-- Full Config Save/Load -->
      <div class="config-bar">
        <button class="btn-sm" onclick={saveConfigToFile}>Save Config</button>
        <button class="btn-sm" onclick={() => configInputEl.click()}>Load Config</button>
        <input bind:this={configInputEl} type="file" accept=".json"
               onchange={loadConfigFromFile} style="display:none" />
      </div>
      <StashList
        storageKey="full_config"
        label="Config Stash"
        getCurrentData={getFullConfig}
        getEntryName={getConfigName}
        onrecall={recallFullConfig}
        onshiftrecall={recallAndGenerate}
      />
    </div>

    <!-- Col 4: Mel + TTS -->
    <div class="column">
      <MelPanel />
      <StashList
        storageKey="mel"
        label="Mel Stash"
        getCurrentData={getMelData}
        getEntryName={getMelName}
        onrecall={recallMel}
        onshiftrecall={recallAndGenerate}
      />
      <TTSPanel />
    </div>
  </main>
</div>

<style>
  .app {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
  }
  .app-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 16px;
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
  }
  .app-header h1 {
    font-size: 18px;
    font-weight: 700;
    color: var(--accent);
  }
  .subtitle {
    font-size: 12px;
    color: var(--fg-dim);
  }
  .btn-ref {
    padding: 4px 14px;
    font-size: 12px;
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    white-space: nowrap;
  }
  .btn-ref:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-ref:hover:not(:disabled) { background: var(--accent-dim); }
  .btn-apply { background: var(--success, #27ae60); }
  .btn-apply:hover { background: #219a52; }
  .btn-rec { background: transparent; color: #e74c3c; border: 1px solid #e74c3c; }
  .btn-rec:hover:not(:disabled) { background: rgba(231, 76, 60, 0.15); }
  .btn-rec-stop { background: #e74c3c; color: #fff; border: none; animation: rec-pulse 1s ease-in-out infinite; }
  .btn-rec-stop:hover { background: #c0392b; }
  @keyframes rec-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
  .status-msg {
    font-size: 11px;
    color: var(--success);
  }
  .error-bar {
    margin-left: auto;
    padding: 4px 12px;
    background: rgba(231, 76, 60, 0.2);
    border: 1px solid #e74c3c;
    border-radius: 4px;
    font-size: 11px;
    color: #e74c3c;
    max-width: 400px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .app-main {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    gap: 8px;
    padding: 8px;
    flex: 1;
    overflow: auto;
  }
  .column {
    display: flex;
    flex-direction: column;
    gap: 8px;
    min-width: 0;
  }
  .infer-bar {
    display: flex;
    justify-content: center;
    padding: 4px;
  }
  .btn-primary {
    padding: 6px 16px;
    font-size: 12px;
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
  }
  .btn-primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .btn-primary:hover:not(:disabled) {
    background: var(--accent-dim);
  }

  .random-k-bar {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-left: 8px;
  }
  .kappa-slider {
    width: 80px;
    height: 16px;
  }
  .kappa-label {
    font-size: 11px;
    color: var(--fg-dim);
    min-width: 45px;
  }
  .explore-toggle {
    font-size: 11px;
    color: var(--fg-dim);
    display: flex;
    align-items: center;
    gap: 3px;
    cursor: pointer;
    white-space: nowrap;
  }
  .explore-toggle input { cursor: pointer; }
  .explore-count {
    width: 42px;
    padding: 2px 4px;
    font-size: 11px;
    background: var(--bg-input);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 3px;
    text-align: center;
  }
  .config-bar {
    display: flex;
    gap: 6px;
    padding: 4px 8px;
  }
  .btn-sm {
    padding: 3px 10px;
    font-size: 11px;
    background: var(--bg-panel);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 4px;
    cursor: pointer;
  }
  .btn-sm:hover {
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }

  /* Responsive: stack on narrow screens */
  @media (max-width: 1400px) {
    .app-main {
      grid-template-columns: 1fr 1fr;
    }
  }
  @media (max-width: 750px) {
    .app-main {
      grid-template-columns: 1fr;
    }
  }
</style>
