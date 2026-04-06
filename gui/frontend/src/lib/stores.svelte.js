/**
 * Svelte 5 reactive stores for application state.
 *
 * Uses $state runes for fine-grained reactivity.
 */

import { apiGet, apiPost, apiUpload } from "./api.js";

// --- RotSAE state (dual: VE + CP) ---
export const saeState = $state({
  config: null,          // {ve: {n_planes, alive_planes, k, ...}, cp: {...}}
  features: { ve: [], cp: [] },  // plane metadata arrays
  hierarchy: null,       // {ve: {metadata, root}, cp: {metadata, root}}
  veActivations: {},     // {index_str: angle_value} — VE RotSAE activations
  cpActivations: {},     // {index_str: angle_value} — CP RotSAE activations
  loading: false,
});

// --- Emotions state ---
export const emotionsState = $state({
  columns: [],           // emotion column names
  metrics: [],           // metric column names
  values: {},            // {name: float}
  mask: {},              // {name: 0|1}
  loading: false,
});

// --- Embeddings state ---
export const embeddingsState = $state({
  latent: null,          // float[160]
  ve: null,              // float[256]
  cp: null,              // float[80]
  autoInfer: true,       // auto-infer on SAE/emotion change
  inferring: false,
});

// --- Mel state ---
export const melState = $state({
  mel: null,             // float[80][T]
  audioB64: null,        // base64 WAV
  generating: false,
  nFrames: 256,
  ddimSteps: 50,
  guidanceScale: 3.5,
  seedStrength: 0.075,   // 0=pure noise, 0.05-0.1=subtle seed scaffold
});

// --- T3 prompt state ---
export const t3PromptState = $state({
  source: "none",        // "none" | "reference" | "custom" | "stash"
  audioId: null,         // audio_id for custom/reference upload
  fileName: null,        // display name for custom upload
  tokens: null,          // int[] — pre-extracted T3 speech tokens
  uploading: false,
});

// --- TTS state ---
export const ttsState = $state({
  text: localStorage.getItem("charvogen_tts_text") ?? "",
  conditionalsId: null,
  preparing: false,
  exaggeration: 0.5,
  cfgWeight: 0.5,
  temperature: 0.8,
  repetitionPenalty: 1.2,
  minP: 0.05,
  topP: 1.0,
  jobs: [],              // [{job_id, status, text, label, audio_b64, duration_s}]
});

// --- Annotations (per subspace) ---
export const annotationsState = $state({
  ve: {},       // {index_str: description} for VE planes
  cp: {},       // {index_str: description} for CP planes
});

// --- Reference state (analyzed but not yet applied) ---
export const referenceState = $state({
  loaded: false,         // whether a reference has been analyzed
  audioId: null,         // audio_id for TTS conditioning
  veActivations: {},     // {index_str: angle} — VE RotSAE
  cpActivations: {},     // {index_str: angle} — CP RotSAE
  emotions: {},          // {name: float}
  ve: null,              // float[256]
  cp: null,              // float[80]
  latent: null,          // float[160]
  mel: null,             // float[80][T]
});

// --- UI state ---
export const uiState = $state({
  activeTab: "design",   // "design" | "compare" | "project"
  error: null,
  statusMessage: null,
  analyzing: false,      // reference audio analysis in progress
  exploreMode: false,    // batch random → stash instead of in-place mutation
  exploreCount: 5,       // number of variants to generate in explore mode
  stashVersion: 0,       // bumped after programmatic stash writes → triggers StashList refresh
  lastRecalled: null,    // { label, name, stashType, data } — last recalled stash entry
});


// --- Persistence ---

$effect.root(() => {
  $effect(() => {
    localStorage.setItem("charvogen_tts_text", ttsState.text);
  });
});

// --- Stash recall tracking ---

const EPS = 1e-4;

function dictsMatch(a, b) {
  if (a === b) return true;
  if (!a || !b) return false;
  const keysA = Object.keys(a);
  const keysB = Object.keys(b);
  if (keysA.length !== keysB.length) return false;
  for (const k of keysA) {
    const va = a[k], vb = b[k];
    if (typeof va === "number" && typeof vb === "number") {
      if (Math.abs(va - vb) > EPS) return false;
    } else if (va !== vb) return false;
  }
  return true;
}

/** Check if last-recalled stash still matches current state. */
export function getStashAutoLabel() {
  const r = uiState.lastRecalled;
  if (!r) return "";
  const d = r.data;
  const prefixes = {
    full_config: "All", sae_ve: "VE", sae_cp: "CP",
    emotions: "Emo", ve: "VE~", cp: "CP~", mel: "Mel", t3: "T3",
  };
  const tag = `${prefixes[r.stashType] ?? r.stashType}: ${r.name}`;
  switch (r.stashType) {
    case "full_config":
      if (!d.sae) return "";
      if (!dictsMatch(d.sae.veActivations, saeState.veActivations)) return "";
      if (!dictsMatch(d.sae.cpActivations, saeState.cpActivations)) return "";
      if (d.emotions && !dictsMatch(d.emotions.values, emotionsState.values)) return "";
      return tag;
    case "sae_ve":
      if (!dictsMatch(d.activations, saeState.veActivations)) return "";
      return tag;
    case "sae_cp":
      if (!dictsMatch(d.activations, saeState.cpActivations)) return "";
      return tag;
    case "emotions":
      if (!dictsMatch(d.values, emotionsState.values)) return "";
      if (!dictsMatch(d.mask, emotionsState.mask)) return "";
      return tag;
    case "ve":
      if (!d.ve || !embeddingsState.ve) return "";
      if (d.ve.length !== embeddingsState.ve.length) return "";
      for (let i = 0; i < d.ve.length; i++)
        if (Math.abs(d.ve[i] - embeddingsState.ve[i]) > EPS) return "";
      return tag;
    case "cp":
      if (!d.cp || !embeddingsState.cp) return "";
      if (d.cp.length !== embeddingsState.cp.length) return "";
      for (let i = 0; i < d.cp.length; i++)
        if (Math.abs(d.cp[i] - embeddingsState.cp[i]) > EPS) return "";
      return tag;
    case "mel":
      // Mel stash — just check dimensions match (full array compare too expensive)
      if (!d.mel || !melState.mel) return "";
      if (d.mel.length !== melState.mel.length) return "";
      if (d.mel[0]?.length !== melState.mel[0]?.length) return "";
      return tag;
    default:
      return "";
  }
}

export function setLastRecalled(stashType, label, name, data) {
  uiState.lastRecalled = { stashType, label, name, data };
}

// --- Actions ---

let _inferDebounceTimer = null;

export function scheduleInfer() {
  if (!embeddingsState.autoInfer) return;
  clearTimeout(_inferDebounceTimer);
  _inferDebounceTimer = setTimeout(() => runInfer(), 300);
}

export async function runInfer() {
  embeddingsState.inferring = true;
  uiState.error = null;
  try {
    const result = await apiPost("/api/vae/full-pipeline", $state.snapshot({
      ve_features: saeState.veActivations,
      cp_features: saeState.cpActivations,
      emotions: emotionsState.values,
      emotion_mask: emotionsState.mask,
    }));
    embeddingsState.latent = result.latent;
    embeddingsState.ve = result.ve;
    embeddingsState.cp = result.cp;
  } catch (e) {
    uiState.error = e.message;
  } finally {
    embeddingsState.inferring = false;
  }
}

export function setFeature(subspace, index, value) {
  const dict = subspace === "cp" ? saeState.cpActivations : saeState.veActivations;
  if (value === 0) {
    delete dict[String(index)];
  } else {
    dict[String(index)] = value;
  }
  scheduleInfer();
}

export function setEmotion(name, value) {
  emotionsState.values[name] = value;
  scheduleInfer();
}

export function toggleEmotionMask(name) {
  emotionsState.mask[name] = emotionsState.mask[name] ? 0 : 1;
  scheduleInfer();
}

export async function loadInitialData() {
  try {
    const [config, features, hierarchy, emoData, annots] = await Promise.all([
      apiGet("/api/sae/config"),
      apiGet("/api/sae/features"),
      apiGet("/api/sae/hierarchy"),
      apiGet("/api/emotions"),
      apiGet("/api/annotations"),
    ]);
    saeState.config = config;
    saeState.features = { ve: features.ve || [], cp: features.cp || [] };
    saeState.hierarchy = hierarchy;
    emotionsState.columns = emoData.emotions;
    emotionsState.metrics = emoData.metrics;

    // Initialize emotion values and mask to defaults
    for (const name of [...emoData.emotions, ...emoData.metrics]) {
      if (!(name in emotionsState.values)) emotionsState.values[name] = 0.5;
      if (!(name in emotionsState.mask)) emotionsState.mask[name] = 0;
    }

    annotationsState.ve = annots.ve || {};
    annotationsState.cp = annots.cp || {};
  } catch (e) {
    uiState.error = `Failed to load initial data: ${e.message}`;
  }
}

export async function reloadAnnotations() {
  try {
    const annots = await apiGet("/api/annotations");
    annotationsState.ve = annots.ve || {};
    annotationsState.cp = annots.cp || {};
  } catch (e) {
    console.error("Failed to reload annotations:", e);
  }
}

export async function generateMel() {
  if (!embeddingsState.ve || !embeddingsState.cp) {
    uiState.error = "Run inference first to get VE/CP embeddings";
    return;
  }
  melState.generating = true;
  uiState.error = null;
  try {
    const result = await apiPost("/api/mel/generate", $state.snapshot({
      ve: embeddingsState.ve,
      cp: embeddingsState.cp,
      n_frames: melState.nFrames,
      ddim_steps: melState.ddimSteps,
      guidance_scale: melState.guidanceScale,
      seed_strength: melState.seedStrength,
    }));
    melState.mel = result.mel;
    melState.audioB64 = result.audio_b64;
    ttsState.conditionalsId = null;  // mel changed, re-prepare on next TTS
  } catch (e) {
    uiState.error = e.message;
  } finally {
    melState.generating = false;
  }
}

export async function vocodeMel() {
  if (!melState.mel) {
    uiState.error = "No mel spectrogram to vocode";
    return;
  }
  melState.generating = true;
  uiState.error = null;
  try {
    const result = await apiPost("/api/mel/to-audio", { mel: $state.snapshot(melState.mel) });
    melState.audioB64 = result.audio_b64;
  } catch (e) {
    uiState.error = e.message;
  } finally {
    melState.generating = false;
  }
}

async function prepareConditionals() {
  if (!embeddingsState.ve) {
    throw new Error("Run inference first");
  }
  ttsState.preparing = true;
  try {
    // Snapshot reactive proxies — Svelte 5 $state proxies may not
    // serialize correctly via JSON.stringify for deeply nested arrays
    const payload = $state.snapshot({
      ve: embeddingsState.ve,
      cp: embeddingsState.cp,
      mel: melState.mel,
      t3_tokens: t3PromptState.tokens,
      audio_id: t3PromptState.audioId,
      exaggeration: ttsState.exaggeration,
    });
    const result = await apiPost("/api/tts/prepare-conditionals", payload);
    ttsState.conditionalsId = result.conditionals_id;
  } finally {
    ttsState.preparing = false;
  }
}

export async function enqueueTTS() {
  if (!embeddingsState.ve) {
    uiState.error = "Run inference first to get VE embedding";
    return;
  }
  if (!melState.mel) {
    uiState.error = "Generate a mel spectrogram first (Mel panel)";
    return;
  }
  if (!t3PromptState.tokens && !t3PromptState.audioId) {
    uiState.error = "T3 speech prompt required — upload or apply a reference in the T3 Prompt panel";
    return;
  }
  if (!ttsState.text.trim()) {
    uiState.error = "Enter text to synthesize";
    return;
  }
  uiState.error = null;
  try {
    // Auto-prepare if needed
    if (!ttsState.conditionalsId) {
      await prepareConditionals();
    }
    const result = await apiPost("/api/tts/enqueue", {
      text: ttsState.text,
      conditionals_id: ttsState.conditionalsId,
      cfg_weight: ttsState.cfgWeight,
      temperature: ttsState.temperature,
      repetition_penalty: ttsState.repetitionPenalty,
      min_p: ttsState.minP,
      top_p: ttsState.topP,
      exaggeration: ttsState.exaggeration,
    });
    ttsState.jobs = [...ttsState.jobs, {
      job_id: result.job_id,
      status: "queued",
      text: ttsState.text,
      label: getStashAutoLabel(),
      audio_b64: null,
      duration_s: null,
    }];
  } catch (e) {
    uiState.error = e.message;
  }
}

export function removeJob(jobId) {
  ttsState.jobs = ttsState.jobs.filter(j => j.job_id !== jobId);
}

export function clearFinishedJobs() {
  ttsState.jobs = ttsState.jobs.filter(j => j.status !== "done" && j.status !== "error");
}

// --- Reference analysis & per-panel apply ---

export async function analyzeReference(file) {
  uiState.analyzing = true;
  uiState.error = null;
  uiState.statusMessage = "Analyzing reference audio...";
  try {
    const result = await apiUpload("/api/audio/analyze", file);

    // Store in referenceState (not directly into panels)
    referenceState.audioId = result.audio_id;
    referenceState.veActivations = result.ve_activations;
    referenceState.cpActivations = result.cp_activations;
    referenceState.emotions = result.emotions;
    referenceState.ve = result.ve;
    referenceState.cp = result.cp;
    referenceState.latent = result.latent;
    referenceState.mel = result.mel;
    referenceState.loaded = true;

    const nVe = Object.keys(result.ve_activations).length;
    const nCp = Object.keys(result.cp_activations).length;
    uiState.statusMessage = `Reference analyzed (VE: ${nVe}, CP: ${nCp} active planes) — apply to panels`;
  } catch (e) {
    uiState.error = e.message;
    uiState.statusMessage = null;
  } finally {
    uiState.analyzing = false;
  }
}

export function applyRefSAE() {
  if (!referenceState.loaded) return;
  for (const key of Object.keys(saeState.veActivations)) delete saeState.veActivations[key];
  for (const [k, v] of Object.entries(referenceState.veActivations)) saeState.veActivations[k] = v;
  for (const key of Object.keys(saeState.cpActivations)) delete saeState.cpActivations[key];
  for (const [k, v] of Object.entries(referenceState.cpActivations)) saeState.cpActivations[k] = v;
  ttsState.conditionalsId = null;
  scheduleInfer();
}

export function applyRefSAEVE() {
  if (!referenceState.loaded) return;
  for (const key of Object.keys(saeState.veActivations)) delete saeState.veActivations[key];
  for (const [k, v] of Object.entries(referenceState.veActivations)) saeState.veActivations[k] = v;
  ttsState.conditionalsId = null;
  scheduleInfer();
}

export function applyRefSAECP() {
  if (!referenceState.loaded) return;
  for (const key of Object.keys(saeState.cpActivations)) delete saeState.cpActivations[key];
  for (const [k, v] of Object.entries(referenceState.cpActivations)) saeState.cpActivations[k] = v;
  ttsState.conditionalsId = null;
  scheduleInfer();
}

export function applyRefEmotions() {
  if (!referenceState.loaded) return;
  for (const [name, val] of Object.entries(referenceState.emotions)) {
    emotionsState.values[name] = val;
    emotionsState.mask[name] = 1;
  }
  ttsState.conditionalsId = null;
  scheduleInfer();
}

export function applyRefVE() {
  if (!referenceState.loaded) return;
  embeddingsState.ve = referenceState.ve;
  embeddingsState.latent = referenceState.latent;
  ttsState.conditionalsId = null;
}

export function applyRefCP() {
  if (!referenceState.loaded) return;
  embeddingsState.cp = referenceState.cp;
  ttsState.conditionalsId = null;
}

export function applyRefMel() {
  if (!referenceState.loaded) return;
  melState.mel = referenceState.mel.map(row => row.slice());
  melState.audioB64 = null;
  ttsState.conditionalsId = null;  // mel changed
}

export async function applyRefAll() {
  if (!referenceState.loaded) return;
  applyRefSAE();
  applyRefEmotions();
  applyRefVE();
  applyRefCP();
  applyRefMel();
  await applyRefT3();
}

// --- T3 prompt audio actions ---

export async function uploadT3Prompt(file) {
  t3PromptState.uploading = true;
  uiState.error = null;
  try {
    const result = await apiUpload("/api/audio/upload", file);
    const tokResult = await apiPost(`/api/audio/${result.audio_id}/t3-tokens`);
    t3PromptState.audioId = result.audio_id;
    t3PromptState.tokens = tokResult.tokens;
    t3PromptState.source = "custom";
    t3PromptState.fileName = file.name;
    ttsState.conditionalsId = null;  // invalidate
  } catch (e) {
    uiState.error = e.message;
  } finally {
    t3PromptState.uploading = false;
  }
}

export async function applyRefT3() {
  if (!referenceState.loaded) return;
  t3PromptState.audioId = referenceState.audioId;
  t3PromptState.source = "reference";
  t3PromptState.fileName = null;
  const tokResult = await apiPost(`/api/audio/${referenceState.audioId}/t3-tokens`);
  t3PromptState.tokens = tokResult.tokens;
  ttsState.conditionalsId = null;
}

export function resetT3ToSeed() {
  t3PromptState.audioId = null;
  t3PromptState.tokens = null;
  t3PromptState.source = "none";
  t3PromptState.fileName = null;
  ttsState.conditionalsId = null;
}

export async function randomK(kappa = 100) {
  embeddingsState.inferring = true;
  uiState.error = null;
  try {
    const result = await apiPost("/api/vae/random-k", $state.snapshot({
      ve_features: saeState.veActivations,
      cp_features: saeState.cpActivations,
      emotions: emotionsState.values,
      emotion_mask: emotionsState.mask,
      kappa,
    }));
    // Apply all SAE activations
    for (const key of Object.keys(saeState.veActivations)) delete saeState.veActivations[key];
    for (const [k, v] of Object.entries(result.ve_activations)) saeState.veActivations[k] = v;
    for (const key of Object.keys(saeState.cpActivations)) delete saeState.cpActivations[key];
    for (const [k, v] of Object.entries(result.cp_activations)) saeState.cpActivations[k] = v;
    // Apply emotions
    for (const [name, val] of Object.entries(result.emotions)) {
      emotionsState.values[name] = val;
    }
    // Apply embeddings
    embeddingsState.latent = result.latent;
    embeddingsState.ve = result.ve;
    embeddingsState.cp = result.cp;
    // Invalidate downstream
    ttsState.conditionalsId = null;
  } catch (e) {
    uiState.error = e.message;
  } finally {
    embeddingsState.inferring = false;
  }
}

export async function randomKBatch(kappa = 100, nSamples = 5, subspace = "all") {
  const result = await apiPost("/api/vae/random-k-batch", $state.snapshot({
    ve_features: saeState.veActivations,
    cp_features: saeState.cpActivations,
    emotions: emotionsState.values,
    emotion_mask: emotionsState.mask,
    kappa,
    subspace,
    n_samples: nSamples,
  }));
  return result.samples;
}

// --- Per-pane frontend-only randomizers (no cross-pane effects) ---

function _gaussNoise(sigma) {
  const u1 = Math.random(), u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * sigma;
}

export function randomizeVE(kappa = 100) {
  const sigma = 2.0 / kappa;
  const snap = $state.snapshot(saeState.veActivations);
  const alive = (saeState.features.ve || []).filter(f => (f.activation_freq ?? 0) > 0);
  for (const key of Object.keys(saeState.veActivations)) delete saeState.veActivations[key];
  for (const feat of alive) {
    const idx = String(feat.index);
    const cur = snap[idx] ?? 0;
    const val = cur + _gaussNoise(sigma);
    if (Math.abs(val) > 1e-4) saeState.veActivations[idx] = val;
  }
  scheduleInfer();
}

export function randomizeCP(kappa = 100) {
  const sigma = 2.0 / kappa;
  const snap = $state.snapshot(saeState.cpActivations);
  const alive = (saeState.features.cp || []).filter(f => (f.activation_freq ?? 0) > 0);
  for (const key of Object.keys(saeState.cpActivations)) delete saeState.cpActivations[key];
  for (const feat of alive) {
    const idx = String(feat.index);
    const cur = snap[idx] ?? 0;
    const val = cur + _gaussNoise(sigma);
    if (Math.abs(val) > 1e-4) saeState.cpActivations[idx] = val;
  }
  scheduleInfer();
}

export function randomizeEmotions(kappa = 100) {
  const sigma = 0.3 / kappa;
  for (const name of emotionsState.columns) {
    if (name in emotionsState.values) {
      emotionsState.values[name] = Math.max(0, Math.min(1, emotionsState.values[name] + _gaussNoise(sigma)));
    }
  }
  scheduleInfer();
}

export function randomizeMetrics(kappa = 100) {
  const sigma = 0.3 / kappa;
  for (const name of emotionsState.metrics) {
    if (name in emotionsState.values) {
      emotionsState.values[name] = Math.max(0, Math.min(1, emotionsState.values[name] + _gaussNoise(sigma)));
    }
  }
  scheduleInfer();
}

export function handleWSMessage(msg) {
  if (msg.type === "job_started") {
    const job = ttsState.jobs.find(j => j.job_id === msg.job_id);
    if (job) job.status = "running";
  } else if (msg.type === "job_done") {
    const job = ttsState.jobs.find(j => j.job_id === msg.job_id);
    if (job) {
      job.status = "done";
      job.audio_b64 = msg.audio_b64;
      job.duration_s = msg.duration_s;
    }
  } else if (msg.type === "job_error") {
    const job = ttsState.jobs.find(j => j.job_id === msg.job_id);
    if (job) {
      job.status = "error";
      job.error = msg.error;
    }
  }
}
