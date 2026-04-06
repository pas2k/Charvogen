"""Pydantic request/response schemas for the Charvogen API."""

from pydantic import BaseModel, Field


# --- RotSAE ---

class RotSAESubConfig(BaseModel):
    """RotSAE config for one subspace (VE or CP)."""
    n_planes: int
    alive_planes: int
    k: int
    ve_dim: int
    cp_dim: int


class DualRotSAEConfig(BaseModel):
    ve: RotSAESubConfig
    cp: RotSAESubConfig


class PlaneInfo(BaseModel):
    index: int
    subspace: str          # "VE" or "CP"
    angle_variance: float
    activation_freq: float | None = None
    mean_abs_angle: float | None = None
    mean_signed_angle: float | None = None
    angle_std: float | None = None
    top_emotion_correlations: list[dict] = []


# --- VAE ---

class FullPipelineRequest(BaseModel):
    ve_features: dict[str, float] = Field(
        default_factory=dict,
        description="VE RotSAE plane activations: {index_str: angle}",
    )
    cp_features: dict[str, float] = Field(
        default_factory=dict,
        description="CP RotSAE plane activations: {index_str: angle}",
    )
    emotions: dict[str, float] = Field(
        default_factory=dict,
        description="Emotion values: {name: value}",
    )
    emotion_mask: dict[str, int] = Field(
        default_factory=dict,
        description="Emotion mask: {name: 0|1}",
    )


class FullPipelineResponse(BaseModel):
    latent: list[float]
    ve: list[float]
    cp: list[float]


class RandomKRequest(BaseModel):
    ve_features: dict[str, float] = Field(default_factory=dict)
    cp_features: dict[str, float] = Field(default_factory=dict)
    emotions: dict[str, float] = Field(default_factory=dict)
    emotion_mask: dict[str, int] = Field(default_factory=dict)
    kappa: float = Field(
        default=100.0,
        gt=0,
        description="vMF concentration. Higher=closer, lower=more variation.",
    )
    subspace: str = Field(
        default="all",
        pattern="^(all|ve|cp)$",
        description="Which latent subspace to perturb: all, ve, or cp.",
    )


class RandomKResponse(BaseModel):
    ve_activations: dict[str, float]
    cp_activations: dict[str, float]
    emotions: dict[str, float]
    latent: list[float]
    ve: list[float]
    cp: list[float]


class RandomKBatchRequest(BaseModel):
    ve_features: dict[str, float] = Field(default_factory=dict)
    cp_features: dict[str, float] = Field(default_factory=dict)
    emotions: dict[str, float] = Field(default_factory=dict)
    emotion_mask: dict[str, int] = Field(default_factory=dict)
    kappa: float = Field(default=100.0, gt=0)
    subspace: str = Field(default="all", pattern="^(all|ve|cp)$")
    n_samples: int = Field(default=5, ge=1, le=20)


class RandomKBatchResponse(BaseModel):
    samples: list[RandomKResponse]


# --- Mel ---

class MelGenerateRequest(BaseModel):
    ve: list[float]
    cp: list[float]
    n_frames: int = 256
    ddim_steps: int = 50
    guidance_scale: float = 3.5
    seed_strength: float = Field(
        default=0.075,
        ge=0.0,
        le=1.0,
        description="How much seed mel structure to preserve (0=none, 1=full). "
                    "0.05-0.1 is enough to avoid noisy artefacts.",
    )


class MelGenerateResponse(BaseModel):
    mel: list[list[float]]
    audio_b64: str


class MelToAudioRequest(BaseModel):
    mel: list[list[float]]


class MelToAudioResponse(BaseModel):
    audio_b64: str


# --- Audio upload ---

class AudioUploadResponse(BaseModel):
    audio_id: str
    duration_s: float


class ExtractEmbeddingsResponse(BaseModel):
    ve: list[float]
    cp_80: list[float]
    cp_192: list[float]


class ExtractMelResponse(BaseModel):
    mel: list[list[float]]


class AnalyzeReferenceResponse(BaseModel):
    audio_id: str
    ve: list[float]
    cp: list[float]
    latent: list[float]
    ve_activations: dict[str, float]
    cp_activations: dict[str, float]
    emotions: dict[str, float]
    mel: list[list[float]]


# --- TTS ---

class T3TokensResponse(BaseModel):
    tokens: list[int]


class TTSPrepareRequest(BaseModel):
    ve: list[float]
    cp: list[float] | None = None
    mel: list[list[float]] | None = None
    audio_id: str | None = None
    t3_tokens: list[int] | None = None
    exaggeration: float = 0.5


class TTSPrepareResponse(BaseModel):
    conditionals_id: str


class TTSEnqueueRequest(BaseModel):
    text: str
    conditionals_id: str
    cfg_weight: float = 0.5
    temperature: float = 0.8
    repetition_penalty: float = 1.2
    min_p: float = 0.05
    top_p: float = 1.0
    exaggeration: float = 0.5


class TTSEnqueueResponse(BaseModel):
    job_id: str
    position: int


class TTSJobStatus(BaseModel):
    job_id: str
    status: str  # "queued" | "running" | "done" | "error"
    text: str
    error: str | None = None


class TTSResultResponse(BaseModel):
    audio_b64: str
    duration_s: float


# --- Annotations ---

class AnnotationsResponse(BaseModel):
    ve: dict[str, str] = Field(default_factory=dict)
    cp: dict[str, str] = Field(default_factory=dict)


class AnnotationUpdate(BaseModel):
    description: str


# --- Project ---

class ProjectSaveRequest(BaseModel):
    name: str
    state: dict


class ProjectInfo(BaseModel):
    name: str
    modified: str


class ProjectListResponse(BaseModel):
    projects: list[ProjectInfo]
