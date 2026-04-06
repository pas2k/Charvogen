"""VAE routes: full pipeline decode (dual RotSAE features + emotions -> VE + CP)."""

from fastapi import APIRouter

from ..state import app_state
from ..api_types import (
    FullPipelineRequest, FullPipelineResponse,
    RandomKRequest, RandomKResponse,
    RandomKBatchRequest, RandomKBatchResponse,
)
from ..models import full_pipeline, random_k, random_k_batch

router = APIRouter(prefix="/api/vae", tags=["vae"])


@router.post("/full-pipeline", response_model=FullPipelineResponse)
async def decode_full_pipeline(req: FullPipelineRequest):
    latent, ve, cp = full_pipeline(
        app_state,
        req.ve_features,
        req.cp_features,
        req.emotions,
        req.emotion_mask,
    )
    return FullPipelineResponse(latent=latent, ve=ve, cp=cp)


@router.post("/random-k", response_model=RandomKResponse)
async def random_k_endpoint(req: RandomKRequest):
    result = random_k(
        app_state,
        req.ve_features, req.cp_features,
        req.emotions, req.emotion_mask,
        req.kappa,
        req.subspace,
    )
    return RandomKResponse(**result)


@router.post("/random-k-batch", response_model=RandomKBatchResponse)
async def random_k_batch_endpoint(req: RandomKBatchRequest):
    samples = random_k_batch(
        app_state,
        req.ve_features, req.cp_features,
        req.emotions, req.emotion_mask,
        req.kappa, req.subspace, req.n_samples,
    )
    return RandomKBatchResponse(samples=[RandomKResponse(**s) for s in samples])
