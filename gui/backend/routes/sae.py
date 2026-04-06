"""RotSAE routes: dual config, planes metadata, hierarchy."""

from fastapi import APIRouter

from ..state import app_state
from ..api_types import DualRotSAEConfig, RotSAESubConfig

router = APIRouter(prefix="/api/sae", tags=["sae"])


@router.get("/config", response_model=DualRotSAEConfig)
async def get_config():
    return DualRotSAEConfig(
        ve=RotSAESubConfig(
            n_planes=app_state.ve_rotsae_config.get("n_planes", 2048),
            alive_planes=app_state.ve_rotsae_config.get("alive_planes", len(app_state.ve_planes_meta)),
            k=app_state.ve_rotsae_config.get("k", 60),
            ve_dim=app_state.sae_ve_dim,
            cp_dim=app_state.sae_cp_dim,
        ),
        cp=RotSAESubConfig(
            n_planes=app_state.cp_rotsae_config.get("n_planes", 2048),
            alive_planes=app_state.cp_rotsae_config.get("alive_planes", len(app_state.cp_planes_meta)),
            k=app_state.cp_rotsae_config.get("k", 60),
            ve_dim=app_state.sae_ve_dim,
            cp_dim=app_state.sae_cp_dim,
        ),
    )


@router.get("/features")
async def get_features():
    return {
        "ve": app_state.ve_planes_meta,
        "cp": app_state.cp_planes_meta,
    }


@router.get("/hierarchy")
async def get_hierarchy():
    return {
        "ve": app_state.ve_hierarchy,
        "cp": app_state.cp_hierarchy,
    }
