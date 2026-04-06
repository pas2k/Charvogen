"""Annotations routes: read/write SAE feature descriptions (per subspace)."""

import json

from fastapi import APIRouter, HTTPException

from ..state import ANNOTATIONS_DIR
from ..api_types import AnnotationsResponse, AnnotationUpdate

router = APIRouter(prefix="/api/annotations", tags=["annotations"])

_VE_FILE = ANNOTATIONS_DIR / "ve_features.json"
_CP_FILE = ANNOTATIONS_DIR / "cp_features.json"


def _load(path) -> dict[str, str]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save(path, data: dict[str, str]):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _file_for(subspace: str):
    if subspace == "ve":
        return _VE_FILE
    elif subspace == "cp":
        return _CP_FILE
    else:
        raise HTTPException(400, f"Invalid subspace: {subspace!r} (expected 've' or 'cp')")


@router.get("", response_model=AnnotationsResponse)
async def get_annotations():
    return AnnotationsResponse(ve=_load(_VE_FILE), cp=_load(_CP_FILE))


@router.put("/{subspace}/{key:path}")
async def update_annotation(subspace: str, key: str, body: AnnotationUpdate):
    path = _file_for(subspace)
    data = _load(path)
    if body.description:
        data[key] = body.description
    else:
        data.pop(key, None)
    _save(path, data)
    return {"ok": True}
