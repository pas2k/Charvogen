"""Project routes: save/load full frontend state."""

import json
from datetime import datetime

from fastapi import APIRouter, HTTPException

from ..state import PROJECTS_DIR
from ..api_types import ProjectSaveRequest, ProjectListResponse, ProjectInfo

router = APIRouter(prefix="/api/project", tags=["project"])


@router.post("/save")
async def save_project(req: ProjectSaveRequest):
    safe_name = "".join(c for c in req.name if c.isalnum() or c in "-_ ").strip()
    if not safe_name:
        raise HTTPException(400, "Invalid project name")

    path = PROJECTS_DIR / f"{safe_name}.json"
    with open(path, "w") as f:
        json.dump(req.state, f, indent=2)
    return {"ok": True, "name": safe_name}


@router.get("/list", response_model=ProjectListResponse)
async def list_projects():
    projects = []
    for p in sorted(PROJECTS_DIR.glob("*.json")):
        stat = p.stat()
        projects.append(ProjectInfo(
            name=p.stem,
            modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
        ))
    return ProjectListResponse(projects=projects)


@router.get("/load/{name}")
async def load_project(name: str):
    path = PROJECTS_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, "Project not found")
    with open(path) as f:
        return json.load(f)
