"""
FastAPI server for the Cloud FinOps Optimizer.

Uses create_fastapi_app() from openenv.core.env_server which automatically
provides: /ws  /step  /state  /health  /web  /docs

Custom endpoints defined here (overriding or supplementing the framework):
    POST /reset     — explicit override guaranteeing task_id body routing
    GET  /grade     — deterministic episode score (0.0-1.0) + explainability
    GET  /tasks     — list available tasks
    POST /simulate  — project action outcome without mutating state

/reset override rationale:
    create_fastapi_app() may or may not forward JSON body kwargs to env.reset().
    Rather than rely on framework internals, we define /reset explicitly here so
    {"task_id": "task1"} in the request body is always forwarded correctly.
    The framework's WebSocket /ws endpoint uses its own session management for
    interactive clients; our HTTP /reset serves the validator and inference script.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from openenv.core.env_server import create_fastapi_app
from pydantic import BaseModel

from models import FinOpsAction, FinOpsObservation
from server.environment import CloudFinOpsEnvironment, TASK_CONFIGS

# ---------------------------------------------------------------------------
# Module-level environment instance.
# Shared by all HTTP endpoints. The WebSocket path managed by create_fastapi_app
# uses its own instance; we keep them in sync via the startup hook + middleware.
# ---------------------------------------------------------------------------
_env = CloudFinOpsEnvironment()

# ---------------------------------------------------------------------------
# Build the base OpenEnv application
# ---------------------------------------------------------------------------
def _env_factory():
    return _env

app = create_fastapi_app(_env_factory, FinOpsAction, FinOpsObservation)

# Remove the framework's /reset and /state so our explicit overrides take effect
app.router.routes = [
    r for r in app.router.routes 
    if getattr(r, "path", None) not in ("/reset", "/state")
]


def _get_env() -> CloudFinOpsEnvironment:
    """
    Return the authoritative environment instance.
    Prefers app.state.env (set by create_fastapi_app) if available,
    otherwise falls back to the module-level _env.
    """
    try:
        env = app.state.env
        if env is not None:
            return env
    except AttributeError:
        pass
    return _env


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to the interactive API docs."""
    return RedirectResponse(url="/docs")


# ---------------------------------------------------------------------------
# Startup hook
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _startup():
    """Wire module-level _env into app.state if the framework has not done so."""
    try:
        if app.state.env is None:
            app.state.env = _env
    except AttributeError:
        app.state.env = _env


# ---------------------------------------------------------------------------
# Middleware: keep _env in sync after /step calls
# ---------------------------------------------------------------------------

@app.middleware("http")
async def _sync_env_middleware(request: Request, call_next):
    global _env
    response = await call_next(request)
    if request.url.path in ("/step",):
        try:
            fw_env = app.state.env
            if fw_env is not None and fw_env is not _env:
                _env = fw_env
        except AttributeError:
            pass
    return response


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task1"


# ---------------------------------------------------------------------------
# /reset — explicit override so task_id is always forwarded from body
# ---------------------------------------------------------------------------

@app.post("/reset", tags=["env"])
async def reset_episode(request_body: ResetRequest):
    """
    Start a new episode for the specified task.

    Body:
        {"task_id": "task1"}   easy   — Waste Cleanup
        {"task_id": "task2"}   medium — Resource Optimization
        {"task_id": "task3"}   hard   — Strategic Planning

    Returns a FinOpsObservation with full resource list, dependency graph,
    SLA status, and the episode goal.
    """
    env = _get_env()
    try:
        obs = env.reset(task_id=request_body.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Serialize — FinOpsObservation is a Pydantic model
    try:
        return JSONResponse(content=obs.model_dump())
    except AttributeError:
        # Pydantic v1 fallback
        return JSONResponse(content=obs.dict())


# ---------------------------------------------------------------------------
# /grade — deterministic score
# ---------------------------------------------------------------------------

@app.get("/grade", tags=["grading"])
async def grade_episode():
    """
    Return the deterministic episode grade (0.0-1.0) plus explainability_score.
    Safe to call at any point during or after an episode.
    """
    env = _get_env()
    if not env.state.task_id:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call /reset first.",
        )
    return JSONResponse(content=env.grade())


# ---------------------------------------------------------------------------
# /tasks — task catalogue
# ---------------------------------------------------------------------------

@app.get("/tasks", tags=["meta"])
async def list_tasks():
    """List all available tasks with descriptions, budgets, and step limits."""
    return {
        task_id: {
            "description":     cfg["description"],
            "budget_per_hour": cfg["budget_per_hour"],
            "max_steps":       cfg["max_steps"],
        }
        for task_id, cfg in TASK_CONFIGS.items()
    }


# ---------------------------------------------------------------------------
# /simulate — what-if projection
# ---------------------------------------------------------------------------

@app.post("/simulate", tags=["tools"])
async def simulate_action(request: Request):
    """
    Project the outcome of a proposed action without mutating environment state.

    Body (same format as a /step action):
        {"action_type": "terminate", "resource_id": "ebs-001"}
        {"action_type": "resize",    "resource_id": "ec2-t01", "target_size": "t2.medium"}
        {"action_type": "reserve",   "resource_id": "ec2-h01"}

    Returns SimulateResult:
        projected_cost_per_hour, projected_budget_remaining, projected_reward,
        projected_sla_violations, cascading_risks, recommendation, safe_to_apply
    """
    env = _get_env()
    if not env.state.task_id:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call /reset first.",
        )
    body = await request.json()
    result = env._simulate(body)
    try:
        return JSONResponse(content=result.model_dump())
    except AttributeError:
        return JSONResponse(content=result.dict())


# ---------------------------------------------------------------------------
# /state - explicit override to preserve custom subclass fields
# ---------------------------------------------------------------------------

@app.get("/state", tags=["State Management"])
async def get_state_override():
    """Return the raw FinOpsState so extended fields like task_id aren't dropped."""
    env = _get_env()
    try:
        return JSONResponse(content=env.state.model_dump())
    except AttributeError:
        return JSONResponse(content=env.state.dict())
