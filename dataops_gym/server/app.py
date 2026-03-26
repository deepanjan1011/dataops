"""
FastAPI server for DataOps Gym.
Exposes all required OpenEnv endpoints on port 7860.
"""

import os
import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from dataops_gym.models import DataOpsAction, DataOpsObservation, DataOpsState
from dataops_gym.server.dataops_environment import DataOpsEnvironment
from dataops_gym.graders.grader import grade

logger = logging.getLogger(__name__)

# ─── APP SETUP ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="DataOps Gym",
    description="An RL environment for training AI agents on real-world data engineering tasks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance (stateful per-process)
env = DataOpsEnvironment()


# ─── REQUEST MODELS ──────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "easy"


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _safe_json(obj: Any) -> Any:
    """Recursively make an object JSON-serializable."""
    import numpy as np
    import math

    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return obj


def _obs_to_dict(obs: DataOpsObservation) -> Dict[str, Any]:
    """Convert observation to a clean JSON-safe dict."""
    return _safe_json(obs.model_dump())


def _state_to_dict(state: DataOpsState) -> Dict[str, Any]:
    return _safe_json(state.model_dump())


# ─── ENDPOINTS ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "DataOps Gym",
        "description": "AI Data Quality & Curation Environment for OpenEnv",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "tasks": "/tasks",
            "reset": "/reset (POST)",
            "step": "/step (POST)",
            "state": "/state",
            "grader": "/grader (POST)",
            "baseline": "/baseline (POST)",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/tasks")
def tasks():
    action_schema = DataOpsAction.model_json_schema()
    task_list = [
        {
            "task_id": "easy",
            "description": (
                "Clean a messy product sales dataset: fix types, remove duplicates, "
                "handle nulls, standardize formats."
            ),
            "difficulty": "easy",
            "max_steps": 30,
            "action_schema": action_schema,
        },
        {
            "task_id": "medium",
            "description": (
                "Clean and merge two related tables (users + purchases): standardize IDs, "
                "fix dates, merge correctly without losing active users."
            ),
            "difficulty": "medium",
            "max_steps": 30,
            "action_schema": action_schema,
        },
        {
            "task_id": "hard",
            "description": (
                "Redact all PII (emails, phone numbers, credit cards, SSNs) from "
                "web-scraped text documents while preserving non-PII content."
            ),
            "difficulty": "hard",
            "max_steps": 30,
            "action_schema": action_schema,
        },
    ]
    return {"tasks": task_list}


@app.post("/reset")
def reset(request: ResetRequest):
    try:
        obs = env.reset(request.task_id)
        return JSONResponse(content=_obs_to_dict(obs))
    except Exception as e:
        logger.exception("Error in /reset")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(action: DataOpsAction):
    try:
        obs = env.step(action)
        return JSONResponse(content=_obs_to_dict(obs))
    except Exception as e:
        logger.exception("Error in /step")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    try:
        s = env.state()
        return JSONResponse(content=_state_to_dict(s))
    except Exception as e:
        logger.exception("Error in /state")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/grader")
def grader():
    try:
        if not env.current_task:
            raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")

        main_df = env.dataframes.get("main")
        if main_df is None:
            raise HTTPException(status_code=400, detail="No dataframe in current episode.")

        score = grade(env.current_task, main_df.copy(), env.golden_df)

        details = {
            "rows_in_final": int(len(main_df)),
            "rows_in_golden": int(len(env.golden_df)) if env.golden_df is not None else 0,
            "null_count_final": int(main_df.isna().sum().sum()),
            "null_count_golden": int(env.golden_df.isna().sum().sum()) if env.golden_df is not None else 0,
            "step_count": env.step_count,
            "cumulative_reward": float(env.cumulative_reward),
            "episode_done": env.done,
        }

        return {
            "task_id": env.current_task,
            "score": score,
            "details": details,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /grader")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/baseline")
def baseline():
    """Run the baseline LLM agent on all 3 tasks. Requires OPENAI_API_KEY."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}

    try:
        from dataops_gym.baseline.inference import run_all_tasks
        scores = run_all_tasks()
        return _safe_json(scores)
    except Exception as e:
        logger.exception("Error in /baseline")
        raise HTTPException(status_code=500, detail=str(e))
