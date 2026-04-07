"""
FastAPI server for DataOps Gym.
Exposes all required OpenEnv endpoints on port 7860.
"""

import io
import os
import logging
import random
import uuid
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import numpy as np

from dataops_gym.models import (
    DataOpsAction, DataOpsObservation, DataOpsState,
    CurriculumRequest, CurriculumState,
    AdversarialStartRequest, AdversarialStepRequest, AdversarialState,
    MultiAgentStartRequest, MultiAgentStepRequest, AgentAssignment, MultiAgentState,
)
from dataops_gym.server.dataops_environment import DataOpsEnvironment
from dataops_gym.graders.grader import grade, grade_by_criteria

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
    seed: Optional[int] = None
    num_rows: Optional[int] = 50
    num_users: Optional[int] = 40
    num_purchases: Optional[int] = 60
    num_docs: Optional[int] = 30
    null_percentage: Optional[float] = None   # 0.0–0.5
    duplicate_rate: Optional[float] = None    # 0.0–0.3
    pii_density: Optional[float] = None       # 0.0–1.0 (hard task only)
    outlier_rate: Optional[float] = None      # 0.0–0.3 (outlier_detection)
    legitimate_extreme_rate: Optional[float] = None  # 0.0–0.1 (outlier_detection)
    migration_complexity: Optional[float] = None     # 0.0–1.0 (schema_migration)
    drift_severity: Optional[float] = None           # 0.0–1.0 (drift_detection)
    poison_rate: Optional[float] = None              # 0.0–0.5 (poisoning_detection)
    num_stream_batches: Optional[int] = None         # (drift_detection)
    drift_start_batch: Optional[int] = None          # (drift_detection)


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

# @app.get("/")
# def root():
#     return {
#         "name": "DataOps Gym",
#         "description": "AI Data Quality & Curation Environment for OpenEnv",
#         "version": "1.0.0",
#         "endpoints": {
#             "health": "/health",
#             "tasks": "/tasks",
#             "reset": "/reset (POST)",
#             "step": "/step (POST)",
#             "state": "/state",
#             "grader": "/grader (POST)",
#             "baseline": "/baseline (POST)",
#             "docs": "/docs",
#         },
#     }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    """OpenEnv metadata endpoint."""
    return {
        "name": "dataops_gym",
        "description": "AI Data Quality & Curation Environment for OpenEnv — 7 graded data engineering tasks",
        "version": "2.0.0",
        "author": "deepanjan1011",
        "url": "https://huggingface.co/spaces/deepanjan1011/dataops-gym",
        "tags": ["openenv", "data-cleaning", "data-engineering", "rl-environment"],
    }


@app.get("/schema")
def schema():
    """OpenEnv schema endpoint — returns action, observation, and state schemas."""
    return {
        "action": DataOpsAction.model_json_schema(),
        "observation": DataOpsObservation.model_json_schema(),
        "state": DataOpsState.model_json_schema(),
    }


@app.post("/mcp")
def mcp_endpoint(request_body: dict = {}):
    """Minimal MCP (Model Context Protocol) JSON-RPC endpoint."""
    method = request_body.get("method", "")
    rpc_id = request_body.get("id", 1)

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "dataops_gym", "version": "2.0.0"},
            },
        }

    # Default: return capabilities
    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "result": {
            "name": "dataops_gym",
            "version": "2.0.0",
            "capabilities": ["reset", "step", "state", "grader"],
        },
    }


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
            "configurable_params": {
                "num_rows": {"default": 50, "min": 10, "max": 1000, "description": "Base row count before duplicates"},
                "null_percentage": {"default": 0.08, "min": 0.0, "max": 0.5, "description": "Fraction of cells nulled per column"},
                "duplicate_rate": {"default": 0.10, "min": 0.0, "max": 0.3, "description": "Fraction of duplicate rows injected"},
                "format_inconsistency": {"default": 0.5, "min": 0.0, "max": 1.0, "description": "Date format variety (0=all YYYY-MM-DD, 1=max variety)"},
                "seed": {"default": None, "description": "Set for reproducibility"},
            },
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
            "configurable_params": {
                "num_users": {"default": 40, "min": 10, "max": 500, "description": "Number of user rows"},
                "num_purchases": {"default": 60, "min": 10, "max": 1000, "description": "Number of purchase rows"},
                "null_percentage": {"default": 0.05, "min": 0.0, "max": 0.5, "description": "Fraction of cells nulled per column"},
                "duplicate_rate": {"default": 0.08, "min": 0.0, "max": 0.3, "description": "Fraction of duplicate user rows injected"},
                "id_format_variety": {"default": 0.7, "min": 0.0, "max": 1.0, "description": "Fraction of user_ids with non-plain formats"},
                "seed": {"default": None, "description": "Set for reproducibility"},
            },
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
            "configurable_params": {
                "num_docs": {"default": 30, "min": 5, "max": 500, "description": "Number of text documents"},
                "pii_density": {"default": 0.3, "min": 0.0, "max": 1.0, "description": "Fraction of docs containing PII"},
                "pii_variety": {"default": 0.5, "min": 0.0, "max": 1.0, "description": "How many PII types appear per doc"},
                "seed": {"default": None, "description": "Set for reproducibility"},
            },
        },
        {
            "task_id": "outlier_detection",
            "description": (
                "Detect and handle outliers in an employee dataset. Distinguish genuine errors "
                "from legitimate extreme values using context (e.g., executive salaries)."
            ),
            "difficulty": "medium-hard",
            "max_steps": 30,
            "action_schema": action_schema,
            "configurable_params": {
                "num_rows": {"default": 100, "min": 20, "max": 1000, "description": "Number of employee rows"},
                "outlier_rate": {"default": 0.08, "min": 0.0, "max": 0.3, "description": "Fraction of rows with planted outliers"},
                "legitimate_extreme_rate": {"default": 0.03, "min": 0.0, "max": 0.1, "description": "Fraction of rows with legitimate extremes"},
                "seed": {"default": None, "description": "Set for reproducibility"},
            },
        },
        {
            "task_id": "schema_migration",
            "description": (
                "Restructure a dataset: split combined columns (name, address, datetime), "
                "standardize phone numbers, separate price/currency, map status codes to strings."
            ),
            "difficulty": "hard",
            "max_steps": 30,
            "action_schema": action_schema,
            "configurable_params": {
                "num_rows": {"default": 60, "min": 10, "max": 500, "description": "Number of rows"},
                "migration_complexity": {"default": 0.5, "min": 0.0, "max": 1.0, "description": "Complexity of schema migration"},
                "seed": {"default": None, "description": "Set for reproducibility"},
            },
        },
        {
            "task_id": "drift_detection",
            "description": (
                "Detect data drift in a streaming e-commerce dataset. Analyze each incoming batch "
                "against historical data and label it as 'normal' or 'drift'."
            ),
            "difficulty": "hard",
            "max_steps": 60,
            "action_schema": action_schema,
            "configurable_params": {
                "num_historical_rows": {"default": 200, "min": 50, "max": 1000, "description": "Historical baseline rows"},
                "num_stream_batches": {"default": 15, "min": 5, "max": 50, "description": "Number of stream batches"},
                "drift_start_batch": {"default": 8, "min": 1, "max": 50, "description": "Batch index where drift starts"},
                "drift_severity": {"default": 0.5, "min": 0.0, "max": 1.0, "description": "How severe the drift is"},
                "seed": {"default": None, "description": "Set for reproducibility"},
            },
        },
        {
            "task_id": "poisoning_detection",
            "description": (
                "Detect poisoned samples in a sentiment classification dataset. Find label flips, "
                "subtle mislabels, and trigger phrase injections without flagging clean data."
            ),
            "difficulty": "very hard",
            "max_steps": 30,
            "action_schema": action_schema,
            "configurable_params": {
                "num_rows": {"default": 100, "min": 20, "max": 500, "description": "Number of rows"},
                "poison_rate": {"default": 0.10, "min": 0.0, "max": 0.5, "description": "Fraction of poisoned rows"},
                "seed": {"default": None, "description": "Set for reproducibility"},
            },
        },
    ]
    return {"tasks": task_list}


@app.post("/reset")
def reset(request: ResetRequest):
    try:
        kwargs = {
            "num_users": request.num_users or 40,
            "num_purchases": request.num_purchases or 60,
            "num_docs": request.num_docs or 30,
        }
        if request.null_percentage is not None:
            kwargs["null_percentage"] = request.null_percentage
        if request.duplicate_rate is not None:
            kwargs["duplicate_rate"] = request.duplicate_rate
        if request.pii_density is not None:
            kwargs["pii_density"] = request.pii_density
        if request.outlier_rate is not None:
            kwargs["outlier_rate"] = request.outlier_rate
        if request.legitimate_extreme_rate is not None:
            kwargs["legitimate_extreme_rate"] = request.legitimate_extreme_rate
        if request.migration_complexity is not None:
            kwargs["migration_complexity"] = request.migration_complexity
        if request.drift_severity is not None:
            kwargs["drift_severity"] = request.drift_severity
        if request.num_stream_batches is not None:
            kwargs["num_stream_batches"] = request.num_stream_batches
        if request.drift_start_batch is not None:
            kwargs["drift_start_batch"] = request.drift_start_batch
        if request.poison_rate is not None:
            kwargs["poison_rate"] = request.poison_rate

        obs = env.reset(
            request.task_id,
            seed=request.seed,
            num_rows=request.num_rows or 50,
            **kwargs,
        )
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

        # Prefer criteria-based grading (procedural mode), fall back to golden
        if env.grading_criteria:
            score = grade_by_criteria(env.current_task, main_df.copy(), env.grading_criteria)
            grading_mode = "criteria"
        else:
            score = grade(env.current_task, main_df.copy(), env.golden_df)
            grading_mode = "golden"

        details = {
            "rows_in_final": int(len(main_df)),
            "rows_in_golden": int(len(env.golden_df)) if env.golden_df is not None else 0,
            "null_count_final": int(main_df.isna().sum().sum()),
            "null_count_golden": int(env.golden_df.isna().sum().sum()) if env.golden_df is not None else 0,
            "step_count": env.step_count,
            "cumulative_reward": float(env.cumulative_reward),
            "episode_done": env.done,
            "grading_mode": grading_mode,
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


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a custom CSV or JSON dataset. The environment auto-detects data quality
    issues and creates a cleaning task around it.
    Accepts: .csv or .json files (max 10 MB).
    Returns: detected issues, task config, and initial observation.
    """
    from dataops_gym.tasks.auto_detect import detect_data_issues, build_criteria_from_issues

    MAX_BYTES = 10 * 1024 * 1024  # 10 MB

    content = await file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10 MB.")

    filename = file.filename or ""
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith(".json"):
            df = pd.read_json(io.BytesIO(content))
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Upload a .csv or .json file.",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(df) > 10_000:
        df = df.head(10_000)

    issues = detect_data_issues(df)
    criteria = build_criteria_from_issues(df, issues)

    # Load into the shared environment as a "custom" episode
    env.dataframes = {"main": df.copy()}
    env.grading_criteria = criteria
    env.golden_df = None
    env.current_task = "custom"
    env.step_count = 0
    env.done = False
    env.episode_id = str(uuid.uuid4())
    env.cumulative_reward = 0.0
    env._last_penalty = 0.0
    env._state_history = []
    env.previous_health_score = env._calculate_health_score()
    env.last_action_result = "Custom dataset loaded."

    obs = env._build_observation()
    obs.reward = 0.0

    return _safe_json({
        "status": "loaded",
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "detected_issues": issues,
        "task_description": (
            f"Clean this custom dataset. Detected issues: "
            + (", ".join(issues.keys()) if issues else "none")
        ),
        "observation": obs.model_dump(),
    })


# ─── CURRICULUM LEARNING ────────────────────────────────────────────────────

CURRICULUM_LEVELS = {
    1:  {"task_id": "easy", "num_rows": 30, "null_percentage": 0.05, "duplicate_rate": 0.05},
    2:  {"task_id": "easy", "num_rows": 50, "null_percentage": 0.10, "duplicate_rate": 0.10},
    3:  {"task_id": "easy", "num_rows": 100, "null_percentage": 0.20, "duplicate_rate": 0.15},
    4:  {"task_id": "medium", "num_rows": 40, "null_percentage": 0.05},
    5:  {"task_id": "medium", "num_rows": 80, "null_percentage": 0.15},
    6:  {"task_id": "outlier_detection", "num_rows": 80, "outlier_rate": 0.05},
    7:  {"task_id": "hard", "num_docs": 30, "pii_density": 0.2},
    8:  {"task_id": "schema_migration", "num_rows": 60, "migration_complexity": 0.5},
    9:  {"task_id": "drift_detection", "drift_severity": 0.3},
    10: {"task_id": "poisoning_detection", "num_rows": 150, "poison_rate": 0.15},
}

curriculum_state = CurriculumState()


@app.post("/curriculum")
def curriculum(request: CurriculumRequest):
    global curriculum_state

    try:
        if request.action == "start":
            curriculum_state = CurriculumState()
            params = dict(CURRICULUM_LEVELS[1])
            task_id = params.pop("task_id")
            curriculum_state.current_task = task_id
            curriculum_state.current_params = {"task_id": task_id, **params}
            obs = env.reset(task_id, seed=random.randint(1, 99999), **params)
            curriculum_state.total_episodes += 1
            return _safe_json({"curriculum": curriculum_state.model_dump(), "observation": obs.model_dump()})

        elif request.action == "next":
            score = grade_by_criteria(
                env.current_task,
                env.dataframes.get("main", pd.DataFrame()),
                env.grading_criteria,
            )

            curriculum_state.history.append({
                "level": curriculum_state.current_level,
                "task": curriculum_state.current_task,
                "score": score,
                "episode": curriculum_state.total_episodes,
            })

            all_scores = [h["score"] for h in curriculum_state.history]
            curriculum_state.average_score = round(sum(all_scores) / len(all_scores), 4)

            if score > request.score_threshold:
                curriculum_state.current_level = min(10, curriculum_state.current_level + 1)
            elif score < request.score_floor:
                curriculum_state.current_level = max(1, curriculum_state.current_level - 1)

            params = dict(CURRICULUM_LEVELS[curriculum_state.current_level])
            task_id = params.pop("task_id")
            curriculum_state.current_task = task_id
            curriculum_state.current_params = {"task_id": task_id, **params}
            obs = env.reset(task_id, seed=random.randint(1, 99999), **params)
            curriculum_state.total_episodes += 1

            return _safe_json({
                "curriculum": curriculum_state.model_dump(),
                "observation": obs.model_dump(),
                "last_score": score,
            })

        elif request.action == "status":
            return _safe_json(curriculum_state.model_dump())

        elif request.action == "reset":
            curriculum_state = CurriculumState()
            return _safe_json({"message": "Curriculum reset", "curriculum": curriculum_state.model_dump()})

    except Exception as e:
        logger.exception("Error in /curriculum")
        raise HTTPException(status_code=500, detail=str(e))


# ─── ADVERSARIAL MODE ──────────────────────────────────────────────────────

adversarial_state = None
adversarial_clean_snapshot = None
adversarial_corrupted_snapshot = None


def execute_corruption(action: DataOpsAction, df: pd.DataFrame) -> str:
    """Execute a corruptor action on the dataframe."""
    if action.action_type == "inject_nulls":
        count = min(action.inject_count or 5, len(df))
        indices = np.random.choice(len(df), size=count, replace=False)
        if action.column_name and action.column_name in df.columns:
            df.loc[indices, action.column_name] = None
        return f"Injected {count} nulls in {action.column_name}"

    elif action.action_type == "introduce_typos":
        if action.column_name and action.column_name in df.columns:
            rate = action.typo_rate or 0.1
            mask = np.random.random(len(df)) < rate
            def add_typo(s):
                if not isinstance(s, str) or len(s) < 2:
                    return s
                pos = random.randint(0, len(s) - 1)
                return s[:pos] + random.choice('abcdefghijklmnop') + s[pos + 1:]
            df.loc[mask, action.column_name] = df.loc[mask, action.column_name].apply(add_typo)
        return f"Introduced typos in {action.column_name}"

    elif action.action_type == "swap_values":
        if action.column_name and action.column_name in df.columns:
            count = min(action.inject_count or 5, len(df) // 2)
            for _ in range(count):
                i, j = random.sample(range(len(df)), 2)
                df.at[i, action.column_name], df.at[j, action.column_name] = \
                    df.at[j, action.column_name], df.at[i, action.column_name]
        return f"Swapped {count} pairs in {action.column_name}"

    elif action.action_type == "flip_labels":
        if action.column_name and action.column_name in df.columns:
            count = min(action.inject_count or 5, len(df))
            indices = np.random.choice(len(df), size=count, replace=False)
            unique_vals = df[action.column_name].dropna().unique().tolist()
            if len(unique_vals) > 1:
                for idx in indices:
                    current = df.at[idx, action.column_name]
                    others = [v for v in unique_vals if v != current]
                    if others:
                        df.at[idx, action.column_name] = random.choice(others)
        return f"Flipped {count} labels in {action.column_name}"

    elif action.action_type == "inject_pii":
        if action.column_name and action.column_name in df.columns:
            count = min(action.inject_count or 3, len(df))
            indices = np.random.choice(len(df), size=count, replace=False)
            pii_templates = [
                "contact john@example.com",
                "call (555) 123-4567",
                "card 4532-1234-5678-9012",
            ]
            for idx in indices:
                current = str(df.at[idx, action.column_name])
                df.at[idx, action.column_name] = current + " " + random.choice(pii_templates)
        return f"Injected PII into {count} rows of {action.column_name}"

    return "Unknown corruption action"


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """Compare two dataframes and return a similarity score 0.0-1.0."""
    if df1.shape != df2.shape:
        row_ratio = min(len(df1), len(df2)) / max(len(df1), len(df2)) if max(len(df1), len(df2)) > 0 else 0.0
        col_ratio = min(len(df1.columns), len(df2.columns)) / max(len(df1.columns), len(df2.columns)) if max(len(df1.columns), len(df2.columns)) > 0 else 0.0
        shape_penalty = (row_ratio + col_ratio) / 2
    else:
        shape_penalty = 1.0

    # Compare on shared columns and min rows
    shared_cols = list(set(df1.columns) & set(df2.columns))
    if not shared_cols:
        return 0.0
    min_rows = min(len(df1), len(df2))
    if min_rows == 0:
        return 0.0

    matches = 0
    total = 0
    for col in shared_cols:
        for i in range(min_rows):
            total += 1
            v1 = df1[col].iloc[i]
            v2 = df2[col].iloc[i]
            if pd.isna(v1) and pd.isna(v2):
                matches += 1
            elif str(v1) == str(v2):
                matches += 1

    cell_match = matches / total if total > 0 else 0.0
    return round(cell_match * shape_penalty, 4)


@app.post("/adversarial/start")
def adversarial_start(request: AdversarialStartRequest):
    global adversarial_state, adversarial_clean_snapshot
    from dataops_gym.tasks.generators import generate_easy_dataset

    dirty_df, _ = generate_easy_dataset(
        seed=request.seed, num_rows=request.num_rows,
        null_percentage=0.0, duplicate_rate=0.0,
    )

    env.dataframes = {"main": dirty_df}
    env.current_task = "adversarial"
    env.step_count = 0
    env.done = False
    env.episode_id = str(uuid.uuid4())
    env.cumulative_reward = 0.0

    adversarial_clean_snapshot = dirty_df.copy()
    adversarial_state = AdversarialState()

    return _safe_json({"state": adversarial_state.model_dump(), "observation": env._build_observation().model_dump()})


@app.post("/adversarial/step")
def adversarial_step(request: AdversarialStepRequest):
    global adversarial_state, adversarial_corrupted_snapshot

    if adversarial_state is None:
        return _safe_json({"error": "Start adversarial mode first with POST /adversarial/start"})

    if adversarial_state.phase == "done":
        return _safe_json({"error": "Adversarial episode is done. Start a new one."})

    if request.role == "corruptor" and adversarial_state.phase == "corrupt":
        execute_corruption(request.action, env.dataframes["main"])
        adversarial_state.round_number += 1
        adversarial_state.corruptions_planted += 1

        if adversarial_state.round_number >= adversarial_state.max_rounds_per_phase:
            adversarial_state.phase = "clean"
            adversarial_state.round_number = 0
            adversarial_corrupted_snapshot = env.dataframes["main"].copy()

    elif request.role == "cleaner" and adversarial_state.phase == "clean":
        env.step(request.action)
        adversarial_state.round_number += 1

        if adversarial_state.round_number >= adversarial_state.max_rounds_per_phase:
            adversarial_state.phase = "done"
            clean_match = compare_dataframes(env.dataframes["main"], adversarial_clean_snapshot)
            adversarial_state.cleaner_score = clean_match
            adversarial_state.corruptor_score = round(1.0 - clean_match, 4)

    else:
        return _safe_json({"error": f"Wrong role for current phase. Phase: {adversarial_state.phase}"})

    return _safe_json({"state": adversarial_state.model_dump(), "observation": env._build_observation().model_dump()})


# ─── MULTI-AGENT COLLABORATIVE MODE ────────────────────────────────────────

multi_agent_state = None


@app.post("/multi_agent/start")
def multi_agent_start(request: MultiAgentStartRequest):
    global multi_agent_state
    env.reset(request.task_id, seed=request.seed, num_rows=request.num_rows)

    columns = list(env.dataframes["main"].columns)
    chunk_size = max(1, len(columns) // request.num_agents)

    agents = []
    responsibilities = [
        "null_handling", "type_fixing", "deduplication_and_cleanup",
        "format_standardization", "outlier_handling",
    ]

    for i in range(request.num_agents):
        start = i * chunk_size
        end = start + chunk_size if i < request.num_agents - 1 else len(columns)
        agents.append(AgentAssignment(
            agent_id=f"agent_{i+1}",
            responsibility=responsibilities[i % len(responsibilities)],
            assigned_columns=columns[start:end],
        ))

    multi_agent_state = MultiAgentState(agents=agents)

    return _safe_json({
        "state": multi_agent_state.model_dump(),
        "observation": env._build_observation().model_dump(),
    })


@app.post("/multi_agent/step")
def multi_agent_step(request: MultiAgentStepRequest):
    global multi_agent_state
    if multi_agent_state is None:
        return _safe_json({"error": "Start multi-agent mode first"})

    agent = next((a for a in multi_agent_state.agents if a.agent_id == request.agent_id), None)
    if not agent:
        return _safe_json({"error": f"Unknown agent: {request.agent_id}. Valid: {[a.agent_id for a in multi_agent_state.agents]}"})

    # Conflict detection
    if request.action.column_name and request.action.column_name not in agent.assigned_columns:
        multi_agent_state.conflicts.append({
            "agent": request.agent_id,
            "column": request.action.column_name,
            "assigned_to": next(
                (a.agent_id for a in multi_agent_state.agents
                 if request.action.column_name in a.assigned_columns), "unknown"
            ),
            "step": multi_agent_state.total_steps + 1,
        })

    obs = env.step(request.action)
    multi_agent_state.total_steps += 1

    multi_agent_state.action_log.append({
        "agent_id": request.agent_id,
        "action": request.action.action_type,
        "column": request.action.column_name,
        "result": obs.last_action_result,
        "step": multi_agent_state.total_steps,
    })

    # Coordination score
    total = multi_agent_state.total_steps
    conflicts = len(multi_agent_state.conflicts)
    multi_agent_state.coordination_score = round(1.0 - (conflicts / max(total, 1)), 4)

    return _safe_json({
        "state": multi_agent_state.model_dump(),
        "observation": obs.model_dump(),
    })


@app.get("/multi_agent/status")
def multi_agent_status():
    if multi_agent_state is None:
        return _safe_json({"error": "No active multi-agent session"})
    return _safe_json(multi_agent_state.model_dump())


# ─── BASELINE ───────────────────────────────────────────────────────────────

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


# ─── GRADIO DASHBOARD ─────────────────────────────────────────────────────

try:
    import gradio as gr
    from dataops_gym.server.gradio_app import create_gradio_interface
    demo = create_gradio_interface(env)
    app = gr.mount_gradio_app(app, demo, path="/")
except ImportError:
    # Gradio not installed — API-only mode
    pass
