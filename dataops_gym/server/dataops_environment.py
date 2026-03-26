"""
Core environment logic for DataOps Gym.
Manages dataset state and executes agent actions.
"""

import os
import json
import uuid
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from dataops_gym.models import (
    ActionType,
    DataOpsAction,
    DataOpsObservation,
    DataOpsState,
    ColumnSummary,
)

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "tasks", "datasets")

TASK_DESCRIPTIONS = {
    "easy": (
        "Clean a messy product sales dataset: strip whitespace, remove duplicates, "
        "fix price strings to floats, standardize date formats to YYYY-MM-DD, "
        "lowercase categories, and impute missing values."
    ),
    "medium": (
        "Clean and merge two related tables (users + purchases): standardize user_id "
        "formats to plain integers, fix dates, convert amount strings to floats, "
        "remove duplicates, retain only active users, then merge the tables."
    ),
    "hard": (
        "Redact all PII (emails, phone numbers, credit card numbers, SSNs) from "
        "web-scraped text documents using regex, while preserving all non-PII content."
    ),
}


def _to_python(val: Any) -> Any:
    """Recursively convert numpy scalar types to Python native types."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    if pd.isna(val) if not isinstance(val, (list, dict, str)) else False:
        return None
    return val


def _safe(val: Any) -> Any:
    """Make a value JSON-safe."""
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return _to_python(val)


class DataOpsEnvironment:
    """RL environment that simulates a data engineering workflow."""

    def __init__(self):
        self.current_task: str = ""
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.golden_df: Optional[pd.DataFrame] = None
        self.step_count: int = 0
        self.max_steps: int = 30
        self.episode_id: str = ""
        self.cumulative_reward: float = 0.0
        self.done: bool = False
        self.previous_health_score: float = 0.0
        self.last_action_result: str = ""
        self._last_penalty: float = 0.0  # penalty flags set during action execution

    # ─── PUBLIC API ──────────────────────────────────────────────────────────

    def reset(self, task_id: str = "easy") -> DataOpsObservation:
        """Load task data, reset state, return initial observation."""
        if task_id not in ("easy", "medium", "hard"):
            task_id = "easy"

        self.current_task = task_id
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.done = False
        self.last_action_result = "Episode started."
        self.episode_id = str(uuid.uuid4())
        self._last_penalty = 0.0
        self.dataframes = {}

        self._load_datasets(task_id)

        health = self._calculate_health_score()
        self.previous_health_score = health

        obs = self._build_observation()
        obs.reward = 0.0
        obs.last_action_result = "Episode started."
        return obs

    def step(self, action: DataOpsAction) -> DataOpsObservation:
        """Execute one action and return the new observation."""
        if self.done:
            obs = self._build_observation()
            obs.error = "Episode is already done. Call reset() to start a new episode."
            obs.reward = 0.0
            return obs

        self.step_count += 1
        self._last_penalty = 0.0

        old_health = self.previous_health_score

        # Handle SUBMIT
        if action.action_type == ActionType.SUBMIT:
            self.last_action_result = "Episode submitted for grading."
            self.done = True
            new_health = self._calculate_health_score()
            reward = self._calculate_reward(old_health, new_health, action)
            self.cumulative_reward += reward
            self.previous_health_score = new_health
            obs = self._build_observation()
            obs.reward = reward
            obs.done = True
            return obs

        # Execute action
        result_msg = self._execute_action(action)
        self.last_action_result = result_msg

        new_health = self._calculate_health_score()
        reward = self._calculate_reward(old_health, new_health, action)
        self.cumulative_reward += reward
        self.previous_health_score = new_health

        # Auto-done at max steps
        if self.step_count >= self.max_steps:
            self.done = True

        obs = self._build_observation()
        obs.reward = reward
        return obs

    def state(self) -> DataOpsState:
        """Return current episode state."""
        return DataOpsState(
            episode_id=self.episode_id,
            task_id=self.current_task,
            step_count=self.step_count,
            max_steps=self.max_steps,
            cumulative_reward=self.cumulative_reward,
            done=self.done,
        )

    # ─── DATASET LOADING ─────────────────────────────────────────────────────

    def _load_datasets(self, task_id: str) -> None:
        datasets_dir = os.path.abspath(DATASETS_DIR)

        if task_id == "easy":
            dirty = pd.read_csv(os.path.join(datasets_dir, "easy_dirty.csv"))
            self.dataframes["main"] = dirty.copy()
            self.golden_df = pd.read_csv(os.path.join(datasets_dir, "easy_golden.csv"))

        elif task_id == "medium":
            users = pd.read_csv(os.path.join(datasets_dir, "medium_users_dirty.csv"))
            purchases = pd.read_csv(os.path.join(datasets_dir, "medium_purchases_dirty.csv"))
            self.dataframes["main"] = users.copy()
            self.dataframes["purchases"] = purchases.copy()
            self.golden_df = pd.read_csv(os.path.join(datasets_dir, "medium_golden.csv"))

        elif task_id == "hard":
            with open(os.path.join(datasets_dir, "hard_dirty.json")) as f:
                docs = json.load(f)
            self.dataframes["main"] = pd.DataFrame(docs)
            with open(os.path.join(datasets_dir, "hard_golden.json")) as f:
                golden_docs = json.load(f)
            self.golden_df = pd.DataFrame(golden_docs)

    # ─── ACTION EXECUTION ────────────────────────────────────────────────────

    def _execute_action(self, action: DataOpsAction) -> str:
        """Dispatch action and return a result message. Never raises."""
        try:
            return self._dispatch(action)
        except Exception as e:
            self._last_penalty += 0.1  # invalid action penalty
            return f"ERROR: {str(e)}"

    def _dispatch(self, action: DataOpsAction) -> str:
        df = self.dataframes["main"].copy()
        col = action.action_type

        # ── DROP_NULLS ──
        if action.action_type == ActionType.DROP_NULLS:
            column_name = action.column_name
            if column_name is None:
                raise ValueError("column_name is required for drop_nulls")
            if column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found. Available: {list(df.columns)}")
            before = len(df)
            df = df.dropna(subset=[column_name]).reset_index(drop=True)
            dropped = before - len(df)
            self.dataframes["main"] = df
            return f"Dropped {dropped} rows with null values in '{column_name}'"

        # ── IMPUTE_MISSING ──
        elif action.action_type == ActionType.IMPUTE_MISSING:
            column_name = action.column_name
            strategy = action.strategy
            if column_name is None:
                raise ValueError("column_name is required for impute_missing")
            if column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found")
            if strategy is None:
                raise ValueError("strategy is required for impute_missing")

            before_nulls = df[column_name].isna().sum()

            if strategy == "mean":
                if not pd.api.types.is_numeric_dtype(df[column_name]):
                    raise ValueError(f"Column '{column_name}' is not numeric; cannot use mean imputation")
                fill = df[column_name].mean()
                df[column_name] = df[column_name].fillna(fill)
            elif strategy == "median":
                if not pd.api.types.is_numeric_dtype(df[column_name]):
                    raise ValueError(f"Column '{column_name}' is not numeric; cannot use median imputation")
                fill = df[column_name].median()
                df[column_name] = df[column_name].fillna(fill)
            elif strategy == "mode":
                fill = df[column_name].mode()
                if len(fill) == 0:
                    raise ValueError(f"Column '{column_name}' has no mode (all nulls?)")
                df[column_name] = df[column_name].fillna(fill[0])
            elif strategy == "ffill":
                df[column_name] = df[column_name].ffill()
            elif strategy == "bfill":
                df[column_name] = df[column_name].bfill()

            imputed = before_nulls - df[column_name].isna().sum()
            self.dataframes["main"] = df
            return f"Imputed {imputed} missing values in '{column_name}' using {strategy}"

        # ── DROP_DUPLICATES ──
        elif action.action_type == ActionType.DROP_DUPLICATES:
            before = len(df)
            df = df.drop_duplicates().reset_index(drop=True)
            dropped = before - len(df)
            self.dataframes["main"] = df
            return f"Dropped {dropped} duplicate rows"

        # ── DROP_COLUMN ──
        elif action.action_type == ActionType.DROP_COLUMN:
            column_name = action.column_name
            if column_name is None:
                raise ValueError("column_name is required for drop_column")
            if column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found")
            # Penalty if column exists in golden
            if self.golden_df is not None and column_name in self.golden_df.columns:
                self._last_penalty += 0.5
            df = df.drop(columns=[column_name])
            self.dataframes["main"] = df
            return f"Dropped column '{column_name}'"

        # ── RENAME_COLUMN ──
        elif action.action_type == ActionType.RENAME_COLUMN:
            column_name = action.column_name
            new_name = action.new_name
            if column_name is None or new_name is None:
                raise ValueError("column_name and new_name are required for rename_column")
            if column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found")
            df = df.rename(columns={column_name: new_name})
            self.dataframes["main"] = df
            return f"Renamed '{column_name}' to '{new_name}'"

        # ── CAST_TYPE ──
        elif action.action_type == ActionType.CAST_TYPE:
            column_name = action.column_name
            target_type = action.target_type
            if column_name is None or target_type is None:
                raise ValueError("column_name and target_type are required for cast_type")
            if column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found")

            if target_type == "float":
                df[column_name] = (
                    df[column_name].astype(str)
                    .str.replace(r"[\$,]", "", regex=True)
                    .str.strip()
                )
                df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
            elif target_type == "int":
                df[column_name] = pd.to_numeric(df[column_name], errors="coerce").astype("Int64")
            elif target_type == "str":
                df[column_name] = df[column_name].astype(str)
            elif target_type == "datetime":
                df[column_name] = pd.to_datetime(df[column_name], errors="coerce")
            elif target_type == "bool":
                df[column_name] = df[column_name].astype(bool)

            self.dataframes["main"] = df
            return f"Cast '{column_name}' to {target_type}"

        # ── APPLY_REGEX ──
        elif action.action_type == ActionType.APPLY_REGEX:
            column_name = action.column_name
            pattern = action.pattern
            replacement = action.replacement
            if column_name is None or pattern is None or replacement is None:
                raise ValueError("column_name, pattern, and replacement are required for apply_regex")
            if column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found")

            before = df[column_name].astype(str).copy()
            df[column_name] = df[column_name].astype(str).str.replace(pattern, replacement, regex=True)
            changed = (before != df[column_name]).sum()
            self.dataframes["main"] = df
            return f"Applied regex on '{column_name}': replaced '{pattern}' with '{replacement}' ({changed} cells changed)"

        # ── FORMAT_DATE ──
        elif action.action_type == ActionType.FORMAT_DATE:
            column_name = action.column_name
            target_format = action.target_format
            if column_name is None or target_format is None:
                raise ValueError("column_name and target_format are required for format_date")
            if column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found")

            df[column_name] = pd.to_datetime(df[column_name], errors="coerce")
            df[column_name] = df[column_name].dt.strftime(target_format)
            self.dataframes["main"] = df
            return f"Formatted '{column_name}' dates to {target_format}"

        # ── STRIP_WHITESPACE ──
        elif action.action_type == ActionType.STRIP_WHITESPACE:
            column_name = action.column_name
            if column_name is None:
                raise ValueError("column_name is required for strip_whitespace")
            if column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found")

            df[column_name] = df[column_name].astype(str).str.strip()
            # Convert "nan" strings back to NaN
            df[column_name] = df[column_name].replace("nan", np.nan)
            self.dataframes["main"] = df
            return f"Stripped whitespace from '{column_name}'"

        # ── MERGE_TABLES ──
        elif action.action_type == ActionType.MERGE_TABLES:
            right_table = action.right_table
            merge_on = action.merge_on
            merge_how = action.merge_how or "inner"
            if right_table is None or merge_on is None:
                raise ValueError("right_table and merge_on are required for merge_tables")
            if right_table not in self.dataframes:
                raise ValueError(f"Table '{right_table}' not found. Available: {list(self.dataframes.keys())}")

            left_df = self.dataframes["main"]
            right_df = self.dataframes[right_table]

            if merge_on not in left_df.columns:
                raise ValueError(f"Column '{merge_on}' not found in main table")
            if merge_on not in right_df.columns:
                raise ValueError(f"Column '{merge_on}' not found in '{right_table}' table")

            merged = left_df.merge(right_df, on=merge_on, how=merge_how)
            self.dataframes["main"] = merged.reset_index(drop=True)
            return f"Merged 'main' with '{right_table}' on '{merge_on}' ({merge_how} join): {len(merged)} rows"

        # ── FILTER_ROWS ──
        elif action.action_type == ActionType.FILTER_ROWS:
            filter_condition = action.filter_condition
            if filter_condition is None:
                raise ValueError("filter_condition is required for filter_rows")

            before = len(df)
            filtered = df.query(filter_condition)
            after = len(filtered)

            # Penalty if losing more than 20% of rows
            if before > 0 and (before - after) / before > 0.2:
                self._last_penalty += 0.3

            self.dataframes["main"] = filtered.reset_index(drop=True)
            return f"Filtered rows: {before} -> {after}"

        # ── FILL_VALUE ──
        elif action.action_type == ActionType.FILL_VALUE:
            column_name = action.column_name
            fill_value = action.fill_value
            if column_name is None or fill_value is None:
                raise ValueError("column_name and fill_value are required for fill_value")
            if column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found")

            before_nulls = df[column_name].isna().sum()
            df[column_name] = df[column_name].fillna(fill_value)
            filled = before_nulls - df[column_name].isna().sum()
            self.dataframes["main"] = df
            return f"Filled {filled} null values in '{column_name}' with '{fill_value}'"

        else:
            raise ValueError(f"Unknown action type: {action.action_type}")

    # ─── HEALTH SCORE ─────────────────────────────────────────────────────────

    def _calculate_health_score(self) -> float:
        """Score 0.0–1.0 measuring how close current state is to golden."""
        if not self.dataframes or "main" not in self.dataframes:
            return 0.0

        df = self.dataframes["main"]
        golden = self.golden_df

        if df.empty or golden is None or golden.empty:
            return 0.0

        total_cells = df.shape[0] * df.shape[1]
        if total_cells == 0:
            return 0.0

        # 1. Null ratio (0.3 weight) — fewer nulls is better
        total_nulls = df.isna().sum().sum()
        null_score = (1.0 - total_nulls / total_cells) * 0.3

        # 2. Type correctness (0.3 weight) — columns matching golden dtypes
        common_cols = [c for c in golden.columns if c in df.columns]
        if common_cols:
            matching_types = sum(
                1 for c in common_cols
                if _dtype_category(df[c].dtype) == _dtype_category(golden[c].dtype)
            )
            type_score = (matching_types / len(golden.columns)) * 0.3
        else:
            type_score = 0.0

        # 3. Row retention (0.2 weight) — row count close to golden
        golden_rows = len(golden)
        current_rows = len(df)
        row_score = min(current_rows / golden_rows, 1.0) * 0.2 if golden_rows > 0 else 0.0

        # 4. Duplicate ratio (0.2 weight) — fewer duplicates is better
        if len(df) > 0:
            num_dupes = len(df) - len(df.drop_duplicates())
            dup_score = (1.0 - num_dupes / len(df)) * 0.2
        else:
            dup_score = 0.0

        total = null_score + type_score + row_score + dup_score
        return float(min(max(total, 0.0), 1.0))

    # ─── REWARD ───────────────────────────────────────────────────────────────

    def _calculate_reward(
        self, old_health: float, new_health: float, action: DataOpsAction
    ) -> float:
        """Compute reward for the current step."""
        # Base: improvement in health score
        base = (new_health - old_health) * 2.0

        # Accumulated penalties from _execute_action
        penalty = -self._last_penalty

        # Per-step efficiency penalty
        step_penalty = -0.01

        # Bonus: good submit
        bonus = 0.0
        if action.action_type == ActionType.SUBMIT and new_health > 0.9:
            bonus = 0.5

        # PII bonus for hard task
        if self.current_task == "hard" and action.action_type == ActionType.APPLY_REGEX:
            if self.golden_df is not None and "text" in self.dataframes.get("main", pd.DataFrame()).columns:
                bonus += self._calculate_pii_bonus()

        total = base + penalty + step_penalty + bonus
        return float(min(max(total, -1.0), 1.0))

    def _calculate_pii_bonus(self) -> float:
        """Give bonus reward for each PII type successfully redacted in hard task."""
        import re
        df = self.dataframes["main"]
        golden = self.golden_df

        if golden is None or "text" not in df.columns or "text" not in golden.columns:
            return 0.0

        pii_patterns = {
            "email": r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
            "phone": r"\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}",
            "cc": r"\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        }

        bonus = 0.0
        current_texts = df["text"].astype(str).tolist()
        golden_texts = golden["text"].astype(str).tolist()

        for pii_type, pattern in pii_patterns.items():
            golden_has_redacted = any("[REDACTED]" in t for t in golden_texts)
            current_has_original_pii = any(bool(re.search(pattern, t)) for t in current_texts)

            if golden_has_redacted and not current_has_original_pii:
                bonus += 0.2

        return min(bonus, 0.8)

    # ─── OBSERVATION BUILDER ──────────────────────────────────────────────────

    def _build_observation(self) -> DataOpsObservation:
        """Construct a full observation from current state."""
        df = self.dataframes.get("main", pd.DataFrame())

        column_summaries = []
        for col in df.columns:
            series = df[col]
            null_count = int(series.isna().sum())
            total = len(series)
            null_pct = round(null_count / total, 4) if total > 0 else 0.0

            sample_raw = series.dropna().head(5).tolist()
            sample_values = [_safe(v) for v in sample_raw]

            summary = ColumnSummary(
                name=col,
                dtype=str(series.dtype),
                null_count=null_count,
                null_percentage=null_pct,
                unique_count=int(series.nunique(dropna=True)),
                sample_values=sample_values,
            )

            if pd.api.types.is_numeric_dtype(series):
                clean = series.dropna()
                if len(clean) > 0:
                    summary.mean = _safe(clean.mean())
                    summary.min_val = _safe(clean.min())
                    summary.max_val = _safe(clean.max())

            column_summaries.append(summary)

        # Preview rows — convert all values to JSON-safe types
        preview = []
        for row in df.head(5).to_dict(orient="records"):
            safe_row = {k: _safe(v) for k, v in row.items()}
            preview.append(safe_row)

        health = self._calculate_health_score()

        error_msg = None
        if self.last_action_result.startswith("ERROR:"):
            error_msg = self.last_action_result

        return DataOpsObservation(
            task_id=self.current_task,
            task_description=TASK_DESCRIPTIONS.get(self.current_task, ""),
            step_number=self.step_count,
            total_rows=int(len(df)),
            total_columns=int(len(df.columns)),
            column_summaries=column_summaries,
            preview_rows=preview,
            data_health_score=round(health, 4),
            available_tables=list(self.dataframes.keys()),
            last_action_result=self.last_action_result,
            reward=0.0,  # caller sets this
            done=self.done,
            error=error_msg,
        )


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _dtype_category(dtype) -> str:
    """Bucket dtype into broad category for comparison."""
    if pd.api.types.is_integer_dtype(dtype):
        return "int"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    return "str"
