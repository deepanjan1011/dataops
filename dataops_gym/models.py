from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum


# ─── ACTION TYPES ───
class ActionType(str, Enum):
    DROP_NULLS = "drop_nulls"
    IMPUTE_MISSING = "impute_missing"
    DROP_DUPLICATES = "drop_duplicates"
    DROP_COLUMN = "drop_column"
    RENAME_COLUMN = "rename_column"
    CAST_TYPE = "cast_type"
    APPLY_REGEX = "apply_regex"
    FORMAT_DATE = "format_date"
    STRIP_WHITESPACE = "strip_whitespace"
    MERGE_TABLES = "merge_tables"
    FILTER_ROWS = "filter_rows"
    FILL_VALUE = "fill_value"
    SUBMIT = "submit"  # Agent declares it's done


class DataOpsAction(BaseModel):
    """Action the agent takes on the dataset."""
    action_type: ActionType = Field(..., description="The type of data operation to perform")
    column_name: Optional[str] = Field(None, description="Target column name")

    # For impute_missing
    strategy: Optional[Literal["mean", "median", "mode", "ffill", "bfill"]] = Field(
        None, description="Imputation strategy"
    )

    # For cast_type
    target_type: Optional[Literal["int", "float", "str", "datetime", "bool"]] = Field(
        None, description="Target data type"
    )

    # For apply_regex
    pattern: Optional[str] = Field(None, description="Regex pattern to search for")
    replacement: Optional[str] = Field(None, description="Replacement string")

    # For format_date
    target_format: Optional[str] = Field(None, description="Target date format string")

    # For rename_column
    new_name: Optional[str] = Field(None, description="New column name")

    # For merge_tables
    right_table: Optional[str] = Field(None, description="Name of the right table to merge with")
    merge_on: Optional[str] = Field(None, description="Column name to merge on")
    merge_how: Optional[Literal["inner", "left", "right", "outer"]] = Field(
        "inner", description="Type of merge"
    )

    # For filter_rows
    filter_condition: Optional[str] = Field(None, description="Pandas query string for filtering")

    # For fill_value
    fill_value: Optional[str] = Field(None, description="Value to fill missing entries with")


# ─── OBSERVATION ───
class ColumnSummary(BaseModel):
    """Summary statistics for a single column."""
    name: str
    dtype: str
    null_count: int
    null_percentage: float
    unique_count: int
    sample_values: List[Any] = Field(default_factory=list, description="Up to 5 sample values")

    # Numeric columns only
    mean: Optional[float] = None
    min_val: Optional[Any] = None
    max_val: Optional[Any] = None


class DataOpsObservation(BaseModel):
    """What the agent sees after each action."""
    task_id: str = Field(..., description="Current task identifier")
    task_description: str = Field(..., description="Natural language description of the task")
    step_number: int = Field(0, description="Current step in the episode")
    total_rows: int = Field(..., description="Total rows in the current dataset")
    total_columns: int = Field(..., description="Total columns in the current dataset")
    column_summaries: List[ColumnSummary] = Field(default_factory=list, description="Per-column stats")
    preview_rows: List[Dict[str, Any]] = Field(default_factory=list, description="First 5 rows as dicts")
    data_health_score: float = Field(0.0, description="Overall data cleanliness 0.0-1.0")
    available_tables: List[str] = Field(default_factory=list, description="List of available table names")
    last_action_result: str = Field("", description="Success/error message from last action")
    reward: float = Field(0.0, description="Reward from last action")
    done: bool = Field(False, description="Whether the episode is over")
    error: Optional[str] = Field(None, description="Error message if last action failed")


# ─── STATE ───
class DataOpsState(BaseModel):
    """Internal environment state."""
    episode_id: str = Field("", description="Unique episode identifier")
    task_id: str = Field("", description="Current task")
    step_count: int = Field(0, description="Steps taken so far")
    max_steps: int = Field(30, description="Maximum steps allowed")
    cumulative_reward: float = Field(0.0, description="Total reward accumulated")
    done: bool = Field(False, description="Episode finished")
