"""
Tests for DataOps Gym environment.
Run with: python -m pytest dataops_gym/tests/ -v
"""

import pytest
import pandas as pd
from dataops_gym.server.dataops_environment import DataOpsEnvironment
from dataops_gym.models import DataOpsAction, DataOpsObservation, DataOpsState
from dataops_gym.graders.grader import grade


@pytest.fixture
def env():
    return DataOpsEnvironment()


# ── Test 1: reset("easy") returns valid observation ───────────────────────────

def test_reset_easy_returns_valid_observation(env):
    obs = env.reset("easy")
    assert isinstance(obs, DataOpsObservation)
    assert obs.task_id == "easy"
    assert obs.total_rows == 50
    assert obs.total_columns == 5
    assert len(obs.column_summaries) == 5
    assert 0.0 <= obs.data_health_score <= 1.0
    assert obs.step_number == 0
    assert obs.done is False
    col_names = [c.name for c in obs.column_summaries]
    assert "product_name" in col_names
    assert "price" in col_names
    assert "quantity" in col_names
    assert "date_sold" in col_names
    assert "category" in col_names


# ── Test 2: reset("medium") loads two tables ──────────────────────────────────

def test_reset_medium_loads_two_tables(env):
    obs = env.reset("medium")
    assert isinstance(obs, DataOpsObservation)
    assert obs.task_id == "medium"
    assert "main" in obs.available_tables
    assert "purchases" in obs.available_tables
    assert len(obs.available_tables) == 2
    assert obs.total_rows == 40   # users table
    assert obs.total_columns == 5


# ── Test 3: reset("hard") loads text documents ────────────────────────────────

def test_reset_hard_loads_text_documents(env):
    obs = env.reset("hard")
    assert isinstance(obs, DataOpsObservation)
    assert obs.task_id == "hard"
    assert obs.total_rows == 30
    col_names = [c.name for c in obs.column_summaries]
    assert "id" in col_names
    assert "text" in col_names
    assert obs.total_columns == 2


# ── Test 4: valid action returns updated observation ──────────────────────────

def test_valid_action_returns_updated_observation(env):
    env.reset("easy")
    action = DataOpsAction(action_type="drop_duplicates")
    obs = env.step(action)
    assert isinstance(obs, DataOpsObservation)
    assert obs.total_rows == 45          # 5 duplicates removed
    assert obs.step_number == 1
    assert "Dropped 5" in obs.last_action_result
    assert obs.reward != 0.0 or obs.total_rows < 50  # something changed


# ── Test 5: invalid action returns error but does not crash ───────────────────

def test_invalid_action_returns_error_no_crash(env):
    env.reset("easy")
    bad_action = DataOpsAction(action_type="drop_nulls", column_name="nonexistent_column")
    obs = env.step(bad_action)
    assert isinstance(obs, DataOpsObservation)
    assert obs.error is not None
    assert "ERROR" in obs.error
    assert "nonexistent_column" in obs.error or "not found" in obs.error.lower()
    assert obs.done is False            # episode still alive


# ── Test 6: submit action marks episode as done ───────────────────────────────

def test_submit_marks_episode_done(env):
    env.reset("easy")
    submit = DataOpsAction(action_type="submit")
    obs = env.step(submit)
    assert obs.done is True
    assert "submitted" in obs.last_action_result.lower()
    # Further steps should return done with error
    extra = env.step(DataOpsAction(action_type="drop_duplicates"))
    assert extra.done is True
    assert extra.error is not None


# ── Test 7: grader returns float in [0.0, 1.0] for each task ─────────────────

def test_grader_returns_valid_score_all_tasks(env):
    for task_id in ["easy", "medium", "hard"]:
        env.reset(task_id)
        score = grade(task_id, env.dataframes["main"].copy(), env.golden_df)
        assert isinstance(score, float), f"{task_id}: expected float, got {type(score)}"
        assert 0.0 <= score <= 1.0, f"{task_id}: score {score} out of [0,1]"

    # Golden vs golden should be near-perfect
    env.reset("easy")
    perfect = grade("easy", env.golden_df.copy(), env.golden_df)
    assert perfect >= 0.95

    # Empty df should not crash
    empty_score = grade("easy", pd.DataFrame(), env.golden_df)
    assert 0.0 <= empty_score <= 1.0


# ── Test 8: health score changes after valid actions ─────────────────────────

def test_health_score_changes_after_valid_action(env):
    obs_before = env.reset("easy")
    health_before = obs_before.data_health_score

    # drop_duplicates should help health
    obs_after = env.step(DataOpsAction(action_type="drop_duplicates"))
    health_after = obs_after.data_health_score

    # Health should change (improve or stay same — not get worse from deduplication)
    assert health_after >= health_before - 0.01   # allow tiny float drift


# ── Test 9: step count increments correctly ───────────────────────────────────

def test_step_count_increments(env):
    env.reset("easy")
    assert env.step_count == 0

    for i in range(1, 4):
        env.step(DataOpsAction(action_type="drop_duplicates"))
        assert env.step_count == i

    state = env.state()
    assert isinstance(state, DataOpsState)
    assert state.step_count == 3


# ── Test 10: max steps triggers auto-done ─────────────────────────────────────

def test_max_steps_triggers_auto_done(env):
    env.reset("easy")
    env.max_steps = 3   # override for speed

    for _ in range(3):
        obs = env.step(DataOpsAction(action_type="strip_whitespace", column_name="product_name"))

    assert obs.done is True
    assert env.step_count >= env.max_steps
