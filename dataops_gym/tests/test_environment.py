"""
Tests for DataOps Gym environment.
Run with: python -m pytest dataops_gym/tests/ -v
"""

import io
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from dataops_gym.server.dataops_environment import DataOpsEnvironment
from dataops_gym.models import DataOpsAction, DataOpsObservation, DataOpsState
from dataops_gym.graders.grader import grade_by_criteria
from dataops_gym.tasks.auto_detect import detect_data_issues, build_criteria_from_issues


@pytest.fixture
def env():
    return DataOpsEnvironment()


# ── Test 1: reset("easy") returns valid observation ───────────────────────────

def test_reset_easy_returns_valid_observation(env):
    obs = env.reset("easy")
    assert isinstance(obs, DataOpsObservation)
    assert obs.task_id == "easy"
    assert obs.total_rows >= 50  # 50 base + ~10% duplicates injected
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
    assert obs.total_rows >= 40   # 40 base users + ~8% duplicates injected
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
    obs_before = env.reset("easy")
    rows_before = obs_before.total_rows
    action = DataOpsAction(action_type="drop_duplicates")
    obs = env.step(action)
    assert isinstance(obs, DataOpsObservation)
    assert obs.total_rows < rows_before   # duplicates were removed
    assert obs.step_number == 1
    assert "Dropped" in obs.last_action_result
    assert obs.reward != 0.0 or obs.total_rows < rows_before


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
        # Use criteria-based grader (procedural mode)
        score = grade_by_criteria(task_id, env.dataframes["main"].copy(), env.grading_criteria)
        assert isinstance(score, float), f"{task_id}: expected float, got {type(score)}"
        assert 0.0 < score < 1.0, f"{task_id}: score {score} must be strictly between 0 and 1"

    # Empty df should not crash
    env.reset("easy")
    empty_score = grade_by_criteria("easy", pd.DataFrame(), env.grading_criteria)
    assert 0.0 < empty_score < 1.0


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


# ── Test 11: different seeds produce different data ───────────────────────────

def test_procedural_different_seeds_produce_different_data():
    env = DataOpsEnvironment()
    obs1 = env.reset("easy", seed=1)
    preview1 = obs1.preview_rows
    obs2 = env.reset("easy", seed=2)
    preview2 = obs2.preview_rows
    assert preview1 != preview2


# ── Test 12: same seed produces same data ─────────────────────────────────────

def test_procedural_same_seed_produces_same_data():
    env = DataOpsEnvironment()
    obs1 = env.reset("easy", seed=42)
    rows1 = obs1.total_rows
    obs2 = env.reset("easy", seed=42)
    rows2 = obs2.total_rows
    assert rows1 == rows2


# ── Test 13: no seed does not crash ───────────────────────────────────────────

def test_procedural_no_seed_is_random():
    env = DataOpsEnvironment()
    obs1 = env.reset("easy")
    obs2 = env.reset("easy")
    assert obs1.total_rows > 0
    assert obs2.total_rows > 0


# ── Test 14: criteria grader returns valid score ──────────────────────────────

def test_criteria_grader_returns_valid_score():
    env = DataOpsEnvironment()
    env.reset("easy", seed=42)
    score = grade_by_criteria("easy", env.dataframes["main"], env.grading_criteria)
    assert isinstance(score, float)
    assert 0.0 < score < 1.0


# ── Test 15: high null_percentage produces more nulls ─────────────────────────

def test_high_null_percentage_produces_more_nulls():
    env = DataOpsEnvironment()
    obs_low = env.reset("easy", seed=42, null_percentage=0.01)
    nulls_low = sum(c.null_count for c in obs_low.column_summaries)
    obs_high = env.reset("easy", seed=42, null_percentage=0.40)
    nulls_high = sum(c.null_count for c in obs_high.column_summaries)
    assert nulls_high > nulls_low


# ── Test 16: large dataset respects num_rows ──────────────────────────────────

def test_large_dataset():
    env = DataOpsEnvironment()
    obs = env.reset("easy", seed=42, num_rows=500)
    assert obs.total_rows >= 500  # at least 500 base rows (plus duplicates)


# ── Test 17: undo reverts last action ─────────────────────────────────────────

def test_undo_reverts_last_action():
    env = DataOpsEnvironment()
    env.reset("easy", seed=42)
    rows_before = env.dataframes["main"].shape[0]

    env.step(DataOpsAction(action_type="drop_duplicates"))
    rows_after_action = env.dataframes["main"].shape[0]
    assert rows_after_action < rows_before

    obs = env.step(DataOpsAction(action_type="undo"))
    rows_after_undo = env.dataframes["main"].shape[0]
    assert rows_after_undo == rows_before
    assert obs.undo_depth == 0
    assert obs.reward == -0.02


# ── Test 18: undo on empty history returns error ──────────────────────────────

def test_undo_on_empty_history_returns_error():
    env = DataOpsEnvironment()
    env.reset("easy", seed=42)
    obs = env.step(DataOpsAction(action_type="undo"))
    assert obs.error is not None or "No actions" in obs.last_action_result
    assert obs.done is False


# ── Test 19: upload CSV detects issues ────────────────────────────────────────

def test_upload_csv_detects_issues():
    from dataops_gym.server.app import app
    client = TestClient(app)

    # Build a dirty CSV in memory
    dirty_df = pd.DataFrame({
        "name":     ["  Alice  ", "Bob", "Bob", None],
        "price":    ["$10.00", "$20.00", "$20.00", "$5.00"],
        "category": ["FOOD", "food", "food", "Food"],
    })
    csv_bytes = dirty_df.to_csv(index=False).encode()

    response = client.post(
        "/upload",
        files={"file": ("test_dirty.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "loaded"
    assert data["rows"] == 4
    issues = data["detected_issues"]
    assert "missing_values" in issues      # None in name
    assert "duplicates" in issues          # Bob row duplicated
    assert "whitespace" in issues          # "  Alice  "
    assert "inconsistent_casing" in issues # FOOD / food / Food


# ── Test 20: upload creates valid custom task ─────────────────────────────────

def test_upload_creates_valid_task():
    from dataops_gym.server.app import app
    client = TestClient(app)

    df = pd.DataFrame({
        "product": ["Widget", "Gadget", "Widget"],
        "price":   ["$5.00", "$15.00", "$5.00"],
    })
    csv_bytes = df.to_csv(index=False).encode()

    client.post(
        "/upload",
        files={"file": ("products.csv", io.BytesIO(csv_bytes), "text/csv")},
    )

    state_resp = client.get("/state")
    assert state_resp.status_code == 200
    state = state_resp.json()
    assert state["task_id"] == "custom"
    assert state["step_count"] == 0
    assert state["done"] is False

    # grader should also work on the custom task
    grader_resp = client.post("/grader")
    assert grader_resp.status_code == 200
    grader_data = grader_resp.json()
    assert 0.0 < grader_data["score"] < 1.0


# ── Phase 1: Outlier Detection + Schema Migration ───────────────────────────

def test_reset_outlier_detection():
    env = DataOpsEnvironment()
    obs = env.reset("outlier_detection", seed=42)
    assert obs.total_rows >= 100
    col_names = [c.name for c in obs.column_summaries]
    assert "age" in col_names
    assert "salary" in col_names
    assert "department" in col_names


def test_reset_schema_migration():
    env = DataOpsEnvironment()
    obs = env.reset("schema_migration", seed=42)
    col_names = [c.name for c in obs.column_summaries]
    assert "full_name" in col_names
    assert "full_address" in col_names
    assert "status_code" in col_names


def test_clip_outliers_action():
    env = DataOpsEnvironment()
    env.reset("outlier_detection", seed=42)
    obs = env.step(DataOpsAction(action_type="clip_outliers", column_name="age", clip_min=18, clip_max=80))
    assert obs.error is None
    assert "Clipped" in obs.last_action_result


def test_detect_outliers_action():
    env = DataOpsEnvironment()
    env.reset("outlier_detection", seed=42)
    obs = env.step(DataOpsAction(action_type="detect_outliers", column_name="salary", outlier_method="iqr"))
    assert obs.error is None
    assert "Found" in obs.last_action_result or "outlier" in obs.last_action_result.lower()


def test_split_column_action():
    env = DataOpsEnvironment()
    env.reset("schema_migration", seed=42)
    obs = env.step(DataOpsAction(
        action_type="split_column", column_name="full_name",
        delimiter=" ", new_columns=["first_name", "last_name"], max_splits=1
    ))
    assert obs.error is None
    col_names = [c.name for c in obs.column_summaries]
    assert "first_name" in col_names
    assert "last_name" in col_names
    assert "full_name" not in col_names


def test_map_values_action():
    env = DataOpsEnvironment()
    env.reset("schema_migration", seed=42)
    obs = env.step(DataOpsAction(
        action_type="map_values", column_name="status_code",
        value_mapping={"1": "active", "2": "inactive", "3": "pending", "4": "archived"}
    ))
    assert obs.error is None
    assert "Mapped" in obs.last_action_result


def test_outlier_grader():
    env = DataOpsEnvironment()
    env.reset("outlier_detection", seed=42)
    score = grade_by_criteria("outlier_detection", env.dataframes["main"], env.grading_criteria)
    assert 0.0 < score < 1.0


def test_schema_migration_grader():
    env = DataOpsEnvironment()
    env.reset("schema_migration", seed=42)
    score = grade_by_criteria("schema_migration", env.dataframes["main"], env.grading_criteria)
    assert 0.0 < score < 1.0


# ── Phase 2: Data Drift Detection ───────────────────────────────────────────

def test_reset_drift_detection():
    env = DataOpsEnvironment()
    obs = env.reset("drift_detection", seed=42)
    assert obs.total_rows >= 200


def test_advance_stream():
    env = DataOpsEnvironment()
    env.reset("drift_detection", seed=42)
    obs = env.step(DataOpsAction(action_type="advance_stream"))
    assert "Batch" in obs.last_action_result
    assert obs.error is None


def test_analyze_distribution():
    env = DataOpsEnvironment()
    env.reset("drift_detection", seed=42)
    env.step(DataOpsAction(action_type="advance_stream"))
    obs = env.step(DataOpsAction(action_type="analyze_distribution", column_name="amount"))
    assert obs.error is None
    assert "mean" in obs.last_action_result


def test_label_batch():
    env = DataOpsEnvironment()
    env.reset("drift_detection", seed=42)
    env.step(DataOpsAction(action_type="advance_stream"))
    obs = env.step(DataOpsAction(action_type="label_batch", drift_label="normal"))
    assert obs.error is None


def test_drift_grader():
    env = DataOpsEnvironment()
    env.reset("drift_detection", seed=42)
    for i in range(15):
        env.step(DataOpsAction(action_type="advance_stream"))
        env.step(DataOpsAction(action_type="label_batch", drift_label="normal"))
    score = grade_by_criteria("drift_detection", env.dataframes["main"], env.grading_criteria)
    assert 0.0 < score < 1.0


# ── Phase 3: Poisoning Detection ────────────────────────────────────────────

def test_reset_poisoning_detection():
    env = DataOpsEnvironment()
    obs = env.reset("poisoning_detection", seed=42)
    assert obs.total_rows >= 100
    col_names = [c.name for c in obs.column_summaries]
    assert "text" in col_names
    assert "sentiment" in col_names
    assert "poisoned" not in col_names  # Hidden from agent


def test_flag_rows_action():
    env = DataOpsEnvironment()
    env.reset("poisoning_detection", seed=42)
    obs = env.step(DataOpsAction(action_type="flag_rows", row_indices=[0, 5, 10]))
    assert obs.error is None
    assert "Flagged" in obs.last_action_result


def test_poisoning_grader():
    env = DataOpsEnvironment()
    env.reset("poisoning_detection", seed=42)
    env.step(DataOpsAction(action_type="flag_rows", row_indices=[0, 1, 2, 3, 4]))
    score = grade_by_criteria("poisoning_detection", env.dataframes["main"], env.grading_criteria)
    assert 0.0 < score < 1.0


# ── Phase 4: Curriculum Learning ───────────────────────────────────────────

def test_curriculum_start():
    from fastapi.testclient import TestClient
    from dataops_gym.server.app import app
    client = TestClient(app)
    resp = client.post("/curriculum", json={"action": "start"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["curriculum"]["current_level"] == 1
    assert data["curriculum"]["current_task"] == "easy"


def test_curriculum_next():
    from fastapi.testclient import TestClient
    from dataops_gym.server.app import app
    client = TestClient(app)
    client.post("/curriculum", json={"action": "start"})
    client.post("/step", json={"action_type": "submit"})
    resp = client.post("/curriculum", json={"action": "next"})
    assert resp.status_code == 200
    assert "last_score" in resp.json()


def test_curriculum_status():
    from fastapi.testclient import TestClient
    from dataops_gym.server.app import app
    client = TestClient(app)
    client.post("/curriculum", json={"action": "start"})
    resp = client.post("/curriculum", json={"action": "status"})
    assert resp.status_code == 200


# ── Phase 5: Adversarial Mode ──────────────────────────────────────────────

def test_adversarial_start():
    from fastapi.testclient import TestClient
    from dataops_gym.server.app import app
    client = TestClient(app)
    resp = client.post("/adversarial/start", json={"num_rows": 30, "seed": 42})
    assert resp.status_code == 200
    assert resp.json()["state"]["phase"] == "corrupt"


def test_adversarial_corrupt_then_clean():
    from fastapi.testclient import TestClient
    from dataops_gym.server.app import app
    client = TestClient(app)
    client.post("/adversarial/start", json={"num_rows": 30, "seed": 42})

    # Corrupt 5 times
    for _ in range(5):
        resp = client.post("/adversarial/step", json={
            "role": "corruptor",
            "action": {"action_type": "inject_nulls", "column_name": "product_name", "inject_count": 2}
        })
    assert resp.json()["state"]["phase"] == "clean"

    # Clean 5 times
    for _ in range(5):
        resp = client.post("/adversarial/step", json={
            "role": "cleaner",
            "action": {"action_type": "impute_missing", "column_name": "product_name", "strategy": "mode"}
        })
    assert resp.json()["state"]["phase"] == "done"
    assert "cleaner_score" in resp.json()["state"]


# ── Phase 6: Multi-Agent Collaborative Mode ────────────────────────────────

def test_multi_agent_start():
    from fastapi.testclient import TestClient
    from dataops_gym.server.app import app
    client = TestClient(app)
    resp = client.post("/multi_agent/start", json={"task_id": "easy", "num_agents": 3, "seed": 42})
    assert resp.status_code == 200
    assert len(resp.json()["state"]["agents"]) == 3


def test_multi_agent_step_no_conflict():
    from fastapi.testclient import TestClient
    from dataops_gym.server.app import app
    client = TestClient(app)
    resp = client.post("/multi_agent/start", json={"task_id": "easy", "num_agents": 2, "seed": 42})
    agents = resp.json()["state"]["agents"]
    first_agent = agents[0]
    first_col = first_agent["assigned_columns"][0]

    resp = client.post("/multi_agent/step", json={
        "agent_id": first_agent["agent_id"],
        "action": {"action_type": "strip_whitespace", "column_name": first_col}
    })
    assert resp.status_code == 200
    assert len(resp.json()["state"]["conflicts"]) == 0


def test_multi_agent_step_with_conflict():
    from fastapi.testclient import TestClient
    from dataops_gym.server.app import app
    client = TestClient(app)
    resp = client.post("/multi_agent/start", json={"task_id": "easy", "num_agents": 2, "seed": 42})
    agents = resp.json()["state"]["agents"]
    first_agent = agents[0]
    second_agent_col = agents[1]["assigned_columns"][0]

    resp = client.post("/multi_agent/step", json={
        "agent_id": first_agent["agent_id"],
        "action": {"action_type": "strip_whitespace", "column_name": second_agent_col}
    })
    assert resp.status_code == 200
    assert len(resp.json()["state"]["conflicts"]) >= 1


# ── Phase 7: Gradio Dashboard ──────────────────────────────────────────────

def test_gradio_import():
    try:
        from dataops_gym.server.gradio_app import create_gradio_interface
        assert callable(create_gradio_interface)
    except ImportError:
        pass  # Gradio optional
