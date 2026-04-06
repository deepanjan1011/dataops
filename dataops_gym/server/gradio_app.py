"""
Gradio Dashboard for DataOps Gym.
Mounted on the existing FastAPI app at path="/".
"""

import io
import random
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gradio as gr


# ─── Shared state for charts ────────────────────────────────────────────────

_reward_history: List[float] = []
_action_rewards: Dict[str, float] = {}
_curriculum_scores: List[dict] = []


def _reset_chart_state():
    global _reward_history, _action_rewards
    _reward_history.clear()
    _action_rewards.clear()


# ─── Matplotlib helpers ─────────────────────────────────────────────────────

def _fig_to_image(fig):
    """Convert a matplotlib figure to a PIL Image for Gradio."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    plt.close(fig)
    from PIL import Image
    return Image.open(buf)


def _make_reward_line_chart():
    fig, ax = plt.subplots(figsize=(8, 4))
    if _reward_history:
        cumulative = np.cumsum(_reward_history)
        ax.plot(range(1, len(cumulative) + 1), cumulative, marker="o", linewidth=2, color="#2563eb")
        ax.fill_between(range(1, len(cumulative) + 1), cumulative, alpha=0.15, color="#2563eb")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward Over Time")
    ax.grid(True, alpha=0.3)
    return _fig_to_image(fig)


def _make_action_bar_chart():
    fig, ax = plt.subplots(figsize=(8, 4))
    if _action_rewards:
        actions = list(_action_rewards.keys())
        rewards = list(_action_rewards.values())
        colors = ["#22c55e" if r >= 0 else "#ef4444" for r in rewards]
        ax.barh(actions, rewards, color=colors)
    ax.set_xlabel("Total Reward")
    ax.set_title("Reward by Action Type")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return _fig_to_image(fig)


def _make_health_heatmap(env):
    fig, ax = plt.subplots(figsize=(8, 4))
    try:
        df = env.dataframes.get("main")
        if df is not None and len(df.columns) > 0:
            health_data = []
            col_names = []
            for col in df.columns:
                null_pct = df[col].isna().mean()
                health_data.append([1.0 - null_pct])
                col_names.append(col)
            health_arr = np.array(health_data)
            im = ax.imshow(health_arr, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
            ax.set_yticks(range(len(col_names)))
            ax.set_yticklabels(col_names, fontsize=8)
            ax.set_xticks([0])
            ax.set_xticklabels(["Completeness"])
            fig.colorbar(im, ax=ax, label="Health (1.0 = no nulls)")
    except Exception:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Column Health Heatmap")
    fig.tight_layout()
    return _fig_to_image(fig)


def _make_curriculum_chart():
    fig, ax = plt.subplots(figsize=(8, 4))
    if _curriculum_scores:
        episodes = [s.get("episode", i + 1) for i, s in enumerate(_curriculum_scores)]
        scores = [s.get("score", 0) for s in _curriculum_scores]
        levels = [s.get("level", 1) for s in _curriculum_scores]
        ax.plot(episodes, scores, marker="o", linewidth=2, color="#2563eb", label="Score")
        ax2 = ax.twinx()
        ax2.plot(episodes, levels, marker="s", linewidth=1, color="#f59e0b", linestyle="--", label="Level")
        ax2.set_ylabel("Level", color="#f59e0b")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title("Curriculum Performance")
    ax.grid(True, alpha=0.3)
    return _fig_to_image(fig)


# ─── Dashboard factory ──────────────────────────────────────────────────────

def create_gradio_interface(env):
    """Build and return the Gradio Blocks interface."""

    from dataops_gym.models import DataOpsAction
    from dataops_gym.graders.grader import grade_by_criteria

    TASK_IDS = ["easy", "medium", "hard", "outlier_detection", "schema_migration",
                "drift_detection", "poisoning_detection"]

    CLEANING_ACTIONS = [
        "drop_nulls", "impute_missing", "drop_duplicates", "drop_column",
        "rename_column", "cast_type", "apply_regex", "format_date",
        "strip_whitespace", "filter_rows", "fill_value", "submit", "undo",
        "clip_outliers", "detect_outliers", "split_column", "map_values",
        "advance_stream", "analyze_distribution", "label_batch", "flag_rows",
    ]

    CORRUPTOR_ACTIONS = [
        "inject_nulls", "swap_values", "introduce_typos", "flip_labels", "inject_pii",
    ]

    STRATEGIES = ["mean", "median", "mode", "ffill", "bfill"]
    TARGET_TYPES = ["int", "float", "str", "datetime", "bool"]

    # ── Tab 1: Interactive Playground ────────────────────────────────────────

    def playground_reset(task_id, num_rows, null_pct, dup_rate, seed_val):
        _reset_chart_state()
        seed = int(seed_val) if seed_val else None
        kwargs = {}
        if null_pct and float(null_pct) > 0:
            kwargs["null_percentage"] = float(null_pct)
        if dup_rate and float(dup_rate) > 0:
            kwargs["duplicate_rate"] = float(dup_rate)
        obs = env.reset(task_id, seed=seed, num_rows=int(num_rows), **kwargs)
        df = env.dataframes.get("main", pd.DataFrame())
        columns = list(df.columns) if not df.empty else []
        return (
            df.head(20),
            f"{obs.data_health_score:.3f}",
            str(obs.step_number),
            obs.last_action_result or "Environment reset.",
            gr.update(choices=columns, value=columns[0] if columns else None),
            "0.0",
        )

    def playground_upload(file_info):
        if file_info is None:
            return gr.update(), "N/A", "0", "Upload cleared.", gr.update(), "0.0"

        _reset_chart_state()
        try:
            name = file_info.name
            if name.endswith('.csv'):
                df = pd.read_csv(name)
            elif name.endswith('.json'):
                df = pd.read_json(name)
            else:
                return gr.update(), "Error", "0", "Unsupported file", gr.update(), "0.0"

            import uuid
            env.current_task = "custom"
            env.step_count = 0
            env.cumulative_reward = 0.0
            env.done = False
            env.episode_id = str(uuid.uuid4())
            env._last_penalty = 0.0
            env._state_history = []
            env.dataframes = {"main": df}
            env.golden_df = None
            env.grading_criteria = {}
            env.stream_batches = []
            env.previous_health_score = env._calculate_health_score()
            env.last_action_result = "Custom dataset loaded."

            columns = list(df.columns) if not df.empty else []
            return (
                df.head(20),
                f"{env.previous_health_score:.3f}",
                str(env.step_count),
                env.last_action_result,
                gr.update(choices=columns, value=columns[0] if columns else None),
                "0.0",
            )
        except Exception as e:
            return gr.update(), "Error", "0", f"Error: {e}", gr.update(), "0.0"

    def playground_step(action_type, column_name, strategy, target_type, pattern,
                        replacement, new_name, fill_val, filter_cond, drift_label,
                        row_indices_str):
        kwargs = {"action_type": action_type}
        if column_name:
            kwargs["column_name"] = column_name
        if strategy:
            kwargs["strategy"] = strategy
        if target_type:
            kwargs["target_type"] = target_type
        if pattern:
            kwargs["pattern"] = pattern
        if replacement:
            kwargs["replacement"] = replacement
        if new_name:
            kwargs["new_name"] = new_name
        if fill_val:
            kwargs["fill_value"] = fill_val
        if filter_cond:
            kwargs["filter_condition"] = filter_cond
        if drift_label:
            kwargs["drift_label"] = drift_label
        if row_indices_str:
            try:
                kwargs["row_indices"] = [int(x.strip()) for x in row_indices_str.split(",") if x.strip()]
            except ValueError:
                pass

        action = DataOpsAction(**kwargs)
        obs = env.step(action)

        _reward_history.append(obs.reward)
        _action_rewards[action_type] = _action_rewards.get(action_type, 0) + obs.reward

        df = env.dataframes.get("main", pd.DataFrame())
        columns = list(df.columns) if not df.empty else []
        return (
            df.head(20),
            f"{obs.data_health_score:.3f}",
            str(obs.step_number),
            obs.last_action_result or (obs.error or ""),
            gr.update(choices=columns, value=columns[0] if columns else None),
            f"{env.cumulative_reward:.3f}",
        )

    def playground_grade():
        try:
            score = grade_by_criteria(
                env.current_task,
                env.dataframes.get("main", pd.DataFrame()),
                env.grading_criteria,
            )
            return f"Score: {score:.4f}"
        except Exception as e:
            return f"Grading error: {e}"

    # ── Tab 2: Reward Visualization ─────────────────────────────────────────

    def refresh_charts():
        return (
            _make_reward_line_chart(),
            _make_action_bar_chart(),
            _make_health_heatmap(env),
        )

    # ── Tab 3: Curriculum ───────────────────────────────────────────────────

    def curriculum_start():
        global _curriculum_scores
        _curriculum_scores = []
        from dataops_gym.server.app import curriculum_state, CURRICULUM_LEVELS
        import dataops_gym.server.app as app_module

        app_module.curriculum_state = type(app_module.curriculum_state)()
        params = dict(CURRICULUM_LEVELS[1])
        task_id = params.pop("task_id")
        app_module.curriculum_state.current_task = task_id
        app_module.curriculum_state.current_params = {"task_id": task_id, **params}
        env.reset(task_id, seed=random.randint(1, 99999), **params)
        app_module.curriculum_state.total_episodes += 1

        cs = app_module.curriculum_state
        return (
            f"Level: {cs.current_level}",
            f"Task: {cs.current_task}",
            f"Episodes: {cs.total_episodes}",
            f"Avg Score: {cs.average_score:.4f}",
            _make_curriculum_chart(),
        )

    def curriculum_next():
        import dataops_gym.server.app as app_module
        from dataops_gym.server.app import CURRICULUM_LEVELS

        cs = app_module.curriculum_state
        score = grade_by_criteria(
            env.current_task,
            env.dataframes.get("main", pd.DataFrame()),
            env.grading_criteria,
        )

        cs.history.append({
            "level": cs.current_level,
            "task": cs.current_task,
            "score": score,
            "episode": cs.total_episodes,
        })
        _curriculum_scores.append({"level": cs.current_level, "score": score, "episode": cs.total_episodes})

        all_scores = [h["score"] for h in cs.history]
        cs.average_score = round(sum(all_scores) / len(all_scores), 4)

        if score > 0.85:
            cs.current_level = min(10, cs.current_level + 1)
        elif score < 0.40:
            cs.current_level = max(1, cs.current_level - 1)

        params = dict(CURRICULUM_LEVELS[cs.current_level])
        task_id = params.pop("task_id")
        cs.current_task = task_id
        cs.current_params = {"task_id": task_id, **params}
        env.reset(task_id, seed=random.randint(1, 99999), **params)
        cs.total_episodes += 1

        return (
            f"Level: {cs.current_level}",
            f"Task: {cs.current_task}",
            f"Episodes: {cs.total_episodes}",
            f"Avg Score: {cs.average_score:.4f}",
            _make_curriculum_chart(),
        )

    # ── Tab 4: Adversarial ──────────────────────────────────────────────────

    def adversarial_start(num_rows, seed_val):
        import dataops_gym.server.app as app_module
        from dataops_gym.tasks.generators import generate_easy_dataset
        import uuid as _uuid

        seed = int(seed_val) if seed_val else None
        dirty_df, _ = generate_easy_dataset(
            seed=seed, num_rows=int(num_rows),
            null_percentage=0.0, duplicate_rate=0.0,
        )
        env.dataframes = {"main": dirty_df}
        env.current_task = "adversarial"
        env.step_count = 0
        env.done = False
        env.episode_id = str(_uuid.uuid4())
        env.cumulative_reward = 0.0

        from dataops_gym.models import AdversarialState
        app_module.adversarial_clean_snapshot = dirty_df.copy()
        app_module.adversarial_state = AdversarialState()

        st = app_module.adversarial_state
        df = env.dataframes.get("main", pd.DataFrame())
        return (
            f"Phase: {st.phase}",
            f"Round: {st.round_number}",
            f"Corruptor: {st.corruptor_score:.3f}  |  Cleaner: {st.cleaner_score:.3f}",
            df.head(20),
        )

    def adversarial_corrupt(action_type, column_name, inject_count):
        import dataops_gym.server.app as app_module

        st = app_module.adversarial_state
        if st is None or st.phase != "corrupt":
            df = env.dataframes.get("main", pd.DataFrame())
            phase = st.phase if st else "not started"
            return (f"Phase: {phase}", f"Round: {st.round_number if st else 0}",
                    f"Cannot corrupt in phase: {phase}", df.head(20))

        action = DataOpsAction(
            action_type=action_type,
            column_name=column_name,
            inject_count=int(inject_count) if inject_count else 5,
        )
        from dataops_gym.server.app import execute_corruption
        execute_corruption(action, env.dataframes["main"])
        st.round_number += 1
        st.corruptions_planted += 1

        if st.round_number >= st.max_rounds_per_phase:
            st.phase = "clean"
            st.round_number = 0
            app_module.adversarial_corrupted_snapshot = env.dataframes["main"].copy()

        df = env.dataframes.get("main", pd.DataFrame())
        return (
            f"Phase: {st.phase}",
            f"Round: {st.round_number}",
            f"Corruptor: {st.corruptor_score:.3f}  |  Cleaner: {st.cleaner_score:.3f}",
            df.head(20),
        )

    def adversarial_clean(action_type, column_name, strategy):
        import dataops_gym.server.app as app_module
        from dataops_gym.server.app import compare_dataframes

        st = app_module.adversarial_state
        if st is None or st.phase != "clean":
            df = env.dataframes.get("main", pd.DataFrame())
            phase = st.phase if st else "not started"
            return (f"Phase: {phase}", f"Round: {st.round_number if st else 0}",
                    f"Cannot clean in phase: {phase}", df.head(20))

        kwargs = {"action_type": action_type}
        if column_name:
            kwargs["column_name"] = column_name
        if strategy:
            kwargs["strategy"] = strategy
        action = DataOpsAction(**kwargs)
        env.step(action)
        st.round_number += 1

        if st.round_number >= st.max_rounds_per_phase:
            st.phase = "done"
            clean_match = compare_dataframes(env.dataframes["main"], app_module.adversarial_clean_snapshot)
            st.cleaner_score = clean_match
            st.corruptor_score = round(1.0 - clean_match, 4)

        df = env.dataframes.get("main", pd.DataFrame())
        return (
            f"Phase: {st.phase}",
            f"Round: {st.round_number}",
            f"Corruptor: {st.corruptor_score:.3f}  |  Cleaner: {st.cleaner_score:.3f}",
            df.head(20),
        )

    # ── Tab 5: Multi-Agent ──────────────────────────────────────────────────

    def multi_agent_start(task_id, num_agents, seed_val):
        import dataops_gym.server.app as app_module
        from dataops_gym.models import AgentAssignment, MultiAgentState

        seed = int(seed_val) if seed_val else None
        env.reset(task_id, seed=seed, num_rows=100)

        columns = list(env.dataframes["main"].columns)
        n = int(num_agents)
        chunk_size = max(1, len(columns) // n)
        responsibilities = [
            "null_handling", "type_fixing", "deduplication_and_cleanup",
            "format_standardization", "outlier_handling",
        ]
        agents = []
        for i in range(n):
            start = i * chunk_size
            end = start + chunk_size if i < n - 1 else len(columns)
            agents.append(AgentAssignment(
                agent_id=f"agent_{i+1}",
                responsibility=responsibilities[i % len(responsibilities)],
                assigned_columns=columns[start:end],
            ))

        app_module.multi_agent_state = MultiAgentState(agents=agents)
        ms = app_module.multi_agent_state

        agent_info = "\n".join(
            f"{a.agent_id}: {a.responsibility} -> {a.assigned_columns}"
            for a in ms.agents
        )
        agent_ids = [a.agent_id for a in ms.agents]
        return (
            agent_info,
            f"Coordination: {ms.coordination_score:.4f}",
            "No conflicts yet.",
            f"Steps: {ms.total_steps}",
            gr.update(choices=agent_ids, value=agent_ids[0] if agent_ids else None),
        )

    def multi_agent_step(agent_id, action_type, column_name):
        import dataops_gym.server.app as app_module

        ms = app_module.multi_agent_state
        if ms is None:
            return ("No session", "N/A", "Start a session first.", "Steps: 0", gr.update())

        agent = next((a for a in ms.agents if a.agent_id == agent_id), None)
        if not agent:
            return ("Agent not found", "N/A", f"Unknown agent: {agent_id}", f"Steps: {ms.total_steps}", gr.update())

        if column_name and column_name not in agent.assigned_columns:
            ms.conflicts.append({
                "agent": agent_id,
                "column": column_name,
                "assigned_to": next(
                    (a.agent_id for a in ms.agents if column_name in a.assigned_columns), "unknown"
                ),
                "step": ms.total_steps + 1,
            })

        kwargs = {"action_type": action_type}
        if column_name:
            kwargs["column_name"] = column_name
        action = DataOpsAction(**kwargs)
        obs = env.step(action)
        ms.total_steps += 1

        ms.action_log.append({
            "agent_id": agent_id,
            "action": action_type,
            "column": column_name,
            "result": obs.last_action_result,
            "step": ms.total_steps,
        })

        total = ms.total_steps
        conflicts = len(ms.conflicts)
        ms.coordination_score = round(1.0 - (conflicts / max(total, 1)), 4)

        agent_info = "\n".join(
            f"{a.agent_id}: {a.responsibility} -> {a.assigned_columns}"
            for a in ms.agents
        )
        conflict_log = "\n".join(
            f"Step {c['step']}: {c['agent']} touched {c['column']} (assigned to {c['assigned_to']})"
            for c in ms.conflicts
        ) if ms.conflicts else "No conflicts."

        return (
            agent_info,
            f"Coordination: {ms.coordination_score:.4f}",
            conflict_log,
            f"Steps: {ms.total_steps}",
            gr.update(),
        )

    # ── Build Gradio Blocks ─────────────────────────────────────────────────

    with gr.Blocks(title="DataOps Gym") as demo:
        gr.Markdown("# DataOps Gym Dashboard")
        gr.Markdown("Interactive RL environment for training AI agents on data engineering tasks.")

        with gr.Tab("Interactive Playground"):
            with gr.Row():
                with gr.Column(scale=1):
                    task_dd = gr.Dropdown(choices=TASK_IDS, value="easy", label="Task")
                    num_rows_sl = gr.Slider(10, 500, value=50, step=10, label="Rows")
                    null_pct_sl = gr.Slider(0.0, 0.5, value=0.08, step=0.01, label="Null %")
                    dup_rate_sl = gr.Slider(0.0, 0.3, value=0.10, step=0.01, label="Duplicate Rate")
                    seed_tb = gr.Textbox(label="Seed (optional)", value="")
                    reset_btn = gr.Button("Reset Environment", variant="primary")
                    gr.Markdown("---")
                    upload_file = gr.File(label="Upload Custom CSV / JSON", file_types=[".csv", ".json"])

                with gr.Column(scale=2):
                    health_tb = gr.Textbox(label="Health Score", interactive=False)
                    step_tb = gr.Textbox(label="Step", interactive=False)
                    reward_tb = gr.Textbox(label="Cumulative Reward", value="0.0", interactive=False)
                    result_tb = gr.Textbox(label="Last Action Result", interactive=False)

            data_preview = gr.Dataframe(label="Data Preview (first 20 rows)", interactive=False)

            gr.Markdown("### Execute Action")
            with gr.Row():
                action_dd = gr.Dropdown(choices=CLEANING_ACTIONS, value="drop_nulls", label="Action")
                col_dd = gr.Dropdown(choices=[], label="Column", allow_custom_value=True)
            with gr.Row():
                strategy_dd = gr.Dropdown(choices=STRATEGIES, label="Strategy", allow_custom_value=True)
                target_type_dd = gr.Dropdown(choices=TARGET_TYPES, label="Target Type", allow_custom_value=True)
                pattern_tb = gr.Textbox(label="Pattern")
                replacement_tb = gr.Textbox(label="Replacement")
            with gr.Row():
                new_name_tb = gr.Textbox(label="New Name")
                fill_val_tb = gr.Textbox(label="Fill Value")
                filter_cond_tb = gr.Textbox(label="Filter Condition")
                drift_label_dd = gr.Dropdown(choices=["normal", "drift"], label="Drift Label", allow_custom_value=True)
                row_indices_tb = gr.Textbox(label="Row Indices (comma-sep)")
            with gr.Row():
                exec_btn = gr.Button("Execute Action", variant="primary")
                grade_btn = gr.Button("Grade Episode")
                grade_result = gr.Textbox(label="Grade", interactive=False)

            reset_btn.click(
                playground_reset,
                inputs=[task_dd, num_rows_sl, null_pct_sl, dup_rate_sl, seed_tb],
                outputs=[data_preview, health_tb, step_tb, result_tb, col_dd, reward_tb],
            )
            upload_file.upload(
                playground_upload,
                inputs=[upload_file],
                outputs=[data_preview, health_tb, step_tb, result_tb, col_dd, reward_tb],
            )
            exec_btn.click(
                playground_step,
                inputs=[action_dd, col_dd, strategy_dd, target_type_dd, pattern_tb,
                        replacement_tb, new_name_tb, fill_val_tb, filter_cond_tb,
                        drift_label_dd, row_indices_tb],
                outputs=[data_preview, health_tb, step_tb, result_tb, col_dd, reward_tb],
            )
            grade_btn.click(playground_grade, outputs=[grade_result])

        with gr.Tab("Reward Visualization"):
            gr.Markdown("### Charts update after actions in the Playground tab")
            refresh_btn = gr.Button("Refresh Charts", variant="primary")
            reward_line_img = gr.Image(label="Cumulative Reward")
            action_bar_img = gr.Image(label="Reward by Action Type")
            health_heatmap_img = gr.Image(label="Column Health Heatmap")
            refresh_btn.click(
                refresh_charts,
                outputs=[reward_line_img, action_bar_img, health_heatmap_img],
            )

        with gr.Tab("Curriculum"):
            with gr.Row():
                cur_start_btn = gr.Button("Start Curriculum", variant="primary")
                cur_next_btn = gr.Button("Next Episode")
            with gr.Row():
                cur_level_tb = gr.Textbox(label="Level", interactive=False)
                cur_task_tb = gr.Textbox(label="Task", interactive=False)
                cur_episodes_tb = gr.Textbox(label="Episodes", interactive=False)
                cur_avg_tb = gr.Textbox(label="Avg Score", interactive=False)
            cur_chart_img = gr.Image(label="Performance History")
            cur_start_btn.click(
                curriculum_start,
                outputs=[cur_level_tb, cur_task_tb, cur_episodes_tb, cur_avg_tb, cur_chart_img],
            )
            cur_next_btn.click(
                curriculum_next,
                outputs=[cur_level_tb, cur_task_tb, cur_episodes_tb, cur_avg_tb, cur_chart_img],
            )

        with gr.Tab("Adversarial"):
            gr.Markdown("### Two-player adversarial data corruption game")
            with gr.Row():
                adv_rows_sl = gr.Slider(10, 200, value=30, step=5, label="Rows")
                adv_seed_tb = gr.Textbox(label="Seed", value="42")
                adv_start_btn = gr.Button("Start Adversarial", variant="primary")
            with gr.Row():
                adv_phase_tb = gr.Textbox(label="Phase", interactive=False)
                adv_round_tb = gr.Textbox(label="Round", interactive=False)
                adv_scores_tb = gr.Textbox(label="Scores", interactive=False)
            adv_preview = gr.Dataframe(label="Data Preview", interactive=False)

            gr.Markdown("#### Corruptor Panel")
            with gr.Row():
                cor_action_dd = gr.Dropdown(choices=CORRUPTOR_ACTIONS, value="inject_nulls", label="Corruption")
                cor_col_tb = gr.Textbox(label="Column")
                cor_count_tb = gr.Textbox(label="Inject Count", value="5")
                cor_btn = gr.Button("Corrupt!", variant="stop")

            gr.Markdown("#### Cleaner Panel")
            with gr.Row():
                cln_action_dd = gr.Dropdown(choices=CLEANING_ACTIONS[:13], value="impute_missing", label="Clean Action")
                cln_col_tb = gr.Textbox(label="Column")
                cln_strategy_dd = gr.Dropdown(choices=STRATEGIES, label="Strategy", value="mode")
                cln_btn = gr.Button("Clean!", variant="primary")

            adv_start_btn.click(
                adversarial_start,
                inputs=[adv_rows_sl, adv_seed_tb],
                outputs=[adv_phase_tb, adv_round_tb, adv_scores_tb, adv_preview],
            )
            cor_btn.click(
                adversarial_corrupt,
                inputs=[cor_action_dd, cor_col_tb, cor_count_tb],
                outputs=[adv_phase_tb, adv_round_tb, adv_scores_tb, adv_preview],
            )
            cln_btn.click(
                adversarial_clean,
                inputs=[cln_action_dd, cln_col_tb, cln_strategy_dd],
                outputs=[adv_phase_tb, adv_round_tb, adv_scores_tb, adv_preview],
            )

        with gr.Tab("Multi-Agent"):
            gr.Markdown("### Collaborative multi-agent data cleaning")
            with gr.Row():
                ma_task_dd = gr.Dropdown(choices=TASK_IDS, value="easy", label="Task")
                ma_agents_sl = gr.Slider(2, 5, value=3, step=1, label="Agents")
                ma_seed_tb = gr.Textbox(label="Seed", value="42")
                ma_start_btn = gr.Button("Start Multi-Agent", variant="primary")
            with gr.Row():
                ma_assignments_tb = gr.Textbox(label="Agent Assignments", lines=5, interactive=False)
                ma_coord_tb = gr.Textbox(label="Coordination Score", interactive=False)
            ma_conflicts_tb = gr.Textbox(label="Conflict Log", lines=4, interactive=False)
            ma_steps_tb = gr.Textbox(label="Total Steps", interactive=False)

            gr.Markdown("#### Agent Action")
            with gr.Row():
                ma_agent_dd = gr.Dropdown(choices=[], label="Agent", allow_custom_value=True)
                ma_action_dd = gr.Dropdown(choices=CLEANING_ACTIONS[:13], value="drop_nulls", label="Action")
                ma_col_tb = gr.Textbox(label="Column")
                ma_step_btn = gr.Button("Execute", variant="primary")

            ma_start_btn.click(
                multi_agent_start,
                inputs=[ma_task_dd, ma_agents_sl, ma_seed_tb],
                outputs=[ma_assignments_tb, ma_coord_tb, ma_conflicts_tb, ma_steps_tb, ma_agent_dd],
            )
            ma_step_btn.click(
                multi_agent_step,
                inputs=[ma_agent_dd, ma_action_dd, ma_col_tb],
                outputs=[ma_assignments_tb, ma_coord_tb, ma_conflicts_tb, ma_steps_tb, ma_agent_dd],
            )

    return demo
