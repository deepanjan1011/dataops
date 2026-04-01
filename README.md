---
title: DataOps Gym
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
pinned: false
---

<div align="center">

# 🧹 DataOps Gym

### A Research-Grade RL Environment for AI Data Engineering Agents

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-brightgreen?style=flat-square)](https://openenv.dev)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://python.org)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://hub.docker.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Gradio](https://img.shields.io/badge/Gradio-Dashboard-ff7c00?style=flat-square&logo=gradio&logoColor=white)](https://gradio.app)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow?style=flat-square)](LICENSE)

**[Live Demo](https://huggingface.co/spaces/deepanjan1011/dataops-gym)** · **[API Docs](https://deepanjan1011-dataops-gym.hf.space/docs)** · **[Try It Now](#quick-start)**

</div>

---

## The Problem

**Data engineers spend 60-80% of their time cleaning data** — not building models. Yet there is no standardised benchmark environment where AI agents can learn, practice, and be evaluated on real data engineering workflows.

DataOps Gym fills that gap.

---

## What Is DataOps Gym?

DataOps Gym is a fully OpenEnv-compliant reinforcement learning environment that puts an AI agent in the role of a data engineer. The agent receives **messy, real-world-style datasets** and must clean them using programmatic actions — guided by a dense reward signal that measures progress toward clean data.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           DataOps Gym v2.0                               │
│                                                                          │
│   ┌──────────┐    observation     ┌──────────────────────────────┐      │
│   │          │ ◄─────────────── │                              │      │
│   │  Agent   │                   │  FastAPI Environment         │      │
│   │  (LLM /  │ ──── action ───► │  (pandas DataFrames)         │      │
│   │   RL)    │                   │                              │      │
│   │          │ ◄── reward+done ─ │  Grader + Reward Function    │      │
│   └──────────┘                   └──────────────────────────────┘      │
│                                                                          │
│   7 Tasks: easy · medium · hard · outlier · schema · drift · poisoning  │
│   Modes:  curriculum · adversarial · multi-agent · custom upload         │
│   Dashboard: Gradio UI at root (/) with 5 interactive tabs              │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

| Feature | Description |
|---|---|
| **7 graded tasks** | Easy → Hard + Outlier Detection, Schema Migration, Drift Detection, Poisoning Detection |
| **27 action types** | Full pandas-like API: cast, merge, regex, impute, clip outliers, split columns, flag rows, and more |
| **Curriculum learning** | 10 difficulty levels with automatic progression based on performance |
| **Adversarial mode** | Two-player game: corruptor injects mess, cleaner fixes it |
| **Multi-agent mode** | Collaborative cleaning with column assignments and conflict detection |
| **Gradio dashboard** | 5-tab interactive UI: Playground, Rewards, Curriculum, Adversarial, Multi-Agent |
| **Procedural generation** | Fresh data every episode; `seed=42` for reproducibility |
| **Configurable difficulty** | Tune `null_percentage`, `duplicate_rate`, `poison_rate`, and more per reset |
| **Undo/rollback** | Agent can revert last action (max 5 deep, small penalty) |
| **Custom data upload** | `POST /upload` accepts any CSV/JSON, auto-detects issues |
| **Dense reward signal** | Per-step health delta — not just binary end-of-episode |
| **Full OpenEnv spec** | `reset()` · `step()` · `state()` · `openenv.yaml` · typed Pydantic models |

---

## Quick Start

```bash
# Docker (recommended)
docker build -t dataops-gym .
docker run -p 7860:7860 dataops-gym

# Local
pip install fastapi uvicorn pandas numpy faker openai python-multipart python-dotenv gradio matplotlib
uvicorn dataops_gym.server.app:app --port 7860
```

**Try it immediately:**
```bash
# 1. Start an episode
curl -X POST localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}'

# 2. Take an action
curl -X POST localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "drop_duplicates"}'

# 3. Check your score
curl -X POST localhost:7860/grader
```

**Open the Gradio dashboard:** Navigate to `http://localhost:7860` in your browser.

---

## Tasks (7)

### Easy — Product Sales Cleaning
**Difficulty:** Easy | **Max steps:** 30

A 50-row product sales table with injected mess: `"$1,299.99"` strings as prices, mixed date formats, inconsistent category casing, ~8% nulls, ~10% duplicates, whitespace issues.

**Agent goal:** Clean all issues, submit. Graded on null cleanliness, type correctness, deduplication, and format compliance.

### Medium — Multi-Table User/Purchase Merge
**Difficulty:** Medium | **Max steps:** 30

Two related tables with mismatched user IDs (`"1"`, `"001"`, `"USR-001"`), `"$49.99"` strings, mixed dates, status casing issues, and ~8% duplicate user rows.

**Agent goal:** Standardise IDs, clean both tables, left-join, filter active users.

### Hard — PII Redaction
**Difficulty:** Hard | **Max steps:** 30

30 web-scraped text documents with embedded PII (emails, phone numbers, credit cards, SSNs).

**Agent goal:** Replace all PII with `[REDACTED]`. Graded on recall, precision, and text preservation.

### Outlier Detection — Employee Dataset
**Difficulty:** Medium-Hard | **Max steps:** 30

100-row employee dataset with planted outliers AND legitimate extreme values (executive salaries). The agent must distinguish real anomalies from valid data.

**Agent goal:** Clip impossible values while preserving legitimate extremes. Graded on outlier removal (0.35), legitimate preservation (0.35), row retention (0.15), data integrity (0.15).

### Schema Migration — Dataset Restructuring
**Difficulty:** Hard | **Max steps:** 30

60-row dataset needing structural changes: split combined columns (full_name, address), standardize phone numbers, map status codes to strings.

**Agent goal:** Restructure to target schema. Graded on schema match (0.40), value correctness (0.30), row retention (0.15), old columns removed (0.15).

### Drift Detection — Streaming Data
**Difficulty:** Hard | **Max steps:** 60

200 historical rows + 15 streaming batches. Drift starts at batch 8 with configurable severity. The agent must analyze each batch against historical data and label it.

**Agent goal:** Label each batch as "normal" or "drift". Graded on F1 score (0.7) + coverage (0.3).

### Poisoning Detection — Sentiment Dataset
**Difficulty:** Very Hard | **Max steps:** 30

100-row sentiment classification dataset with ~10% poisoned rows: label flips, subtle mislabels, trigger phrase injections ("EVAL_OVERRIDE").

**Agent goal:** Flag poisoned rows without flagging clean ones. Graded on F1 score (0.7) + clean preservation (0.3).

### Custom — Bring Your Own Data
Upload any `.csv` or `.json` file. The environment auto-detects issues and creates a cleaning task.

```bash
curl -X POST localhost:7860/upload -F "file=@my_data.csv"
```

---

## Action Space (27 actions)

```json
{
  "action_type": "cast_type",
  "column_name": "price",
  "target_type": "float"
}
```

### Cleaning Actions

| Action | Parameters | Description |
|---|---|---|
| `drop_nulls` | `column_name` | Drop rows with null in column |
| `impute_missing` | `column_name`, `strategy` | Fill: `mean` `median` `mode` `ffill` `bfill` |
| `drop_duplicates` | — | Remove exact duplicate rows |
| `drop_column` | `column_name` | Remove a column |
| `rename_column` | `column_name`, `new_name` | Rename column |
| `cast_type` | `column_name`, `target_type` | Cast: `int` `float` `str` `datetime` `bool` |
| `apply_regex` | `column_name`, `pattern`, `replacement` | Regex find-and-replace |
| `format_date` | `column_name`, `target_format` | Standardise dates (e.g. `%Y-%m-%d`) |
| `strip_whitespace` | `column_name` | Trim leading/trailing whitespace |
| `merge_tables` | `right_table`, `merge_on`, `merge_how` | Join two tables |
| `filter_rows` | `filter_condition` | Pandas `.query()` string |
| `fill_value` | `column_name`, `fill_value` | Fill nulls with a literal value |
| `submit` | — | End episode, trigger grading |
| `undo` | — | Revert last action (max 5 deep) |

### Outlier / Schema Actions

| Action | Parameters | Description |
|---|---|---|
| `clip_outliers` | `column_name`, `clip_min`, `clip_max` | Clip values to range |
| `detect_outliers` | `column_name`, `outlier_method` | Analyse outliers (read-only) |
| `split_column` | `column_name`, `delimiter`, `new_columns`, `max_splits` | Split column by delimiter |
| `map_values` | `column_name`, `value_mapping` | Map values via dict |

### Drift / Poisoning Actions

| Action | Parameters | Description |
|---|---|---|
| `advance_stream` | — | Load next batch from stream |
| `analyze_distribution` | `column_name` | Compare batch vs historical stats |
| `label_batch` | `drift_label` | Label current batch as "normal" or "drift" |
| `flag_rows` | `row_indices` | Flag suspicious rows by index |

### Adversarial Corruption Actions

| Action | Parameters | Description |
|---|---|---|
| `inject_nulls` | `column_name`, `inject_count` | Inject null values |
| `swap_values` | `column_name`, `inject_count` | Swap random pairs |
| `introduce_typos` | `column_name`, `typo_rate` | Add character-level typos |
| `flip_labels` | `column_name`, `inject_count` | Flip values to random alternatives |
| `inject_pii` | `column_name`, `inject_count` | Append PII strings to cells |

---

## Curriculum Learning (10 Levels)

Automatic difficulty progression based on agent performance:

| Level | Task | Key Parameters |
|---|---|---|
| 1 | easy | 30 rows, 5% nulls, 5% duplicates |
| 2 | easy | 50 rows, 10% nulls, 10% duplicates |
| 3 | easy | 100 rows, 20% nulls, 15% duplicates |
| 4 | medium | 40 rows, 5% nulls |
| 5 | medium | 80 rows, 15% nulls |
| 6 | outlier_detection | 80 rows, 5% outlier rate |
| 7 | hard | 30 docs, 20% PII density |
| 8 | schema_migration | 60 rows, 50% complexity |
| 9 | drift_detection | 30% drift severity |
| 10 | poisoning_detection | 150 rows, 15% poison rate |

**Progression rules:** Score > 0.85 advances, score < 0.40 demotes, otherwise stays.

```bash
# Start curriculum
curl -X POST localhost:7860/curriculum -H "Content-Type: application/json" \
  -d '{"action": "start"}'

# After cleaning, advance to next level
curl -X POST localhost:7860/curriculum -H "Content-Type: application/json" \
  -d '{"action": "next"}'
```

---

## Adversarial Mode

Two-player game where a **corruptor** injects data quality issues and a **cleaner** tries to fix them:

1. **Corrupt phase** (5 rounds): Corruptor uses `inject_nulls`, `swap_values`, `introduce_typos`, `flip_labels`, `inject_pii`
2. **Clean phase** (5 rounds): Cleaner uses standard cleaning actions
3. **Scoring:** Cleaner score = similarity to original clean data; Corruptor score = 1 - cleaner score

```bash
# Start adversarial game
curl -X POST localhost:7860/adversarial/start -H "Content-Type: application/json" \
  -d '{"num_rows": 50, "seed": 42}'

# Corrupt
curl -X POST localhost:7860/adversarial/step -H "Content-Type: application/json" \
  -d '{"role": "corruptor", "action": {"action_type": "inject_nulls", "column_name": "price", "inject_count": 5}}'

# Clean
curl -X POST localhost:7860/adversarial/step -H "Content-Type: application/json" \
  -d '{"role": "cleaner", "action": {"action_type": "impute_missing", "column_name": "price", "strategy": "mean"}}'
```

---

## Multi-Agent Mode

Collaborative cleaning where multiple agents are assigned column subsets:

- Each agent gets a **responsibility** (null handling, type fixing, etc.) and **assigned columns**
- **Conflict detection:** Actions on columns outside an agent's assignment are logged
- **Coordination score:** `1.0 - (conflicts / total_steps)` — measures how well agents stay in their lanes

```bash
# Start 3-agent session
curl -X POST localhost:7860/multi_agent/start -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "num_agents": 3, "seed": 42}'

# Agent 1 acts on its assigned columns
curl -X POST localhost:7860/multi_agent/step -H "Content-Type: application/json" \
  -d '{"agent_id": "agent_1", "action": {"action_type": "drop_nulls", "column_name": "price"}}'
```

---

## Gradio Dashboard

An interactive 5-tab dashboard is served at the root URL (`/`):

| Tab | Features |
|---|---|
| **Interactive Playground** | Task selector, difficulty sliders, reset, data preview, action executor, health score, grading |
| **Reward Visualization** | Cumulative reward line chart, action reward bar chart, column health heatmap |
| **Curriculum** | Start/next buttons, level display, performance history chart |
| **Adversarial** | Corruptor/cleaner panels, phase indicator, scores |
| **Multi-Agent** | Agent assignments, per-agent actions, conflict log, coordination score |

---

## Observation Space

```json
{
  "task_id": "easy",
  "step_number": 3,
  "total_rows": 50,
  "total_columns": 5,
  "data_health_score": 0.847,
  "column_summaries": [
    {
      "name": "price",
      "dtype": "object",
      "null_count": 0,
      "null_percentage": 0.0,
      "unique_count": 48,
      "sample_values": ["$19.99", "$1,299.00", "$5.50"]
    }
  ],
  "preview_rows": [...],
  "reward": 0.042,
  "done": false,
  "undo_available": true,
  "undo_depth": 3
}
```

---

## Reward Function

| Signal | Value | Rationale |
|---|---|---|
| Health score improvement | `delta_health * 2.0` | Reward every step of progress |
| Step penalty | `-0.01` | Encourage efficiency |
| Invalid action | `-0.1` | Penalise misuse of the API |
| Dropped needed column | `-0.5` | Penalise destructive actions |
| Excessive row loss (>20%) | `-0.3` | Penalise over-filtering |
| Submit with health > 0.9 | `+0.5` | Reward high-quality completion |
| PII type redacted (hard) | `+0.2` per type | Reward each PII category caught |
| Undo | `-0.02` | Available but costly |
| Undo on empty history | `-0.05` | Teach boundary awareness |

All rewards clamped to `[-1.0, 1.0]`.

---

## Baseline Scores

`gpt-4o-mini` agent with `seed=42` — fully reproducible:

| Task | Difficulty | Raw Score | After Agent | Model |
|---|---|---|---|---|
| easy | Easy | 0.742 | **0.9623** | gpt-4o-mini |
| medium | Medium | 0.771 | **0.9000** | gpt-4o-mini |
| hard | Hard | 0.597 | **1.0000** | gpt-4o-mini |
| outlier_detection | Medium-Hard | — | *pending* | gpt-4o-mini |
| schema_migration | Hard | — | *pending* | gpt-4o-mini |
| drift_detection | Hard | — | *pending* | gpt-4o-mini |
| poisoning_detection | Very Hard | — | *pending* | gpt-4o-mini |

To reproduce:
```bash
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=https://api.openai.com/v1
export BASELINE_MODEL=gpt-4o-mini
python -m dataops_gym.baseline.inference
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Gradio dashboard (interactive UI) |
| `/health` | GET | `{"status": "healthy"}` |
| `/tasks` | GET | All 7 tasks + action schema + configurable params |
| `/reset` | POST | Start episode: `{"task_id": "easy", "seed": 42, ...}` |
| `/step` | POST | Take action: `{"action_type": "cast_type", ...}` |
| `/state` | GET | Current episode state (step count, reward, done) |
| `/grader` | POST | Score current state; returns 0.0-1.0 |
| `/upload` | POST | Upload CSV/JSON, auto-detect issues |
| `/curriculum` | POST | Curriculum learning: `{"action": "start\|next\|status\|reset"}` |
| `/adversarial/start` | POST | Start adversarial game: `{"num_rows": 50, "seed": 42}` |
| `/adversarial/step` | POST | Corruptor/cleaner action: `{"role": "...", "action": {...}}` |
| `/multi_agent/start` | POST | Start multi-agent: `{"task_id": "easy", "num_agents": 3}` |
| `/multi_agent/step` | POST | Agent action: `{"agent_id": "agent_1", "action": {...}}` |
| `/multi_agent/status` | GET | Current multi-agent session state |
| `/baseline` | POST | Run LLM agent on all tasks (needs `OPENAI_API_KEY`) |
| `/docs` | GET | Interactive Swagger UI |

---

## Project Structure

```
dataops_gym/
├── models.py                   # Pydantic models: Action, Observation, State, Curriculum, Adversarial, MultiAgent
├── server/
│   ├── app.py                  # FastAPI app — all endpoints + Gradio mount
│   ├── gradio_app.py           # Gradio dashboard (5 tabs)
│   ├── dataops_environment.py  # Core RL environment logic
│   └── requirements.txt
├── tasks/
│   ├── generators.py           # Procedural dataset generators (7 tasks)
│   ├── auto_detect.py          # Issue detection for custom uploads
│   ├── generate_datasets.py    # Static fallback dataset generator
│   └── datasets/               # Static fallback CSVs/JSONs
├── graders/
│   └── grader.py               # grade() + grade_by_criteria() for all 7 tasks
├── baseline/
│   └── inference.py            # LLM baseline agent (all 7 tasks)
├── tests/
│   └── test_environment.py     # 45 pytest tests
└── openenv.yaml                # OpenEnv spec compliance file
Dockerfile                      # HF Spaces compatible (user 1000)
```

---

## OpenEnv Compliance

```yaml
# openenv.yaml
name: dataops-gym
version: "2.0.0"
tasks: [easy, medium, hard, outlier_detection, schema_migration, drift_detection, poisoning_detection]
observation_type: structured_json
action_type: structured_json
reward_range: [-1.0, 1.0]
endpoints:
  reset: POST /reset
  step:  POST /step
  state: GET  /state
```

All endpoints return typed Pydantic models. `openenv validate` passes.

---

## Why DataOps Gym?

- **Real-world domain.** Data cleaning is a skill every ML engineer needs. Performance here directly translates to practical value.
- **Rich action space.** 27 typed operations covering the full data engineering workflow.
- **Dense rewards.** Per-step health score delta means agents get signal at every action.
- **Scalable difficulty.** 10-level curriculum + configurable parameters for beginner to expert-level episodes.
- **Adversarial training.** Corruptor/cleaner game creates increasingly challenging scenarios.
- **Multi-agent coordination.** Test collaborative data cleaning with conflict detection.
- **No golden dataset required.** Criteria-based grading enables infinite unique episodes.
- **Interactive dashboard.** Gradio UI for manual exploration, visualization, and debugging.
- **Custom data.** The `/upload` endpoint makes it useful for real private datasets.
