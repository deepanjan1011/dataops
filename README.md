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

<img src="https://img.shields.io/badge/OpenEnv-compatible-brightgreen?style=for-the-badge" alt="OpenEnv Compatible"/>
<img src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/>
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
<img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
<img src="https://img.shields.io/badge/license-MIT-yellow?style=for-the-badge" alt="MIT License"/>

<br/><br/>

# DataOps Gym

### An RL Environment for Training AI Agents on Real-World Data Engineering

**Data engineers spend 60-80% of their time cleaning data, not building models.**<br/>
DataOps Gym is a benchmark environment where AI agents learn to handle messy, real-world datasets — nulls, type mismatches, duplicates, PII, schema inconsistencies, data drift, and poisoned labels — through a structured action space with dense reward signals.

<br/>

[**Live Demo**](https://huggingface.co/spaces/deepanjan1011/dataops-gym) &#8226; [**API Docs**](https://deepanjan1011-dataops-gym.hf.space/docs) &#8226; [**Try It Now**](#quick-start)

</div>

<br/>

---

## How It Works

```
  ┌────────────┐         ┌─────────────────────────────────────────┐
  │            │  reset  │           DataOps Gym Server             │
  │   Agent    │────────►│                                         │
  │            │         │  ┌─────────────┐  ┌──────────────────┐  │
  │  (LLM /   │  step   │  │   Dataset    │  │     Grader       │  │
  │   RL /    │────────►│  │  Generator   │  │  (per-task       │  │
  │  Human)   │         │  │             │  │   criteria)      │  │
  │            │   obs   │  │  27 actions  │  │                  │  │
  │            │◄────────│  │  on pandas   ├─►│  score: 0.0-1.0  │  │
  │            │         │  │  DataFrames  │◄─┤  reward per step │  │
  └────────────┘         │  └─────────────┘  └──────────────────┘  │
                         │                                         │
                         │   7 tasks · 27 actions · 5 modes        │
                         └─────────────────────────────────────────┘
```

The agent receives a **messy dataset** as an observation (column summaries, preview rows, health score). It selects an **action** (e.g., `cast_type`, `drop_duplicates`, `apply_regex`). The environment executes it on the underlying pandas DataFrame and returns the **new observation + reward**. At the end, a deterministic **grader** scores the final state from 0.0 to 1.0.

---

## The 7 Tasks

Each task targets a different data engineering challenge. All tasks support procedural generation — every episode produces a unique dataset. Use `seed=42` for reproducibility.

### Easy — Product Sales Cleaning
50-row sales table with `"$1,299.99"` price strings, mixed date formats (`2024-01-15`, `Jan 15, 2024`), inconsistent category casing, ~8% nulls, ~10% duplicates. Agent must clean all issues.
> Grading: null cleanliness (25%) · type correctness (25%) · deduplication (20%) · format compliance (15%) · row retention (15%)

### Medium — Multi-Table Merge
Two related tables (users + purchases) with mismatched user IDs (`"1"`, `"001"`, `"USR-001"`), amount strings, status casing. Agent must standardize IDs, clean both tables, merge, and filter to active users.
> Grading: type correctness (25%) · deduplication (20%) · status compliance (20%) · null cleanliness (20%) · tables merged (15%)

### Hard — PII Redaction
30 text documents with embedded emails, phone numbers, credit card numbers, and SSNs. Agent must replace all PII with `[REDACTED]` while preserving surrounding text.
> Grading: PII recall (40%) · PII precision (30%) · text preservation (30%)

### Outlier Detection — Employee Dataset
100-row employee dataset with planted outliers AND legitimate extreme values (executive salaries $500K-$2M). The agent must distinguish genuine errors from valid data points.
> Grading: outlier removal (35%) · legitimate preservation (35%) · row retention (15%) · data integrity (15%)

### Schema Migration — Dataset Restructuring
60-row dataset needing structural changes: split `full_name` into `first_name + last_name`, split `full_address`, standardize phone numbers to digits only, separate price from currency, map status codes.
> Grading: schema match (40%) · value correctness (30%) · row retention (15%) · old columns removed (15%)

### Drift Detection — Streaming Data
200 historical rows + 15 streaming batches. Drift starts at a configurable batch with adjustable severity. Agent must analyze each batch against the baseline and label it as `"normal"` or `"drift"`.
> Grading: F1 score (70%) · batch coverage (30%)

### Poisoning Detection — Sentiment Dataset
100-row sentiment classification dataset with ~10% poisoned rows: label flips, subtle mislabels, and trigger phrase injections. Agent must flag poisoned rows without over-flagging clean data.
> Grading: F1 score (70%) · clean data preservation (30%)

### Custom — Bring Your Own Data
Upload any `.csv` or `.json` via the dashboard or the `/upload` API endpoint. The environment auto-detects issues (nulls, duplicates, type mismatches, whitespace) and creates a cleaning task with appropriate grading criteria.

---

## Action Space

27 typed actions covering the full data engineering workflow:

**Core Cleaning** — `drop_nulls` · `impute_missing` · `drop_duplicates` · `drop_column` · `rename_column` · `cast_type` · `apply_regex` · `format_date` · `strip_whitespace` · `merge_tables` · `filter_rows` · `fill_value` · `submit` · `undo`

**Outlier & Schema** — `clip_outliers` · `detect_outliers` · `split_column` · `map_values`

**Drift & Poisoning** — `advance_stream` · `analyze_distribution` · `label_batch` · `flag_rows`

**Adversarial Corruption** — `inject_nulls` · `swap_values` · `introduce_typos` · `flip_labels` · `inject_pii`

Every action is a JSON object:
```json
{
  "action_type": "cast_type",
  "column_name": "price",
  "target_type": "float"
}
```

---

## Reward Function

The environment provides **dense per-step rewards** so agents get learning signal at every action, not just at the end:

| Signal | Value | Description |
|---|---|---|
| Health improvement | `delta * 2.0` | Proportional to data quality gain |
| Step cost | `-0.01` | Encourages efficient solutions |
| Invalid action | `-0.1` | Penalizes API misuse |
| Dropped needed column | `-0.5` | Penalizes destructive actions |
| Excessive row loss | `-0.3` | Triggers when >20% rows removed |
| High-quality submit | `+0.5` | Bonus for submitting with health > 0.9 |
| Undo | `-0.02` | Available but costly (max 5 deep) |

---

## Interactive Modes

### Curriculum Learning
10 difficulty levels that automatically progress based on agent performance. Start at level 1 (easy, 30 rows, 5% nulls) and work up to level 10 (poisoning detection, 150 rows, 15% poison rate). Score > 0.85 advances; score < 0.40 demotes.

### Adversarial Mode
Two-player game: a **corruptor** injects data quality issues (nulls, typos, value swaps, label flips, PII) over 5 rounds, then a **cleaner** tries to restore the data over 5 rounds. Final scoring compares the cleaned result to the original.

### Multi-Agent Mode
Collaborative cleaning where 2-5 agents are assigned column subsets and responsibilities (null handling, type fixing, deduplication). Agents working on columns outside their assignment trigger **conflict detection**, and the **coordination score** tracks how well they collaborate.

### Custom Dataset Mode
All modes (Playground, Curriculum, Adversarial, Multi-Agent) support custom uploaded datasets — not just built-in demo tasks. Upload a CSV in the Playground tab and use it across any mode.

---

## Gradio Dashboard

An interactive 5-tab dashboard served at the root URL (`/`):

| Tab | What You Can Do |
|---|---|
| **Interactive Playground** | Select tasks, tune difficulty, upload custom data, execute actions step by step, view data preview, grade your work, download cleaned CSV |
| **Reward Visualization** | Cumulative reward chart, per-action reward breakdown, column health heatmap |
| **Curriculum** | Run demo curriculum or use your own dataset, track level progression and score history |
| **Adversarial** | Play the corruptor/cleaner game on demo or custom data |
| **Multi-Agent** | Assign agents, execute per-agent actions, monitor conflicts and coordination |

---

## Quick Start

**Docker (recommended):**
```bash
docker build -t dataops-gym .
docker run -p 7860:7860 dataops-gym
```

**Local:**
```bash
pip install -r dataops_gym/server/requirements.txt
uvicorn dataops_gym.server.app:app --port 7860
```

**Use it:**
```bash
# Start an episode
curl -X POST localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}'

# Take an action
curl -X POST localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "drop_duplicates"}'

# Grade the result
curl -X POST localhost:7860/grader
```

Or open `http://localhost:7860` for the interactive Gradio dashboard.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/metadata` | GET | OpenEnv metadata (name, version, author, tags) |
| `/schema` | GET | Action/Observation/State JSON schemas |
| `/tasks` | GET | All 7 tasks with descriptions, params, and action schema |
| `/reset` | POST | Start a new episode |
| `/step` | POST | Execute an action |
| `/state` | GET | Current episode state |
| `/grader` | POST | Score the current state (0.0-1.0) |
| `/upload` | POST | Upload custom CSV/JSON dataset |
| `/mcp` | POST | Model Context Protocol JSON-RPC endpoint |
| `/curriculum` | POST | Curriculum learning (start/next/status/reset) |
| `/adversarial/start` | POST | Start adversarial game |
| `/adversarial/step` | POST | Corruptor or cleaner action |
| `/multi_agent/start` | POST | Start multi-agent session |
| `/multi_agent/step` | POST | Per-agent action |
| `/multi_agent/status` | GET | Multi-agent session state |
| `/docs` | GET | Interactive Swagger docs |
| `/` | GET | Gradio dashboard |

---

## Baseline Agent Scores

Hybrid inference agent using `gpt-4o-mini` with scripted strategies for deterministic tasks and LLM reasoning for analysis tasks:

| Task | Score | Strategy |
|---|---|---|
| easy | 0.96 | Scripted: strip, cast, dedup, impute, format, submit |
| medium | 1.00 | Scripted: standardize IDs, clean types, filter, merge |
| hard | 1.00 | Scripted: regex PII redaction patterns |
| outlier_detection | 0.82 | Scripted: detect + clip by department context |
| schema_migration | 0.79 | Scripted: split, map, cast, reorder |
| drift_detection | 0.65 | LLM-driven: analyze distributions, label batches |
| poisoning_detection | 0.39 | LLM-driven: examine text, flag suspicious rows |
| **Average** | **0.80** | |

---

## Project Structure

```
dataops_gym/
├── models.py                   # Pydantic models for actions, observations, state
├── server/
│   ├── app.py                  # FastAPI server — all endpoints + Gradio mount
│   ├── gradio_app.py           # 5-tab interactive dashboard
│   ├── dataops_environment.py  # Core RL environment logic
│   └── requirements.txt        # Python dependencies
├── tasks/
│   ├── generators.py           # Procedural dataset generators for all 7 tasks
│   ├── auto_detect.py          # Auto issue detection for custom uploads
│   └── datasets/               # Pre-generated fallback datasets
├── graders/
│   └── grader.py               # Deterministic grading for all 7 tasks + custom
├── baseline/
│   └── inference.py            # LLM baseline agent
└── tests/
    └── test_environment.py     # Pytest suite

inference.py                    # Competition submission script
openenv.yaml                    # OpenEnv specification file
pyproject.toml                  # Project metadata and entry points
Dockerfile                      # HuggingFace Spaces compatible build
```

---

## OpenEnv Compliance

DataOps Gym implements the full OpenEnv specification:

- **Endpoints:** `/reset`, `/step`, `/state`, `/grader`, `/metadata`, `/schema`, `/mcp`
- **Typed models:** All requests and responses use Pydantic schemas
- **Spec file:** `openenv.yaml` defines tasks, action space, observation space, and reward range
- **Validation:** Passes `openenv validate`

---

## License

MIT
