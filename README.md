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

### An OpenEnv-Compatible RL Environment for AI Data Engineering Agents

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-brightgreen?style=flat-square)](https://openenv.dev)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://python.org)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://hub.docker.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow?style=flat-square)](LICENSE)

**[🚀 Live Demo](https://huggingface.co/spaces/deepanjan1011/dataops-gym)** · **[📖 API Docs](https://deepanjan1011-dataops-gym.hf.space/docs)** · **[🧪 Try It Now](#quick-start)**

</div>

---

## The Problem

**Data engineers spend 60–80% of their time cleaning data** — not building models. Yet there is no standardised benchmark environment where AI agents can learn, practice, and be evaluated on real data engineering workflows.

DataOps Gym fills that gap.

---

## What Is DataOps Gym?

DataOps Gym is a fully OpenEnv-compliant reinforcement learning environment that puts an AI agent in the role of a data engineer. The agent receives **messy, real-world-style datasets** and must clean them using programmatic actions — strip whitespace, fix types, redact PII, merge tables — guided by a dense reward signal that measures progress toward a clean dataset.

```
┌─────────────────────────────────────────────────────────────────┐
│                        DataOps Gym                              │
│                                                                 │
│   ┌──────────┐    observation     ┌────────────────────────┐   │
│   │          │ ◄─────────────── │                        │   │
│   │  Agent   │                   │  FastAPI Environment   │   │
│   │  (LLM /  │ ──── action ───► │  (pandas DataFrames)   │   │
│   │   RL)    │                   │                        │   │
│   │          │ ◄── reward+done ─ │  Grader + Reward Fn    │   │
│   └──────────┘                   └────────────────────────┘   │
│                                                                 │
│   Tasks: easy (cleaning) · medium (merge) · hard (PII)         │
│   + custom (upload your own CSV/JSON)                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Features

| Feature | Description |
|---|---|
| **3 graded tasks** | Easy → Medium → Hard with deterministic 0.0–1.0 scoring |
| **14 action types** | Full pandas-like API: cast, merge, regex, impute, undo, and more |
| **Procedural generation** | Fresh data every episode; `seed=42` for reproducibility |
| **Configurable difficulty** | Tune `null_percentage`, `duplicate_rate`, `pii_density` per reset |
| **Undo/rollback** | Agent can revert last action (−0.02 penalty — teaches caution) |
| **Custom data upload** | `POST /upload` accepts any CSV/JSON, auto-detects issues |
| **Dense reward signal** | Per-step health delta × 2.0 — not just binary end-of-episode |
| **Full OpenEnv spec** | `reset()` · `step()` · `state()` · `openenv.yaml` · typed Pydantic models |

---

## Quick Start

```bash
# Docker (recommended)
docker build -t dataops-gym .
docker run -p 7860:7860 dataops-gym

# Local
pip install fastapi uvicorn pandas numpy faker openai python-multipart python-dotenv
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

---

## Tasks

### Easy — Product Sales Cleaning
**Score range:** 0.70–0.95 | **Max steps:** 30

A 50-row product sales table with injected real-world mess:

| Issue | Details |
|---|---|
| Price format | `"$1,299.99"` stored as strings |
| Date formats | Mixed: `2024-01-15`, `01/15/2024`, `Jan 15, 2024` |
| Category casing | `"FOOD"`, `"food"`, `"Food"` for the same value |
| Nulls | ~8% of cells across columns |
| Duplicates | ~10% duplicate rows |
| Whitespace | Leading/trailing spaces in product names |

**Agent goal:** Clean all issues, submit. Graded on null cleanliness, type correctness, deduplication, and format compliance.

---

### Medium — Multi-Table User/Purchase Merge
**Score range:** 0.40–0.90 | **Max steps:** 30

Two related tables that must be cleaned and joined:

| Issue | Details |
|---|---|
| User ID formats | `"1"`, `"001"`, `"USR-001"` — all the same user |
| Amount format | `"$49.99"` strings in purchases table |
| Date formats | Mixed across both tables |
| Status casing | `"ACTIVE"`, `"Active"`, `"active"` |
| Nulls | Names and statuses partially missing |
| Duplicates | ~8% duplicate user rows |

**Agent goal:** Standardise IDs → clean both tables → left-join → filter active users only.

---

### Hard — PII Redaction
**Score range:** 0.20–1.0 | **Max steps:** 30

30 web-scraped text documents with embedded PII:

| PII Type | Example | Pattern |
|---|---|---|
| Email | `alice@example.com` | RFC 5322 |
| Phone | `(555) 123-4567` | 3 formats |
| Credit card | `4532-1234-5678-9012` | 16-digit groups |
| SSN | `123-45-6789` | US SSN format |

**Agent goal:** Replace all PII with `[REDACTED]`. Graded on recall (caught all PII), precision (didn't over-redact), and text preservation.

---

### Custom — Bring Your Own Data
Upload any `.csv` or `.json` file. The environment auto-detects:
- Missing values, duplicates, type mismatches
- Whitespace, inconsistent casing
- Potential PII (emails, phones)
- Mixed date formats

```bash
curl -X POST localhost:7860/upload -F "file=@my_data.csv"
```

Returns detected issues, a task description, and the initial observation — then use `/step` and `/grader` as normal.

---

## Procedural Generation

Every `reset()` generates fresh dirty data with the same problem structure but different values. No two episodes are identical by default.

```bash
# Reproducible baseline (same data every time)
curl -X POST localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}'

# Random fresh episode
curl -X POST localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'
```

---

## Configurable Difficulty

Tune difficulty per episode without code changes:

### Easy
| Parameter | Default | Range | Effect |
|---|---|---|---|
| `num_rows` | 50 | 10–1000 | Dataset size |
| `null_percentage` | 0.08 | 0.0–0.5 | How many cells are null |
| `duplicate_rate` | 0.10 | 0.0–0.3 | Fraction of duplicate rows |
| `format_inconsistency` | 0.5 | 0.0–1.0 | Date format variety (0 = uniform) |

### Medium
| Parameter | Default | Range | Effect |
|---|---|---|---|
| `num_users` | 40 | 10–500 | Users table size |
| `num_purchases` | 60 | 10–1000 | Purchases table size |
| `null_percentage` | 0.05 | 0.0–0.5 | Missing values rate |
| `duplicate_rate` | 0.08 | 0.0–0.3 | Duplicate user rows |

### Hard
| Parameter | Default | Range | Effect |
|---|---|---|---|
| `num_docs` | 30 | 5–500 | Number of documents |
| `pii_density` | 0.3 | 0.0–1.0 | Fraction of docs with PII |
| `pii_variety` | 0.5 | 0.0–1.0 | PII types per doc |

**Example — stress test:**
```bash
curl -X POST localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "easy",
    "seed": 42,
    "num_rows": 500,
    "null_percentage": 0.35,
    "duplicate_rate": 0.25
  }'
```

---

## Action Space (14 actions)

```json
{
  "action_type": "cast_type",
  "column_name": "price",
  "target_type": "float"
}
```

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
| Health score improvement | `Δhealth × 2.0` | Reward every step of progress |
| Step penalty | `−0.01` | Encourage efficiency |
| Invalid action | `−0.1` | Penalise misuse of the API |
| Dropped needed column | `−0.5` | Penalise destructive actions |
| Excessive row loss (>20%) | `−0.3` | Penalise over-filtering |
| Submit with health > 0.9 | `+0.5` | Reward high-quality completion |
| PII type redacted (hard) | `+0.2` per type | Reward each PII category caught |
| Undo | `−0.02` | Available but costly |
| Undo on empty history | `−0.05` | Teach boundary awareness |

All rewards clamped to `[−1.0, 1.0]`.

---

## Baseline Scores

`gpt-4o-mini` agent with `seed=42` — fully reproducible:

| Task | Raw Score | After Agent | Model |
|---|---|---|---|
| easy | 0.742 | **0.9623** | gpt-4o-mini |
| medium | 0.771 | **0.9000** | gpt-4o-mini |
| hard | 0.597 | **1.0000** | gpt-4o-mini |

To reproduce:
```bash
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=https://api.openai.com/v1
export BASELINE_MODEL=gpt-4o-mini
python -m dataops_gym.baseline.inference
```

---

## API Reference

| Endpoint | Method | Body / Notes |
|---|---|---|
| `/` | GET | Welcome message + endpoint map |
| `/health` | GET | `{"status": "healthy"}` |
| `/tasks` | GET | All tasks + action schema + configurable params |
| `/reset` | POST | `{"task_id": "easy", "seed": 42, "null_percentage": 0.2}` |
| `/step` | POST | `{"action_type": "cast_type", "column_name": "price", "target_type": "float"}` |
| `/state` | GET | Current episode state (step count, reward, done) |
| `/grader` | POST | Score current state; returns 0.0–1.0 |
| `/baseline` | POST | Run LLM agent across all 3 tasks (needs `OPENAI_API_KEY`) |
| `/upload` | POST | Multipart file upload — `.csv` or `.json`, max 10 MB |
| `/docs` | GET | Interactive Swagger UI |

---

## Project Structure

```
dataops_gym/
├── models.py                   # Pydantic models: Action, Observation, State
├── server/
│   ├── app.py                  # FastAPI app — all endpoints
│   ├── dataops_environment.py  # Core RL environment logic
│   └── requirements.txt
├── tasks/
│   ├── generators.py           # Procedural dataset generators (all 3 tasks)
│   ├── auto_detect.py          # Issue detection for custom uploads
│   ├── generate_datasets.py    # Static fallback dataset generator
│   └── datasets/               # Static fallback CSVs/JSONs
├── graders/
│   └── grader.py               # grade() + grade_by_criteria()
├── baseline/
│   └── inference.py            # LLM baseline agent
├── tests/
│   └── test_environment.py     # 20 pytest tests
└── openenv.yaml                # OpenEnv spec compliance file
Dockerfile                      # HF Spaces compatible (user 1000)
```

---

## OpenEnv Compliance

```yaml
# openenv.yaml
name: dataops-gym
version: "1.0.0"
tasks: [easy, medium, hard]
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

- **Real-world domain.** Data cleaning is a skill every ML engineer needs. Unlike game environments, performance here directly translates to practical value.
- **Rich action space.** 14 typed operations covering the full pandas data-cleaning API.
- **Dense rewards.** Per-step health score delta means agents get signal at every action — not just at episode end.
- **Scalable difficulty.** Configurable parameters make it easy to generate beginner to expert-level episodes without code changes.
- **No golden dataset required.** Criteria-based grading works on procedurally generated data, enabling infinite unique episodes.
- **Undo teaches planning.** The rollback action (with a small cost) trains agents to be cautious rather than greedy.
- **Custom data.** The `/upload` endpoint makes DataOps Gym useful for evaluating agents on real private datasets.
