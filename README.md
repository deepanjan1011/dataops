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

# DataOps Gym — AI Data Quality & Curation Environment

![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-compatible-brightgreen)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Docker](https://img.shields.io/badge/docker-ready-blue)

## Motivation

Data curation is the **#1 bottleneck in machine learning**. Engineers at Meta, HuggingFace, and
across the industry spend up to 80% of their time cleaning and preparing data before a single
model can be trained. This environment trains AI agents to automate that work.

DataOps Gym is an OpenEnv-compatible reinforcement learning environment where an AI agent
acts as a data engineer — receiving messy, real-world-style datasets and learning to clean
them using programmatic actions, guided by a reward signal based on data quality criteria.

---

## Environment Description

The agent observes **dataset statistics** (not raw data): column types, null counts, sample
values, and an overall data health score. At each step it chooses one of **14 pandas-like
operations** to apply. The episode ends when the agent submits or exhausts 30 steps.

Every episode generates **fresh procedural data** with the same problem types but different
values. Pass `seed=42` for reproducible episodes.

The environment runs as a **FastAPI server** (port 7860) that exposes a standard OpenEnv HTTP
interface. Any agent — LLM-based, RL-trained, or rule-based — can interact with it over HTTP.

---

## Action Space (14 actions)

| Action Type | Required Parameters | Description |
|---|---|---|
| `drop_nulls` | `column_name` | Drop rows where column is null |
| `impute_missing` | `column_name`, `strategy` | Fill nulls: `mean`, `median`, `mode`, `ffill`, `bfill` |
| `drop_duplicates` | — | Remove exact duplicate rows |
| `drop_column` | `column_name` | Remove a column (penalty if needed for task) |
| `rename_column` | `column_name`, `new_name` | Rename a column |
| `cast_type` | `column_name`, `target_type` | Cast to `int`, `float`, `str`, `datetime`, `bool` |
| `apply_regex` | `column_name`, `pattern`, `replacement` | Regex find-and-replace on a column |
| `format_date` | `column_name`, `target_format` | Standardize dates (e.g. `%Y-%m-%d`) |
| `strip_whitespace` | `column_name` | Strip leading/trailing whitespace |
| `merge_tables` | `right_table`, `merge_on`, `merge_how` | Merge two tables |
| `filter_rows` | `filter_condition` | Pandas query string filter |
| `fill_value` | `column_name`, `fill_value` | Fill nulls with a literal value |
| `submit` | — | End episode and trigger grading |
| `undo` | — | Revert last action (up to 5 deep, costs −0.02 reward) |

---

## Observation Space

Each step returns a `DataOpsObservation` JSON object:

| Field | Type | Description |
|---|---|---|
| `task_id` | str | Current task: `easy`, `medium`, `hard`, or `custom` |
| `task_description` | str | Natural language task description |
| `step_number` | int | Steps taken so far |
| `total_rows` | int | Row count of current dataset |
| `total_columns` | int | Column count |
| `column_summaries` | list | Per-column: dtype, null count, unique count, sample values, mean/min/max |
| `preview_rows` | list | First 5 rows as dicts |
| `data_health_score` | float | Overall cleanliness score 0.0–1.0 |
| `available_tables` | list | Table names accessible (relevant for merge tasks) |
| `last_action_result` | str | Success or error message from last action |
| `reward` | float | Reward received for the last action |
| `done` | bool | Whether the episode has ended |
| `error` | str \| null | Error detail if last action failed |
| `undo_available` | bool | Whether undo history exists |
| `undo_depth` | int | How many undos are available (max 5) |

---

## Tasks

### Easy — Basic Cleaning
**Difficulty:** Easy | **Max Steps:** 30 | **Expected Score:** 0.70–0.95

A product sales table with mixed price formats (`$1,299.99`), mixed date formats, inconsistent
category casing, whitespace issues, nulls, and duplicate rows.

### Medium — Multi-Table Merge
**Difficulty:** Medium | **Max Steps:** 30 | **Expected Score:** 0.40–0.90

Two tables (users + purchases) requiring ID standardisation (`USR-001` → `1`), date fixing,
deduplication, active-user filtering, and a left-join merge on `user_id`.

### Hard — PII Redaction
**Difficulty:** Hard | **Max Steps:** 30 | **Expected Score:** 0.20–1.0

Text documents containing emails, phone numbers, credit card numbers, and SSNs. The agent must
replace all PII with `[REDACTED]` without over-redacting clean text.

### Custom — Bring Your Own Data
Upload any CSV or JSON file via `POST /upload`. The environment auto-detects issues (nulls,
duplicates, type mismatches, whitespace, casing, PII, date formats) and creates a cleaning task.

---

## Procedural Generation

Every `reset()` call generates fresh data. Use `seed` for reproducibility:

```bash
# Random fresh data (default)
curl -X POST localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'

# Deterministic (same data every time)
curl -X POST localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}'
```

---

## Configurable Difficulty

Control how messy the generated data is via difficulty parameters on `/reset`:

### Easy task parameters

| Parameter | Default | Min | Max | Description |
|---|---|---|---|---|
| `num_rows` | 50 | 10 | 1000 | Base row count (before duplicates) |
| `null_percentage` | 0.08 | 0.0 | 0.5 | Fraction of cells nulled per column |
| `duplicate_rate` | 0.10 | 0.0 | 0.3 | Fraction of duplicate rows injected |
| `format_inconsistency` | 0.5 | 0.0 | 1.0 | Date format variety (0 = all YYYY-MM-DD) |
| `seed` | null | — | — | Set for reproducibility |

### Medium task parameters

| Parameter | Default | Min | Max | Description |
|---|---|---|---|---|
| `num_users` | 40 | 10 | 500 | Number of user rows |
| `num_purchases` | 60 | 10 | 1000 | Number of purchase rows |
| `null_percentage` | 0.05 | 0.0 | 0.5 | Fraction of cells nulled per column |
| `duplicate_rate` | 0.08 | 0.0 | 0.3 | Fraction of duplicate user rows injected |
| `seed` | null | — | — | Set for reproducibility |

### Hard task parameters

| Parameter | Default | Min | Max | Description |
|---|---|---|---|---|
| `num_docs` | 30 | 5 | 500 | Number of text documents |
| `pii_density` | 0.3 | 0.0 | 1.0 | Fraction of docs containing PII |
| `pii_variety` | 0.5 | 0.0 | 1.0 | How many PII types appear per doc |
| `seed` | null | — | — | Set for reproducibility |

**Example — harder version:**
```bash
curl -X POST localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42, "null_percentage": 0.35, "duplicate_rate": 0.25}'
```

---

## Undo / Rollback

Agents can revert their last action with the `undo` action type:

```json
{"action_type": "undo"}
```

- Up to **5 undos** available per episode (oldest snapshot dropped when full)
- Costs **−0.02 reward** per undo (discourages spam; less than −0.5 for destructive actions)
- Undo on empty history costs **−0.05** and returns an error
- Observation includes `undo_available` (bool) and `undo_depth` (int) so agents know their options

---

## Custom Dataset Upload

Upload any `.csv` or `.json` file (max 10 MB) and the environment auto-detects its issues:

```bash
curl -X POST localhost:7860/upload \
  -F "file=@your_data.csv"
```

**Response:**
```json
{
  "status": "loaded",
  "rows": 500,
  "columns": 8,
  "detected_issues": {
    "missing_values": {"email": {"count": 12, "percentage": 0.024}},
    "duplicates": {"count": 5},
    "type_mismatches": {"price": "Looks numeric but stored as string"},
    "whitespace": {"name": 8},
    "inconsistent_casing": {"category": {"unique_raw": 6, "unique_lowered": 3}},
    "date_format_issues": {"created_at": "Likely date column stored as string"}
  },
  "task_description": "Clean this custom dataset. Detected issues: missing_values, duplicates, ...",
  "observation": {...}
}
```

After upload, call `/step` and `/grader` as normal. `task_id` will be `"custom"`.

---

## Reward Function

| Signal | Value |
|---|---|
| Health score improvement | `Δhealth × 2.0` |
| Step penalty (efficiency) | `−0.01` per step |
| Invalid action | `−0.1` |
| Dropped a needed column | `−0.5` |
| Excessive row loss (>20%) | `−0.3` |
| Submit with health > 0.9 | `+0.5` bonus |
| PII type redacted (hard) | `+0.2` per type |
| Undo action | `−0.02` |
| Undo on empty history | `−0.05` |

All rewards clamped to `[−1.0, 1.0]`.

---

## Setup

```bash
# Local
pip install fastapi uvicorn pydantic pandas numpy faker requests openai python-dotenv python-multipart
uvicorn dataops_gym.server.app:app --host 0.0.0.0 --port 7860

# Docker
docker build -t dataops-gym .
docker run -p 7860:7860 dataops-gym
```

---

## Baseline Scores

Scores produced by `gpt-4o-mini` with `seed=42` (fully reproducible):

| Task | Score | Model |
|---|---|---|
| easy | 0.9623 | gpt-4o-mini |
| medium | 0.9000 | gpt-4o-mini |
| hard | 1.0000 | gpt-4o-mini |

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
| `/` | GET | Welcome + endpoint list |
| `/health` | GET | Health check |
| `/tasks` | GET | List tasks + action schema + configurable params |
| `/reset` | POST | Start episode: `{"task_id": "easy", "seed": 42, "null_percentage": 0.1}` |
| `/step` | POST | Take action: `DataOpsAction` JSON |
| `/state` | GET | Episode state |
| `/grader` | POST | Grade current state (criteria-based or golden) |
| `/baseline` | POST | Run LLM baseline (needs `OPENAI_API_KEY`) |
| `/upload` | POST | Upload custom CSV/JSON dataset (multipart) |
