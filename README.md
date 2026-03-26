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
them using programmatic actions, guided by a reward signal based on similarity to a known
"golden" clean version.

---

## Environment Description

The agent observes **dataset statistics** (not raw data): column types, null counts, sample
values, and an overall data health score. At each step it chooses one of 13 pandas-like
operations to apply. The episode ends when the agent submits or exhausts 30 steps.

The environment runs as a **FastAPI server** (port 7860) that exposes a standard OpenEnv HTTP
interface. Any agent — LLM-based, RL-trained, or rule-based — can interact with it over HTTP.

---

## Action Space

| Action Type | Required Parameters | Description |
|---|---|---|
| `drop_nulls` | `column_name` | Drop rows where column is null |
| `impute_missing` | `column_name`, `strategy` | Fill nulls: `mean`, `median`, `mode`, `ffill`, `bfill` |
| `drop_duplicates` | — | Remove exact duplicate rows |
| `drop_column` | `column_name` | Remove a column (penalty if in golden) |
| `rename_column` | `column_name`, `new_name` | Rename a column |
| `cast_type` | `column_name`, `target_type` | Cast to `int`, `float`, `str`, `datetime`, `bool` |
| `apply_regex` | `column_name`, `pattern`, `replacement` | Regex find-and-replace on a column |
| `format_date` | `column_name`, `target_format` | Standardize dates (e.g. `%Y-%m-%d`) |
| `strip_whitespace` | `column_name` | Strip leading/trailing whitespace |
| `merge_tables` | `right_table`, `merge_on`, `merge_how` | Merge two tables |
| `filter_rows` | `filter_condition` | Pandas query string filter |
| `fill_value` | `column_name`, `fill_value` | Fill nulls with a literal value |
| `submit` | — | End episode and trigger grading |

---

## Observation Space

Each step returns a `DataOpsObservation` JSON object:

| Field | Type | Description |
|---|---|---|
| `task_id` | str | Current task: `easy`, `medium`, or `hard` |
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

---

## Tasks

### Easy — Basic Cleaning
**Difficulty:** Easy | **Max Steps:** 30 | **Expected Score:** 0.70–0.90

A 50-row product sales table with mixed price formats, mixed date formats, inconsistent
category casing, whitespace issues, nulls, and 5 exact duplicate rows.

### Medium — Multi-Table Merge
**Difficulty:** Medium | **Max Steps:** 30 | **Expected Score:** 0.40–0.70

Two tables (users + purchases) requiring ID standardisation, date fixing, deduplication,
active-user filtering, and a left-join merge.

### Hard — PII Redaction
**Difficulty:** Hard | **Max Steps:** 30 | **Expected Score:** 0.20–0.60

30 web-scraped text documents containing emails, phone numbers, credit card numbers, and
SSNs. The agent must replace all PII with `[REDACTED]` without over-redacting clean text.

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

All rewards clamped to `[−1.0, 1.0]`.

---

## Setup

```bash
# Local
pip install fastapi uvicorn pydantic pandas numpy faker requests openai python-dotenv
python -m dataops_gym.tasks.generate_datasets
uvicorn dataops_gym.server.app:app --host 0.0.0.0 --port 7860

# Docker
docker build -t dataops-gym .
docker run -p 7860:7860 dataops-gym
```

---

## Baseline Scores

| Task | Score |
|---|---|
| easy | 0.8201 |
| medium | 0.6238 |
| hard | 0.5968 |

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/tasks` | GET | List tasks + action schema |
| `/reset` | POST | Start episode: `{"task_id": "easy"}` |
| `/step` | POST | Take action: `DataOpsAction` JSON |
| `/state` | GET | Episode state |
| `/grader` | POST | Grade current state vs golden |
| `/baseline` | POST | Run LLM baseline (needs `OPENAI_API_KEY`) |
