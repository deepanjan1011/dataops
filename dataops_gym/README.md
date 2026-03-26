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

A 50-row product sales table with:
- Mixed price formats (`$1,299.99` strings → float)
- Mixed date formats (`2024-01-15`, `01/15/2024`, `Jan 15, 2024`)
- Inconsistent category casing (`Electronics`, `electronics`, `ELECTRONICS`)
- Leading/trailing whitespace in product names
- Null values across all columns
- 5 exact duplicate rows

### Medium — Multi-Table Merge
**Difficulty:** Medium | **Max Steps:** 30 | **Expected Score:** 0.40–0.70

Two tables (users + purchases) requiring:
- Standardising mixed `user_id` formats (`USR-001` vs `1`)
- Date standardisation across both tables
- Amount string conversion (`$49.99` → float)
- Deduplication
- Filtering to active users only
- Left-join merge without losing active users

### Hard — PII Redaction
**Difficulty:** Hard | **Max Steps:** 30 | **Expected Score:** 0.20–0.60

30 web-scraped text documents containing:
- Email addresses
- Phone numbers `(555) 123-4567` / `555-123-4567`
- Credit card numbers `4532-1234-5678-9012`
- SSN-like numbers `123-45-6789`
- Some documents with **no PII** (testing for over-redaction)

The agent must use `apply_regex` to replace each PII type with `[REDACTED]`.

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
| PII type successfully redacted | `+0.2` per type (hard task) |

All rewards clamped to `[−1.0, 1.0]`.

Health score components:
- Null ratio: `(1 − nulls/cells) × 0.3`
- Type correctness vs golden: `× 0.3`
- Row retention vs golden: `× 0.2`
- Duplicate ratio: `× 0.2`

---

## Setup Instructions

### Local

```bash
git clone <repo>
cd dataops_gym
python3 -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn pydantic pandas numpy faker requests openai python-dotenv
python -m dataops_gym.tasks.generate_datasets
uvicorn dataops_gym.server.app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t dataops-gym .
docker run -p 7860:7860 dataops-gym
```

### Hugging Face Spaces

1. Create a new Space (SDK: Docker, port: 7860)
2. Push this repo — the Dockerfile at the root auto-builds

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/dataops-gym
git push hf main
```

---

## Baseline Scores

Scores using `google/gemma-3-12b-it` via OpenRouter:

| Task | Score |
|---|---|
| easy | 0.8201 |
| medium | 0.6238 |
| hard | 0.5968 |

> These are unassisted baseline scores (no cleaning steps). LLM-driven runs with sufficient
> API rate limits will score higher.

---

## API Reference

### `POST /reset`
```json
// Request
{"task_id": "easy"}

// Response: DataOpsObservation (see Observation Space above)
```

### `POST /step`
```json
// Request: DataOpsAction
{"action_type": "drop_duplicates"}
{"action_type": "cast_type", "column_name": "price", "target_type": "float"}

// Response: DataOpsObservation
```

### `GET /state`
```json
// Response
{"episode_id": "...", "task_id": "easy", "step_count": 3, "max_steps": 30,
 "cumulative_reward": 0.12, "done": false}
```

### `GET /health`
```json
{"status": "healthy"}
```

### `GET /tasks`
Returns list of all 3 tasks with full `DataOpsAction` JSON schema.

### `POST /grader`
```json
// Response
{"task_id": "easy", "score": 0.87, "details": {...}}
```

### `POST /baseline`
Runs LLM baseline agent on all 3 tasks. Requires `OPENAI_API_KEY`.
```json
{"easy": 0.82, "medium": 0.62, "hard": 0.60}
```
