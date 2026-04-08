"""
Baseline inference script for DataOps Gym.
Uses OpenAI-compatible API to run an LLM agent against all 7 tasks.

Usage:
    export OPENAI_API_KEY=your_key_here
    export OPENAI_BASE_URL=https://api.openai.com/v1   # optional
    export BASELINE_MODEL=gpt-4o-mini                   # optional
    python -m dataops_gym.baseline.inference

Or via the /baseline endpoint.
"""

import os
import json
import time
import requests
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_URL = os.getenv("DATAOPS_GYM_URL", "http://localhost:7860")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
BASELINE_MODEL = os.getenv("BASELINE_MODEL", "gpt-4o-mini")


def _parse_action(text: str) -> dict:
    """Extract a JSON action dict from LLM response, handling markdown code blocks."""
    # Strip markdown code fences
    if "```" in text:
        parts = text.split("```")
        # parts[1] is inside the first fence
        if len(parts) >= 2:
            inner = parts[1]
            if inner.startswith("json"):
                inner = inner[4:]
            text = inner.strip()

    # Try direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find the first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # Fallback
    return {"action_type": "submit"}


MEDIUM_EXTRA_INSTRUCTIONS = """
MEDIUM TASK — CRITICAL MERGE INSTRUCTIONS:
This task has TWO tables: "main" (users) and "purchases". Follow this strict order:

PHASE 1 — Clean the main (users) table first:
  1. strip_whitespace on user_id
  2. apply_regex on user_id to remove any non-digit characters: pattern="[^0-9]", replacement=""
  3. cast_type on user_id to int
  4. format_date on signup_date with target_format="%Y-%m-%d"
  5. drop_duplicates

PHASE 2 — Clean the purchases table (use filter_rows with condition "False" to switch context if needed, or just proceed to merge after main is clean):
  6. cast_type on user_id to int (purchases table user_id must also be int before merging)
  - To operate on purchases, use: {"action_type": "cast_type", "column_name": "user_id", "target_type": "int", "right_table": "purchases"}
  - Actually: just ensure both tables have user_id as int. The merge will fail if types differ.

PHASE 3 — Merge ONLY after both tables are clean:
  7. merge_tables: {"action_type": "merge_tables", "right_table": "purchases", "merge_on": "user_id", "merge_how": "left"}

PHASE 4 — Post-merge cleanup, then submit.

CRITICAL RULES:
- DO NOT merge until user_id is cast to int in the main table.
- If you see a merge error about type mismatch, cast user_id to int first, then retry the merge.
- user_id values may look like "U001", "user_42", or " 7 " — strip all non-digits and cast to int.
"""

OUTLIER_EXTRA_INSTRUCTIONS = """
OUTLIER DETECTION TASK:
You have an employee dataset with planted outliers AND legitimate extreme values (e.g. executive salaries).
1. Use detect_outliers to identify suspicious values per numeric column.
2. Use clip_outliers to constrain clearly wrong values (e.g. negative salary, age > 120).
3. Be careful NOT to clip legitimate executive salaries — only clip impossible values.
4. When done, submit.

Available actions:
- {{"action_type": "detect_outliers", "column_name": "col", "outlier_method": "iqr|zscore|range"}}
- {{"action_type": "clip_outliers", "column_name": "col", "clip_min": 0, "clip_max": 500000}}
"""

SCHEMA_MIGRATION_EXTRA_INSTRUCTIONS = """
SCHEMA MIGRATION TASK:
You need to restructure this dataset by splitting combined columns, standardizing formats, and mapping codes.
1. split_column on "full_name" with delimiter=" ", new_columns=["first_name","last_name"], max_splits=1
2. apply_regex on "phone" to standardize phone numbers
3. map_values on "status_code" to convert codes ("1"="active", "2"="inactive", etc.)
4. split_column on "address" or "datetime" fields as needed
5. When done, submit.

Available actions:
- {{"action_type": "split_column", "column_name": "col", "delimiter": " ", "new_columns": ["a","b"], "max_splits": 1}}
- {{"action_type": "map_values", "column_name": "col", "value_mapping": {{"old": "new"}}}}
"""

DRIFT_EXTRA_INSTRUCTIONS = """
DRIFT DETECTION TASK:
You have historical data and must classify incoming stream batches as "normal" or "drift".
For each batch:
1. advance_stream — loads the next batch into "current_batch"
2. analyze_distribution on key columns — compare batch stats to historical
3. label_batch with drift_label="normal" or "drift" based on your analysis

Repeat for all batches, then submit.

Available actions:
- {{"action_type": "advance_stream"}}
- {{"action_type": "analyze_distribution", "column_name": "col"}}
- {{"action_type": "label_batch", "drift_label": "normal|drift"}}
"""

POISONING_EXTRA_INSTRUCTIONS = """
POISONING DETECTION TASK:
This is a sentiment dataset where some rows have been poisoned (label flips, trigger phrases).
1. Look for suspicious patterns: short generic text with strong labels, "EVAL_OVERRIDE" triggers, mismatched sentiment.
2. Use flag_rows with row_indices to mark suspicious rows.
3. Be precise — flagging clean rows hurts your F1 score.

Available actions:
- {{"action_type": "flag_rows", "row_indices": [0, 5, 10]}}
"""


def _build_system_prompt(task_id: str, obs: dict) -> str:
    base = f"""You are a data engineering agent. Your job is to clean a dataset.
Task: {obs['task_description']}

You can take these actions (one at a time, respond with JSON only):
- {{"action_type": "drop_nulls", "column_name": "col"}}
- {{"action_type": "impute_missing", "column_name": "col", "strategy": "mean|median|mode|ffill|bfill"}}
- {{"action_type": "drop_duplicates"}}
- {{"action_type": "cast_type", "column_name": "col", "target_type": "int|float|str|datetime|bool"}}
- {{"action_type": "apply_regex", "column_name": "col", "pattern": "regex", "replacement": "str"}}
- {{"action_type": "format_date", "column_name": "col", "target_format": "%Y-%m-%d"}}
- {{"action_type": "strip_whitespace", "column_name": "col"}}
- {{"action_type": "fill_value", "column_name": "col", "fill_value": "value"}}
- {{"action_type": "rename_column", "column_name": "old", "new_name": "new"}}
- {{"action_type": "merge_tables", "right_table": "name", "merge_on": "col", "merge_how": "left|inner|right|outer"}}
- {{"action_type": "filter_rows", "filter_condition": "pandas query string"}}
- {{"action_type": "clip_outliers", "column_name": "col", "clip_min": 0, "clip_max": 999}}
- {{"action_type": "detect_outliers", "column_name": "col", "outlier_method": "iqr|zscore|range"}}
- {{"action_type": "split_column", "column_name": "col", "delimiter": " ", "new_columns": ["a","b"], "max_splits": 1}}
- {{"action_type": "map_values", "column_name": "col", "value_mapping": {{"old": "new"}}}}
- {{"action_type": "advance_stream"}}
- {{"action_type": "analyze_distribution", "column_name": "col"}}
- {{"action_type": "label_batch", "drift_label": "normal|drift"}}
- {{"action_type": "flag_rows", "row_indices": [0, 1, 2]}}
- {{"action_type": "submit"}}

Look at the observation carefully: check column summaries, null counts, data types, and sample values.
When you think the data is clean enough, use submit.
Respond ONLY with a single JSON action object. No explanation."""

    task_extras = {
        "medium": MEDIUM_EXTRA_INSTRUCTIONS,
        "outlier_detection": OUTLIER_EXTRA_INSTRUCTIONS,
        "schema_migration": SCHEMA_MIGRATION_EXTRA_INSTRUCTIONS,
        "drift_detection": DRIFT_EXTRA_INSTRUCTIONS,
        "poisoning_detection": POISONING_EXTRA_INSTRUCTIONS,
    }
    if task_id in task_extras:
        base += "\n" + task_extras[task_id]

    return base


def run_task(task_id: str) -> float:
    """Run the baseline agent on a single task. Returns grader score."""
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    # 1. Reset environment
    resp = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id, "seed": 42})
    resp.raise_for_status()
    obs = resp.json()

    # 2. Build system prompt
    system_prompt = _build_system_prompt(task_id, obs)

    messages = [{"role": "system", "content": system_prompt}]

    print(f"  Starting task '{task_id}' — {obs['total_rows']} rows, "
          f"{obs['total_columns']} cols, health={obs['data_health_score']:.3f}")

    max_steps = 60 if task_id == "drift_detection" else 30
    for step in range(max_steps):
        if obs.get("done", False):
            print(f"  Episode done at step {step}.")
            break

        # Format observation for the LLM
        obs_summary = {
            "step": obs["step_number"],
            "rows": obs["total_rows"],
            "columns": obs["total_columns"],
            "health_score": obs["data_health_score"],
            "column_summaries": obs["column_summaries"],
            "preview": obs.get("preview_rows", [])[:3],
            "available_tables": obs.get("available_tables", []),
            "last_result": obs.get("last_action_result", ""),
            "error": obs.get("error"),
        }
        obs_text = json.dumps(obs_summary, indent=2, default=str)

        # Prepend a prominent error notice if the last action failed
        error = obs.get("error")
        if error:
            user_content = (
                f"⚠️ YOUR LAST ACTION FAILED WITH ERROR: {error}\n"
                f"You MUST fix this error before moving on. Choose a corrective action.\n\n"
                f"Current state:\n{obs_text}"
            )
        else:
            user_content = f"Current state:\n{obs_text}"

        messages.append({"role": "user", "content": user_content})

        # Get LLM action (retry up to 3 times on rate limit)
        action_text = '{"action_type": "submit"}'
        for attempt in range(3):
            try:
                time.sleep(2)  # 2s delay between every call to avoid rate limits
                response = client.chat.completions.create(
                    model=BASELINE_MODEL,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=300,
                )
                action_text = response.choices[0].message.content.strip()
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    wait = 5 * (attempt + 1)
                    print(f"  Rate limited (attempt {attempt+1}/3), waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  LLM error at step {step}: {e}")
                    break

        messages.append({"role": "assistant", "content": action_text})

        action = _parse_action(action_text)
        print(f"  Step {step + 1}: {action.get('action_type', '?')}"
              + (f"({action.get('column_name', '')})" if action.get("column_name") else ""))

        # Step environment
        try:
            step_resp = requests.post(f"{BASE_URL}/step", json=action)
            step_resp.raise_for_status()
            obs = step_resp.json()
        except Exception as e:
            print(f"  Step request failed: {e}")
            break

        if obs.get("error"):
            print(f"    └─ {obs['error']}")

    # 3. Get grader score
    try:
        grade_resp = requests.post(f"{BASE_URL}/grader")
        grade_resp.raise_for_status()
        result = grade_resp.json()
        score = result.get("score", 0.0001)
    except Exception as e:
        print(f"  Grader request failed: {e}")
        score = 0.0001

    print(f"  Final health: {obs.get('data_health_score', 0):.3f} | Grader score: {score}")
    return score


ALL_TASKS = ["easy", "medium", "hard", "outlier_detection", "schema_migration",
              "drift_detection", "poisoning_detection"]


def run_all_tasks() -> dict:
    """Run baseline on all 7 tasks and return scores."""
    if not OPENAI_API_KEY:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        return {}

    scores = {}
    for task_id in ALL_TASKS:
        print(f"\n{'='*50}")
        print(f"Task: {task_id.upper()}  (model: {BASELINE_MODEL})")
        print(f"{'='*50}")
        try:
            score = run_task(task_id)
            scores[task_id] = score
        except Exception as e:
            print(f"  Task '{task_id}' failed: {e}")
            scores[task_id] = 0.0001

    print(f"\n{'='*50}")
    print("BASELINE SCORES:")
    for task_id, score in scores.items():
        print(f"  {task_id:<8}: {score:.4f}")
    print(f"{'='*50}")
    return scores


if __name__ == "__main__":
    import sys
    if not OPENAI_API_KEY:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        exit(1)
    # Optional: pass a single task_id as argument, e.g. python -m ... medium
    if len(sys.argv) > 1:
        task_id = sys.argv[1]
        print(f"\n{'='*50}")
        print(f"Task: {task_id.upper()}  (model: {BASELINE_MODEL})")
        print(f"{'='*50}")
        score = run_task(task_id)
        print(f"\nScore: {score:.4f}")
    else:
        run_all_tasks()
