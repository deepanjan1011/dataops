"""
Baseline inference script for DataOps Gym.
Uses OpenAI-compatible API (default: OpenRouter) to run an LLM agent against all 3 tasks.

Usage:
    export OPENAI_API_KEY=your_key_here
    export OPENAI_BASE_URL=https://openrouter.ai/api/v1   # optional
    export BASELINE_MODEL=meta-llama/llama-3.1-8b-instruct:free  # optional
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
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
BASELINE_MODEL = os.getenv("BASELINE_MODEL", "meta-llama/llama-3.1-8b-instruct:free")


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


def run_task(task_id: str) -> float:
    """Run the baseline agent on a single task. Returns grader score."""
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    # 1. Reset environment
    resp = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    obs = resp.json()

    # 2. Build system prompt
    system_prompt = f"""You are a data engineering agent. Your job is to clean a dataset.
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
- {{"action_type": "submit"}}

Look at the observation carefully: check column summaries, null counts, data types, and sample values.
When you think the data is clean enough, use submit.
Respond ONLY with a single JSON action object. No explanation."""

    messages = [{"role": "system", "content": system_prompt}]

    print(f"  Starting task '{task_id}' — {obs['total_rows']} rows, "
          f"{obs['total_columns']} cols, health={obs['data_health_score']:.3f}")

    for step in range(30):
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
        messages.append({"role": "user", "content": f"Current state:\n{obs_text}"})

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
        score = result.get("score", 0.0)
    except Exception as e:
        print(f"  Grader request failed: {e}")
        score = 0.0

    print(f"  Final health: {obs.get('data_health_score', 0):.3f} | Grader score: {score}")
    return score


def run_all_tasks() -> dict:
    """Run baseline on all 3 tasks and return scores."""
    if not OPENAI_API_KEY:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        return {}

    scores = {}
    for task_id in ["easy", "medium", "hard"]:
        print(f"\n{'='*50}")
        print(f"Task: {task_id.upper()}  (model: {BASELINE_MODEL})")
        print(f"{'='*50}")
        try:
            score = run_task(task_id)
            scores[task_id] = score
        except Exception as e:
            print(f"  Task '{task_id}' failed: {e}")
            scores[task_id] = 0.0

    print(f"\n{'='*50}")
    print("BASELINE SCORES:")
    for task_id, score in scores.items():
        print(f"  {task_id:<8}: {score:.4f}")
    print(f"{'='*50}")
    return scores


if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        exit(1)
    run_all_tasks()
