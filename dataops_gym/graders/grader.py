"""
Deterministic grader for DataOps Gym.
Compares agent's final cleaned dataframe against the golden reference.
"""

import re
import numpy as np
import pandas as pd


def grade(task_id: str, final_df: pd.DataFrame, golden_df: pd.DataFrame) -> float:
    """
    Grade the agent's final dataframe against the golden reference.
    Returns a score from 0.0 to 1.0, rounded to 4 decimal places.
    Always deterministic — same inputs produce same output.
    Never raises.
    """
    try:
        if task_id == "hard":
            score = _grade_hard(final_df, golden_df)
        else:
            score = _grade_tabular(final_df, golden_df)
        return round(float(np.clip(score, 0.0, 1.0)), 4)
    except Exception:
        return 0.0


# ─── TABULAR GRADER (easy + medium) ──────────────────────────────────────────

def _grade_tabular(final_df: pd.DataFrame, golden_df: pd.DataFrame) -> float:
    if golden_df is None or golden_df.empty:
        return 0.0
    if final_df is None or final_df.empty:
        return 0.0

    schema   = _schema_match(final_df, golden_df)       # 0.25
    nulls    = _null_cleanliness(final_df, golden_df)   # 0.25
    rows     = _row_accuracy(final_df, golden_df)       # 0.25
    values   = _value_accuracy(final_df, golden_df)     # 0.25

    return 0.25 * schema + 0.25 * nulls + 0.25 * rows + 0.25 * values


def _schema_match(final_df: pd.DataFrame, golden_df: pd.DataFrame) -> float:
    """Fraction of golden columns present in final with matching dtype category."""
    if len(golden_df.columns) == 0:
        return 1.0

    matches = 0
    for col in golden_df.columns:
        if col not in final_df.columns:
            continue
        if _dtype_cat(final_df[col].dtype) == _dtype_cat(golden_df[col].dtype):
            matches += 1
        else:
            # Partial credit: column exists but wrong type
            matches += 0.5

    return matches / len(golden_df.columns)


def _null_cleanliness(final_df: pd.DataFrame, golden_df: pd.DataFrame) -> float:
    """
    How close is the null pattern to golden?
    Score = 1 - (excess_nulls / total_cells), where excess_nulls = max(0, final_nulls - golden_nulls).
    Golden-vs-golden always returns 1.0.
    """
    common = [c for c in golden_df.columns if c in final_df.columns]
    if not common:
        return 0.0

    n_rows = max(len(final_df), len(golden_df))
    total_cells = n_rows * len(common)
    if total_cells == 0:
        return 1.0

    golden_nulls = int(golden_df[common].isna().sum().sum())
    final_nulls = int(final_df[common].isna().sum().sum())

    # Excess = nulls final has that golden doesn't (i.e., nulls that should have been cleaned)
    excess = max(0, final_nulls - golden_nulls)
    score = 1.0 - (excess / total_cells)
    return float(np.clip(score, 0.0, 1.0))


def _row_accuracy(final_df: pd.DataFrame, golden_df: pd.DataFrame) -> float:
    """Row count similarity to golden."""
    g = len(golden_df)
    f = len(final_df)
    if g == 0:
        return 1.0 if f == 0 else 0.0
    return float(np.clip(1.0 - abs(f - g) / g, 0.0, 1.0))


def _value_accuracy(final_df: pd.DataFrame, golden_df: pd.DataFrame) -> float:
    """
    Cell-level value match for overlapping columns.
    Aligns by position (not index) up to min(len(final), len(golden)) rows.
    """
    common_cols = [c for c in golden_df.columns if c in final_df.columns]
    if not common_cols:
        return 0.0

    n_rows = min(len(final_df), len(golden_df))
    if n_rows == 0:
        return 0.0

    total_cells = 0
    matched_cells = 0

    for col in common_cols:
        g_series = golden_df[col].iloc[:n_rows].reset_index(drop=True)
        f_series = final_df[col].iloc[:n_rows].reset_index(drop=True)
        dtype_cat = _dtype_cat(golden_df[col].dtype)

        for i in range(n_rows):
            gv = g_series.iloc[i]
            fv = f_series.iloc[i]
            total_cells += 1

            g_null = _is_null(gv)
            f_null = _is_null(fv)

            if g_null and f_null:
                matched_cells += 1
                continue
            if g_null or f_null:
                continue

            if dtype_cat == "float":
                try:
                    if np.isclose(float(fv), float(gv), rtol=0.01):
                        matched_cells += 1
                except (ValueError, TypeError):
                    pass
            elif dtype_cat == "datetime":
                try:
                    gd = pd.to_datetime(gv, errors="coerce")
                    fd = pd.to_datetime(fv, errors="coerce")
                    if pd.notna(gd) and pd.notna(fd) and gd.date() == fd.date():
                        matched_cells += 1
                except Exception:
                    pass
            elif dtype_cat in ("int",):
                try:
                    if int(float(str(fv))) == int(float(str(gv))):
                        matched_cells += 1
                except (ValueError, TypeError):
                    pass
            else:
                # String comparison: lowercased + stripped
                if str(fv).lower().strip() == str(gv).lower().strip():
                    matched_cells += 1

    if total_cells == 0:
        return 0.0
    return matched_cells / total_cells


# ─── HARD TASK GRADER (PII redaction) ────────────────────────────────────────

# Patterns used to identify PII tokens in text
_PII_PATTERNS = [
    re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),          # email
    re.compile(r"\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}"),                          # phone
    re.compile(r"\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b"),                     # credit card
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                                         # SSN
]
_REDACTED = "[REDACTED]"


def _count_redacted(text: str) -> int:
    return text.count(_REDACTED)


def _has_pii(text: str) -> bool:
    return any(p.search(text) for p in _PII_PATTERNS)


def _grade_hard(final_df: pd.DataFrame, golden_df: pd.DataFrame) -> float:
    if golden_df is None or golden_df.empty:
        return 0.0
    if final_df is None or final_df.empty:
        return 0.0

    if "text" not in final_df.columns or "text" not in golden_df.columns:
        return 0.0

    n = min(len(final_df), len(golden_df))
    if n == 0:
        return 0.0

    # Align by 'id' if possible, otherwise by position
    if "id" in final_df.columns and "id" in golden_df.columns:
        merged = golden_df[["id", "text"]].rename(columns={"text": "golden_text"}).merge(
            final_df[["id", "text"]].rename(columns={"text": "final_text"}),
            on="id", how="inner"
        )
        golden_texts = merged["golden_text"].tolist()
        final_texts = merged["final_text"].tolist()
    else:
        golden_texts = golden_df["text"].iloc[:n].tolist()
        final_texts = final_df["text"].iloc[:n].tolist()

    if not golden_texts:
        return 0.0

    pii_recall    = _pii_recall(final_texts, golden_texts)    # 0.4
    pii_precision = _pii_precision(final_texts, golden_texts)  # 0.3
    text_pres     = _text_preservation(final_texts, golden_texts)  # 0.3

    return 0.4 * pii_recall + 0.3 * pii_precision + 0.3 * text_pres


def _pii_recall(final_texts: list, golden_texts: list) -> float:
    """
    Fraction of [REDACTED] slots in golden that are also [REDACTED] in final.
    """
    total_pii = sum(_count_redacted(g) for g in golden_texts)
    if total_pii == 0:
        return 1.0  # no PII to redact → perfect recall

    caught = 0
    for f_text, g_text in zip(final_texts, golden_texts):
        g_slots = _count_redacted(g_text)
        f_slots = _count_redacted(f_text)
        caught += min(f_slots, g_slots)  # can't claim more than golden has

    return float(np.clip(caught / total_pii, 0.0, 1.0))


def _pii_precision(final_texts: list, golden_texts: list) -> float:
    """
    Penalise over-redaction: fraction of non-PII text preserved correctly.
    Measured as: docs where final has no more [REDACTED] than golden / total docs.
    """
    if not final_texts:
        return 0.0

    correct = 0
    for f_text, g_text in zip(final_texts, golden_texts):
        g_count = _count_redacted(g_text)
        f_count = _count_redacted(f_text)
        # Over-redaction: final has more [REDACTED] than golden → penalise
        if f_count <= g_count:
            correct += 1

    return float(correct / len(final_texts))


def _text_preservation(final_texts: list, golden_texts: list) -> float:
    """
    Word-overlap ratio between final and golden on non-[REDACTED] tokens.
    """
    if not final_texts:
        return 0.0

    total_score = 0.0
    for f_text, g_text in zip(final_texts, golden_texts):
        f_words = set(_non_pii_words(f_text))
        g_words = set(_non_pii_words(g_text))
        if not g_words:
            total_score += 1.0
            continue
        overlap = len(f_words & g_words)
        total_score += overlap / len(g_words)

    return float(np.clip(total_score / len(final_texts), 0.0, 1.0))


def _non_pii_words(text: str) -> list:
    """Return lowercase words from text, excluding the [REDACTED] token."""
    cleaned = text.replace(_REDACTED, " ")
    return [w.lower().strip(".,;:!?\"'()") for w in cleaned.split() if w.strip()]


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _dtype_cat(dtype) -> str:
    if pd.api.types.is_integer_dtype(dtype):
        return "int"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    return "str"


def _is_null(val) -> bool:
    try:
        return bool(pd.isna(val))
    except (TypeError, ValueError):
        return False
