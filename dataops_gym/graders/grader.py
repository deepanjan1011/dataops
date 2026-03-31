"""
Deterministic grader for DataOps Gym.
Compares agent's final cleaned dataframe against the golden reference.
"""

import re
import numpy as np
import pandas as pd


def grade_by_criteria(task_id: str, final_df: pd.DataFrame, criteria: dict) -> float:
    """
    Grade based on criteria dict instead of comparing to a golden dataset.
    Enables grading on procedurally generated datasets where no golden file exists.
    Returns a score from 0.0 to 1.0, rounded to 4 decimal places. Never raises.
    """
    try:
        if final_df is None or final_df.empty:
            return 0.0
        if not criteria:
            return 0.0

        scores = []

        if task_id == "easy":
            # 1. Null cleanliness (0.25)
            null_score = 1.0
            required = criteria.get("no_nulls_in", [])
            for col in required:
                if col in final_df.columns:
                    null_score -= final_df[col].isnull().sum() / len(final_df) / max(len(required), 1)
            scores.append(("null_cleanliness", max(0.0, null_score), 0.25))

            # 2. Type correctness (0.25)
            type_checks = criteria.get("column_types", {})
            type_score = 0.0
            for col, expected in type_checks.items():
                if col not in final_df.columns:
                    continue
                if expected == "float" and pd.api.types.is_float_dtype(final_df[col]):
                    type_score += 1.0
                elif expected == "numeric" and pd.api.types.is_numeric_dtype(final_df[col]):
                    type_score += 1.0
                elif expected == "datetime_str":
                    sample = final_df[col].dropna().head(5)
                    try:
                        pd.to_datetime(sample, errors="raise")
                        type_score += 1.0
                    except Exception:
                        pass
            if type_checks:
                type_score /= len(type_checks)
            scores.append(("type_correctness", type_score, 0.25))

            # 3. No duplicates (0.20)
            dup_score = 1.0 if not final_df.duplicated().any() else (
                1.0 - final_df.duplicated().sum() / len(final_df)
            )
            scores.append(("no_duplicates", float(dup_score), 0.20))

            # 4. Format compliance (0.15)
            fmt_score, checks = 0.0, 0
            if criteria.get("category_lowercase") and "category" in final_df.columns:
                non_null = final_df["category"].dropna()
                if len(non_null) > 0:
                    fmt_score += (non_null == non_null.str.lower()).sum() / len(non_null)
                    checks += 1
            for col in criteria.get("no_whitespace_in", []):
                if col in final_df.columns:
                    non_null = final_df[col].dropna().astype(str)
                    if len(non_null) > 0:
                        fmt_score += (non_null == non_null.str.strip()).sum() / len(non_null)
                        checks += 1
            scores.append(("format_compliance", fmt_score / checks if checks else 0.0, 0.15))

            # 5. Row retention (0.15)
            original = criteria.get("original_row_count", len(final_df))
            retention = min(len(final_df) / original, 1.0) if original > 0 else 0.0
            scores.append(("row_retention", float(retention), 0.15))

        elif task_id == "medium":
            # 1. Type correctness: user_id int + amount numeric (0.25)
            type_score, checks = 0.0, 0
            if criteria.get("user_id_is_integer") and "user_id" in final_df.columns:
                try:
                    final_df["user_id"].astype(int)
                    type_score += 1.0
                except (ValueError, TypeError):
                    pass
                checks += 1
            if criteria.get("amount_is_numeric") and "amount" in final_df.columns:
                if pd.api.types.is_numeric_dtype(final_df["amount"]):
                    type_score += 1.0
                checks += 1
            scores.append(("type_correctness", type_score / checks if checks else 0.0, 0.25))

            # 2. No duplicates (0.20)
            dup_score = 1.0 if not final_df.duplicated().any() else (
                1.0 - final_df.duplicated().sum() / len(final_df)
            )
            scores.append(("no_duplicates", float(dup_score), 0.20))

            # 3. Status compliance: lowercase + all active (0.20)
            status_score = 0.0
            if "status" in final_df.columns:
                non_null = final_df["status"].dropna()
                if len(non_null) > 0:
                    lower_ratio = (non_null == non_null.str.lower()).sum() / len(non_null)
                    active_ratio = (non_null.str.lower() == "active").sum() / len(non_null)
                    status_score = (lower_ratio + active_ratio) / 2
            scores.append(("status_compliance", float(status_score), 0.20))

            # 4. Null cleanliness (0.20)
            null_score = 1.0
            required = criteria.get("no_nulls_in", [])
            for col in required:
                if col in final_df.columns:
                    null_score -= final_df[col].isnull().sum() / len(final_df) / max(len(required), 1)
            scores.append(("null_cleanliness", max(0.0, null_score), 0.20))

            # 5. Tables merged check (0.15)
            merge_score = 1.0 if ("amount" in final_df.columns and "name" in final_df.columns) else 0.0
            scores.append(("tables_merged", merge_score, 0.15))

        elif task_id == "hard":
            import re
            patterns = criteria.get("pii_patterns", {})
            all_text = " ".join(final_df["text"].astype(str).tolist()) if "text" in final_df.columns else ""

            total_remaining = sum(len(re.findall(p, all_text)) for p in patterns.values())
            redaction_count = all_text.count("[REDACTED]")
            expected_pii = sum(criteria.get("pii_counts", {}).values())

            recall = min(redaction_count / max(expected_pii, 1), 1.0)
            scores.append(("pii_recall", float(recall), 0.40))

            precision = 1.0 if total_remaining == 0 else max(0.0, 1.0 - total_remaining / max(expected_pii, 1))
            scores.append(("pii_precision", float(precision), 0.30))

            avg_len = final_df["text"].astype(str).str.len().mean() if "text" in final_df.columns else 0
            preservation = min(avg_len / 100, 1.0)
            scores.append(("text_preservation", float(preservation), 0.30))

        elif task_id == "outlier_detection":
            scores = []
            # 1. Outlier removal (0.35): Check if planted outlier values are gone
            outlier_indices = criteria.get("outlier_indices", {})
            removed = 0
            for idx_str, cols in outlier_indices.items():
                idx = int(idx_str)
                for col, val in cols.items():
                    if idx >= len(final_df) or col not in final_df.columns:
                        removed += 1
                    elif final_df.at[idx, col] != val:
                        removed += 1
            outlier_total = sum(len(v) for v in outlier_indices.values())
            removal_score = removed / max(outlier_total, 1)
            scores.append(("outlier_removal", removal_score, 0.35))

            # 2. Legitimate preservation (0.35): Legitimate extremes still present
            legit_indices = criteria.get("legitimate_extreme_indices", {})
            preserved = 0
            legit_total = sum(len(v) for v in legit_indices.values())
            for idx_str, cols in legit_indices.items():
                idx = int(idx_str)
                for col, val in cols.items():
                    if idx < len(final_df) and col in final_df.columns:
                        try:
                            if abs(float(final_df.at[idx, col]) - float(val)) < 1.0:
                                preserved += 1
                        except (ValueError, TypeError):
                            pass
            preservation_score = preserved / max(legit_total, 1)
            scores.append(("legitimate_preservation", preservation_score, 0.35))

            # 3. Row retention (0.15)
            original = criteria.get("original_row_count", len(final_df))
            retention = min(len(final_df) / max(original, 1), 1.0)
            scores.append(("row_retention", retention, 0.15))

            # 4. Data integrity (0.15): No new nulls, types correct
            null_ratio = final_df.isnull().sum().sum() / max(final_df.size, 1)
            integrity = 1.0 - null_ratio
            scores.append(("data_integrity", integrity, 0.15))

        elif task_id == "schema_migration":
            scores = []
            target = criteria.get("target_schema", {})
            original_cols = criteria.get("original_columns", [])

            # 1. Schema match (0.40): Output columns match target
            target_cols = set(target.keys())
            actual_cols = set(final_df.columns)
            matching = len(target_cols & actual_cols)
            schema_score = matching / max(len(target_cols), 1)
            scores.append(("schema_match", schema_score, 0.40))

            # 2. Value correctness (0.30): Spot checks
            value_score = 0.0
            checks = 0
            if "price" in final_df.columns:
                try:
                    pd.to_numeric(final_df["price"], errors="raise")
                    value_score += 1.0
                except (ValueError, TypeError):
                    pass
                checks += 1
            if "status" in final_df.columns:
                valid_statuses = set(criteria.get("status_mapping", {}).values())
                if valid_statuses:
                    actual = set(final_df["status"].dropna().unique())
                    if actual.issubset(valid_statuses):
                        value_score += 1.0
                checks += 1
            if "phone" in final_df.columns:
                sample = final_df["phone"].dropna().head(10)
                if len(sample) > 0:
                    digits_only = sample.str.match(r'^\d+$').sum()
                    value_score += digits_only / len(sample)
                checks += 1
            if checks > 0:
                value_score /= checks
            scores.append(("value_correctness", value_score, 0.30))

            # 3. Row preservation (0.15)
            original = criteria.get("original_row_count", len(final_df))
            retention = min(len(final_df) / max(original, 1), 1.0)
            scores.append(("row_retention", retention, 0.15))

            # 4. Old columns removed (0.15)
            remaining_old = len(set(original_cols) & actual_cols)
            removal_score = 1.0 - (remaining_old / max(len(original_cols), 1))
            scores.append(("old_columns_removed", removal_score, 0.15))

        elif task_id == "custom":
            # Grade based on whether detected issues were fixed
            issues = criteria.get("detected_issues", {})
            if not issues:
                return 1.0  # nothing to fix → perfect

            checks = []

            # Null cleanliness (weighted by how many null cols exist)
            null_cols = list(issues.get("missing_values", {}).keys())
            if null_cols:
                null_score = 1.0
                present = [c for c in null_cols if c in final_df.columns]
                if present:
                    total_null = sum(int(final_df[c].isnull().sum()) for c in present)
                    total_cells = len(final_df) * len(present)
                    null_score = max(0.0, 1.0 - total_null / total_cells) if total_cells > 0 else 1.0
                checks.append(("null_cleanliness", null_score, 0.30))

            # Duplicate removal
            if "duplicates" in issues:
                dup_score = 1.0 if not final_df.duplicated().any() else (
                    1.0 - final_df.duplicated().sum() / len(final_df)
                )
                checks.append(("no_duplicates", float(dup_score), 0.20))

            # Type fixes
            fix_type_cols = [c for c in criteria.get("fix_types", []) if c in final_df.columns]
            if fix_type_cols:
                fixed = sum(1 for c in fix_type_cols if pd.api.types.is_numeric_dtype(final_df[c]))
                checks.append(("type_correctness", fixed / len(fix_type_cols), 0.20))

            # Whitespace
            ws_cols = [c for c in criteria.get("no_whitespace_in", []) if c in final_df.columns]
            if ws_cols:
                ws_score = 0.0
                for c in ws_cols:
                    non_null = final_df[c].dropna().astype(str)
                    if len(non_null) > 0:
                        ws_score += (non_null == non_null.str.strip()).sum() / len(non_null)
                checks.append(("whitespace", ws_score / len(ws_cols), 0.15))

            # Casing
            casing_cols = [c for c in criteria.get("fix_casing", []) if c in final_df.columns]
            if casing_cols:
                casing_score = 0.0
                for c in casing_cols:
                    non_null = final_df[c].dropna().astype(str)
                    if len(non_null) > 0:
                        casing_score += (non_null == non_null.str.lower()).sum() / len(non_null)
                checks.append(("casing", casing_score / len(casing_cols), 0.15))

            if not checks:
                return 0.5  # issues detected but none we can grade → neutral

            total_w = sum(w for _, _, w in checks)
            scores = [(s, w / total_w) for _, s, w in checks]  # renormalize weights
            total = sum(s * w for s, w in scores)
            return round(float(np.clip(total, 0.0, 1.0)), 4)

        total = sum(score * weight for _, score, weight in scores)
        return round(float(np.clip(total, 0.0, 1.0)), 4)
    except Exception:
        return 0.0


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
