"""
Auto-detection of data quality issues in arbitrary DataFrames.
Used by the /upload endpoint to build criteria for custom tasks.
"""

import re
import pandas as pd


def detect_data_issues(df: pd.DataFrame) -> dict:
    """
    Auto-detect common data quality issues in any DataFrame.
    Returns a dict of issue_type -> details.
    """
    issues = {}

    if df.empty:
        return issues

    # 1. Missing values
    null_cols = {}
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            null_cols[col] = {
                "count": null_count,
                "percentage": round(null_count / len(df), 4),
            }
    if null_cols:
        issues["missing_values"] = null_cols

    # 2. Duplicate rows
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        issues["duplicates"] = {"count": dup_count}

    # 3. Type mismatches — string columns that look numeric
    type_issues = {}
    for col in df.select_dtypes(include=["object", "str"]).columns:
        sample = df[col].dropna().head(100)
        numeric_looking = sample.astype(str).str.replace(r'[$,€£¥]', '', regex=True).str.strip()
        try:
            pd.to_numeric(numeric_looking, errors='raise')
            type_issues[col] = "Looks numeric but stored as string"
        except (ValueError, TypeError, AttributeError):
            pass
    if type_issues:
        issues["type_mismatches"] = type_issues

    # 4. Whitespace issues
    ws_issues = {}
    for col in df.select_dtypes(include=["object", "str"]).columns:
        non_null = df[col].dropna().astype(str)
        ws_count = int((non_null != non_null.str.strip()).sum())
        if ws_count > 0:
            ws_issues[col] = ws_count
    if ws_issues:
        issues["whitespace"] = ws_issues

    # 5. Inconsistent casing
    casing_issues = {}
    for col in df.select_dtypes(include=["object", "str"]).columns:
        non_null = df[col].dropna().astype(str)
        unique_raw = int(non_null.nunique())
        unique_lower = int(non_null.str.lower().nunique())
        if unique_lower < unique_raw and unique_raw <= 50:  # skip high-cardinality cols
            casing_issues[col] = {
                "unique_raw": unique_raw,
                "unique_lowered": unique_lower,
            }
    if casing_issues:
        issues["inconsistent_casing"] = casing_issues

    # 6. Potential PII
    pii_found = {}
    email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    phone_pattern = re.compile(r'[\(]?\d{3}[\)]?[\s.\-]?\d{3}[\s.\-]?\d{4}')
    for col in df.select_dtypes(include=["object", "str"]).columns:
        text = " ".join(df[col].dropna().astype(str).tolist())
        emails = email_pattern.findall(text)
        phones = phone_pattern.findall(text)
        if emails or phones:
            pii_found[col] = {
                "emails_found": len(emails),
                "phones_found": len(phones),
            }
    if pii_found:
        issues["potential_pii"] = pii_found

    # 7. Mixed date formats
    date_issues = {}
    for col in df.select_dtypes(include=["object", "str"]).columns:
        sample = df[col].dropna().head(20)
        if len(sample) == 0:
            continue
        try:
            parsed = pd.to_datetime(sample, errors='coerce', format='mixed')
            if parsed.notna().sum() > len(sample) * 0.5:
                date_issues[col] = "Likely date column stored as string"
        except Exception:
            pass
    if date_issues:
        issues["date_format_issues"] = date_issues

    return issues


def build_criteria_from_issues(df: pd.DataFrame, issues: dict) -> dict:
    """Build grading criteria based on detected issues."""
    return {
        "no_nulls_in": list(issues.get("missing_values", {}).keys()),
        "no_duplicates": "duplicates" in issues,
        "fix_types": list(issues.get("type_mismatches", {}).keys()),
        "no_whitespace_in": list(issues.get("whitespace", {}).keys()),
        "fix_casing": list(issues.get("inconsistent_casing", {}).keys()),
        "redact_pii_in": list(issues.get("potential_pii", {}).keys()),
        "fix_dates": list(issues.get("date_format_issues", {}).keys()),
        "original_row_count": len(df),
        "detected_issues": issues,
    }
