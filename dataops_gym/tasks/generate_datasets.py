"""
Dataset generation script for DataOps Gym.
Generates dirty + golden datasets for all 3 tasks using seed=42 for reproducibility.
Run: python -m dataops_gym.tasks.generate_datasets
"""

import os
import re
import json
import random
import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)


def _parse_dates(series: pd.Series) -> pd.Series:
    """Robustly parse a mixed-format date column."""
    result = pd.to_datetime(series, errors="coerce")
    # Retry remaining NaT values with explicit formats
    for fmt in ["%m/%d/%Y", "%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"]:
        mask = result.isna() & series.notna() & (series.astype(str).str.strip() != "")
        if not mask.any():
            break
        parsed = pd.to_datetime(series[mask], format=fmt, errors="coerce")
        result = result.copy()
        result[mask] = result[mask].fillna(parsed)
    return result

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)


# ─── EASY TASK ───────────────────────────────────────────────────────────────

def generate_easy():
    """Product sales table with various quality issues."""
    n = 45  # 45 unique rows + 5 duplicates = 50 total

    categories = ["Electronics", "electronics", "ELECTRONICS", "Clothing", "clothing",
                  "CLOTHING", "Books", "books", "BOOKS", "Furniture"]

    date_formats = [
        lambda d: d.strftime("%Y-%m-%d"),
        lambda d: d.strftime("%m/%d/%Y"),
        lambda d: d.strftime("%b %d, %Y"),
    ]

    rows = []
    for i in range(n):
        price_val = round(random.uniform(5.0, 2000.0), 2)
        price_str = f"${price_val:,.2f}"
        date_obj = fake.date_between(start_date="-2y", end_date="today")
        date_str = date_formats[i % 3](date_obj)

        name = fake.word().capitalize() + " " + fake.word().capitalize()
        # Add whitespace issues to ~20% of names
        if i % 5 == 0:
            name = "  " + name + "  "

        rows.append({
            "product_name": name,
            "price": price_str,
            "quantity": random.randint(1, 100),
            "date_sold": date_str,
            "category": random.choice(categories),
        })

    df = pd.DataFrame(rows)

    # Inject nulls
    null_indices_name = random.sample(range(n), 3)
    null_indices_price = random.sample(range(n), 2)
    null_indices_qty = random.sample(range(n), 4)
    null_indices_date = random.sample(range(n), 2)
    null_indices_cat = random.sample(range(n), 1)

    for i in null_indices_name:
        df.at[i, "product_name"] = None
    for i in null_indices_price:
        df.at[i, "price"] = None
    for i in null_indices_qty:
        df.at[i, "quantity"] = None
    for i in null_indices_date:
        df.at[i, "date_sold"] = None
    for i in null_indices_cat:
        df.at[i, "category"] = None

    # Add 5 exact duplicate rows
    dup_indices = random.sample(range(n), 5)
    dups = df.iloc[dup_indices].copy()
    df = pd.concat([df, dups], ignore_index=True)

    dirty_path = os.path.join(DATASETS_DIR, "easy_dirty.csv")
    df.to_csv(dirty_path, index=False)
    print(f"  easy_dirty.csv: {len(df)} rows")

    # ─── Golden: clean version ───
    golden = df.copy()

    # Drop exact duplicates
    golden = golden.drop_duplicates()

    # Strip whitespace from product_name
    golden["product_name"] = golden["product_name"].astype(str).str.strip()

    # Fill name nulls
    golden["product_name"] = golden["product_name"].replace("None", None)
    golden["product_name"] = golden["product_name"].fillna("Unknown")

    # Clean price: remove $ and , then cast to float
    golden["price"] = golden["price"].astype(str).str.replace(r"[\$,]", "", regex=True)
    golden["price"] = golden["price"].replace("None", np.nan)
    golden["price"] = pd.to_numeric(golden["price"], errors="coerce")
    price_median = golden["price"].median()
    golden["price"] = golden["price"].fillna(price_median)
    golden["price"] = golden["price"].astype(float)

    # Impute quantity with median
    golden["quantity"] = pd.to_numeric(golden["quantity"], errors="coerce")
    qty_median = golden["quantity"].median()
    golden["quantity"] = golden["quantity"].fillna(qty_median).astype(int)

    # Standardize dates to %Y-%m-%d
    golden["date_sold"] = _parse_dates(golden["date_sold"].astype(str))
    golden["date_sold"] = golden["date_sold"].dt.strftime("%Y-%m-%d")

    # Lowercase category + fill nulls with mode
    golden["category"] = golden["category"].astype(str).str.lower().str.strip()
    golden["category"] = golden["category"].replace("none", np.nan)
    cat_mode = golden["category"].mode()[0]
    golden["category"] = golden["category"].fillna(cat_mode)

    golden = golden.reset_index(drop=True)
    golden_path = os.path.join(DATASETS_DIR, "easy_golden.csv")
    golden.to_csv(golden_path, index=False)
    print(f"  easy_golden.csv: {len(golden)} rows")


# ─── MEDIUM TASK ─────────────────────────────────────────────────────────────

def _format_user_id(uid: int, style: int) -> str:
    if style == 0:
        return str(uid)
    else:
        return f"USR-{uid:03d}"


def generate_medium():
    """Users + purchases tables requiring merge and cleaning."""
    n_users = 37  # 37 unique + 3 duplicates = 40

    date_formats = [
        lambda d: d.strftime("%Y-%m-%d"),
        lambda d: d.strftime("%m/%d/%Y"),
        lambda d: d.strftime("%B %d, %Y"),
    ]
    statuses = ["active", "Active", "ACTIVE", "inactive"]

    user_rows = []
    for i in range(1, n_users + 1):
        name = fake.name()
        if i % 6 == 0:
            name = "  " + name + "  "

        email = fake.email()
        if i % 7 == 0:
            # Malform: remove @
            email = email.replace("@", "")
        elif i % 9 == 0:
            email = " " + email + " "

        date_obj = fake.date_between(start_date="-3y", end_date="-1y")
        date_str = date_formats[i % 3](date_obj)

        uid_str = _format_user_id(i, i % 2)

        user_rows.append({
            "user_id": uid_str,
            "name": name,
            "email": email,
            "signup_date": date_str,
            "status": random.choice(statuses),
        })

    users_df = pd.DataFrame(user_rows)

    # Inject nulls
    null_name = random.sample(range(n_users), 3)
    null_status = random.sample(range(n_users), 2)
    for i in null_name:
        users_df.at[i, "name"] = None
    for i in null_status:
        users_df.at[i, "status"] = None

    # Add 3 duplicate rows
    dup_idx = random.sample(range(n_users), 3)
    dups = users_df.iloc[dup_idx].copy()
    users_df = pd.concat([users_df, dups], ignore_index=True)

    users_dirty_path = os.path.join(DATASETS_DIR, "medium_users_dirty.csv")
    users_df.to_csv(users_dirty_path, index=False)
    print(f"  medium_users_dirty.csv: {len(users_df)} rows")

    # ─── Purchases table ───
    purchase_rows = []
    for pid in range(1, 61):
        uid = random.randint(1, n_users)
        uid_str = _format_user_id(uid, pid % 2)
        amount_val = round(random.uniform(5.0, 500.0), 2)
        amount_str = f"${amount_val}"

        date_obj = fake.date_between(start_date="-2y", end_date="today")
        date_str = date_formats[pid % 3](date_obj)

        purchase_rows.append({
            "purchase_id": pid,
            "user_id": uid_str,
            "amount": amount_str,
            "purchase_date": date_str,
            "product": fake.word().capitalize(),
        })

    purchases_df = pd.DataFrame(purchase_rows)

    # Inject some null amounts
    null_amt = random.sample(range(60), 4)
    for i in null_amt:
        purchases_df.at[i, "amount"] = None

    purchases_dirty_path = os.path.join(DATASETS_DIR, "medium_purchases_dirty.csv")
    purchases_df.to_csv(purchases_dirty_path, index=False)
    print(f"  medium_purchases_dirty.csv: {len(purchases_df)} rows")

    # ─── Golden: clean merged result ───
    users_clean = users_df.copy()
    users_clean = users_clean.drop_duplicates()

    # Standardize user_id to int
    users_clean["user_id"] = (
        users_clean["user_id"].astype(str)
        .str.replace(r"USR-0*", "", regex=True)
        .str.strip()
    )
    users_clean["user_id"] = pd.to_numeric(users_clean["user_id"], errors="coerce").astype("Int64")

    # Clean name
    users_clean["name"] = users_clean["name"].astype(str).str.strip()
    users_clean["name"] = users_clean["name"].replace("None", np.nan).fillna("Unknown")

    # Standardize status
    users_clean["status"] = users_clean["status"].astype(str).str.lower().str.strip()
    users_clean["status"] = users_clean["status"].replace("none", np.nan)
    users_clean["status"] = users_clean["status"].fillna("inactive")

    # Standardize signup_date
    users_clean["signup_date"] = _parse_dates(
        users_clean["signup_date"].astype(str)
    ).dt.strftime("%Y-%m-%d")

    # Keep only active users
    users_clean = users_clean[users_clean["status"] == "active"].copy()

    # Clean purchases
    purchases_clean = purchases_df.copy()
    purchases_clean["user_id"] = (
        purchases_clean["user_id"].astype(str)
        .str.replace(r"USR-0*", "", regex=True)
        .str.strip()
    )
    purchases_clean["user_id"] = pd.to_numeric(purchases_clean["user_id"], errors="coerce").astype("Int64")

    purchases_clean["amount"] = (
        purchases_clean["amount"].astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .replace("None", np.nan)
    )
    purchases_clean["amount"] = pd.to_numeric(purchases_clean["amount"], errors="coerce")
    amt_median = purchases_clean["amount"].median()
    purchases_clean["amount"] = purchases_clean["amount"].fillna(amt_median)

    purchases_clean["purchase_date"] = _parse_dates(
        purchases_clean["purchase_date"].astype(str)
    ).dt.strftime("%Y-%m-%d")

    # Left join: active users → purchases
    golden = users_clean.merge(purchases_clean, on="user_id", how="left")
    golden = golden.reset_index(drop=True)

    golden_path = os.path.join(DATASETS_DIR, "medium_golden.csv")
    golden.to_csv(golden_path, index=False)
    print(f"  medium_golden.csv: {len(golden)} rows")


# ─── HARD TASK ───────────────────────────────────────────────────────────────

def _make_email():
    return fake.email()

def _make_phone():
    styles = [
        lambda: f"({random.randint(200,999)}) {random.randint(100,999)}-{random.randint(1000,9999)}",
        lambda: f"{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
        lambda: f"{random.randint(200,999)}.{random.randint(100,999)}.{random.randint(1000,9999)}",
    ]
    return random.choice(styles)()

def _make_cc():
    groups = [str(random.randint(1000, 9999)) for _ in range(4)]
    return "-".join(groups)

def _make_ssn():
    return f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}"

def _clean_pii(text: str) -> str:
    """Replace all PII with [REDACTED]."""
    # Email
    text = re.sub(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", "[REDACTED]", text)
    # Phone: (555) 123-4567 or 555-123-4567 or 555.123.4567
    text = re.sub(r"\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}", "[REDACTED]", text)
    # Credit card: 4 groups of 4 digits separated by - or space
    text = re.sub(r"\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b", "[REDACTED]", text)
    # SSN: 123-45-6789
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED]", text)
    return text

TOPICS = [
    ("technology", [
        "The latest advancements in artificial intelligence are reshaping industries worldwide.",
        "Cloud computing has dramatically reduced the cost of deploying scalable applications.",
        "Quantum computing promises to solve problems that classical computers cannot handle efficiently.",
        "Machine learning models require vast amounts of clean, labeled data to perform well.",
        "Edge computing brings processing power closer to where data is generated.",
        "Open source software has accelerated innovation across the technology sector.",
        "Cybersecurity threats are becoming more sophisticated with the rise of AI-powered attacks.",
    ]),
    ("science", [
        "Researchers have discovered a new method for synthesizing biodegradable plastics.",
        "The study of microbiomes is revealing complex relationships between gut bacteria and health.",
        "Climate scientists are developing more accurate models for predicting regional weather patterns.",
        "Gene editing technologies like CRISPR hold promise for treating inherited diseases.",
        "Astronomers have identified several exoplanets within the habitable zones of nearby stars.",
        "Neuroscience research is uncovering how sleep affects long-term memory consolidation.",
        "Renewable energy sources now account for a growing share of global electricity production.",
    ]),
    ("business", [
        "Supply chain disruptions have prompted companies to diversify their supplier networks.",
        "Remote work has permanently changed how organizations think about office space.",
        "Consumer behavior shifted dramatically toward digital channels during the pandemic.",
        "Startups are increasingly relying on data analytics to make faster strategic decisions.",
        "Mergers and acquisitions activity has surged as companies seek competitive advantages.",
        "Sustainability initiatives are becoming a core part of corporate strategy.",
        "The gig economy continues to grow as workers seek more flexible employment arrangements.",
    ]),
]

def _make_paragraph(topic_sentences):
    n = random.randint(2, 4)
    return " ".join(random.sample(topic_sentences, min(n, len(topic_sentences))))


def generate_hard():
    """30 text documents with embedded PII for redaction task."""
    documents = []

    # Plan: 10 emails, 8 phones, 6 CCs, 4 SSNs (some docs have multiple types, some have none)
    # Total 30 docs
    pii_assignments = (
        ["email"] * 4 +           # email only
        ["phone"] * 2 +           # phone only
        ["cc"] * 2 +              # cc only
        ["ssn"] * 1 +             # ssn only
        ["email", "phone"] * 2 +  # email + phone (4 docs)
        ["email", "cc"] * 1 +     # email + cc (2 docs)
        ["phone", "cc"] * 1 +     # phone + cc (2 docs)
        ["email", "ssn"] * 1 +    # email + ssn (2 docs)
        ["phone", "ssn"] * 1 +    # phone + ssn (2 docs)
        ["cc", "ssn"] * 1 +       # cc + ssn (2 docs)
        ["none"] * 5              # no PII (5 docs)
    )

    # Flatten paired assignments
    flat_assignments = []
    i = 0
    while i < len(pii_assignments):
        item = pii_assignments[i]
        if isinstance(item, list):
            flat_assignments.append(item)
            i += 1
        else:
            flat_assignments.append([item])
            i += 1

    random.shuffle(flat_assignments)

    # Ensure we have exactly 30
    while len(flat_assignments) < 30:
        flat_assignments.append(["none"])
    flat_assignments = flat_assignments[:30]

    for doc_id in range(1, 31):
        topic_name, topic_sentences = random.choice(TOPICS)
        base_text = _make_paragraph(topic_sentences)
        pii_types = flat_assignments[doc_id - 1]

        text = base_text

        for pii_type in pii_types:
            if pii_type == "email":
                email = _make_email()
                name = fake.first_name().lower()
                insertions = [
                    f"For more information, contact {name} at {email}.",
                    f"Please reach out to {email} for further details.",
                    f"Send your inquiries to {email} directly.",
                ]
                text = text + " " + random.choice(insertions)

            elif pii_type == "phone":
                phone = _make_phone()
                insertions = [
                    f"Call us at {phone} for immediate assistance.",
                    f"Reach our support team at {phone}.",
                    f"You can also reach us at {phone} during business hours.",
                ]
                text = text + " " + random.choice(insertions)

            elif pii_type == "cc":
                cc = _make_cc()
                insertions = [
                    f"The payment was processed with card number {cc}.",
                    f"Transaction completed using card {cc}.",
                    f"Billing was applied to card ending in {cc}.",
                ]
                text = text + " " + random.choice(insertions)

            elif pii_type == "ssn":
                ssn = _make_ssn()
                insertions = [
                    f"The individual's SSN on file is {ssn}.",
                    f"Social Security Number: {ssn} was used for verification.",
                    f"Records indicate SSN {ssn} for this account.",
                ]
                text = text + " " + random.choice(insertions)

        documents.append({"id": doc_id, "text": text.strip()})

    dirty_path = os.path.join(DATASETS_DIR, "hard_dirty.json")
    with open(dirty_path, "w") as f:
        json.dump(documents, f, indent=2)
    print(f"  hard_dirty.json: {len(documents)} documents")

    # ─── Golden: all PII replaced ───
    golden_docs = []
    for doc in documents:
        golden_docs.append({"id": doc["id"], "text": _clean_pii(doc["text"])})

    golden_path = os.path.join(DATASETS_DIR, "hard_golden.json")
    with open(golden_path, "w") as f:
        json.dump(golden_docs, f, indent=2)
    print(f"  hard_golden.json: {len(golden_docs)} documents")


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating datasets...\n")

    print("EASY task:")
    generate_easy()

    print("\nMEDIUM task:")
    generate_medium()

    print("\nHARD task:")
    generate_hard()

    print("\nAll datasets saved to:", DATASETS_DIR)
    print("\nFiles generated:")
    for fname in sorted(os.listdir(DATASETS_DIR)):
        fpath = os.path.join(DATASETS_DIR, fname)
        print(f"  {fname} ({os.path.getsize(fpath):,} bytes)")

    print("\nSample easy_dirty.csv (first 3 rows):")
    df = pd.read_csv(os.path.join(DATASETS_DIR, "easy_dirty.csv"))
    print(df.head(3).to_string())

    print("\nSample hard_dirty.json (doc 1):")
    with open(os.path.join(DATASETS_DIR, "hard_dirty.json")) as f:
        docs = json.load(f)
    print(json.dumps(docs[0], indent=2))
    print("\nDONE.")
