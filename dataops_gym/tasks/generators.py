"""
Procedural dataset generators for DataOps Gym.
Every call produces fresh dirty data with the same problem types.
Difficulty parameters control how messy the injected data is.
"""

import random
import pandas as pd
import numpy as np
from faker import Faker
from typing import Optional, Tuple, Dict


def generate_easy_dataset(
    seed: Optional[int] = None,
    num_rows: int = 50,
    null_percentage: float = 0.08,
    duplicate_rate: float = 0.10,
    format_inconsistency: float = 0.5,
) -> Tuple[pd.DataFrame, dict]:
    """
    Generate a dirty product sales dataset with known issues.

    Args:
        seed: Random seed for reproducibility (None = random)
        num_rows: Base row count before duplicates are injected
        null_percentage: Fraction of cells to null out per nullable column (0.0–0.5)
        duplicate_rate: Fraction of extra duplicate rows to inject (0.0–0.3)
        format_inconsistency: 0.0 = all dates in YYYY-MM-DD, 1.0 = max variety (0.0–1.0)

    Issues injected:
        - price: stored as "$1,299.99" strings
        - product_name: leading/trailing whitespace on ~15% of rows
        - nulls in product_name, quantity, date_sold, category (controlled by null_percentage)
        - date_sold: mixed formats (controlled by format_inconsistency)
        - category: inconsistent casing
        - duplicate rows (controlled by duplicate_rate)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        fake = Faker()
        Faker.seed(seed)
    else:
        fake = Faker()

    categories = ["electronics", "clothing", "home", "sports", "books", "food", "toys"]
    data = {
        "product_name": [fake.catch_phrase() for _ in range(num_rows)],
        "price": [round(random.uniform(5.0, 2000.0), 2) for _ in range(num_rows)],
        "quantity": [random.randint(1, 100) for _ in range(num_rows)],
        "date_sold": [fake.date_between(start_date="-2y", end_date="today") for _ in range(num_rows)],
        "category": [random.choice(categories) for _ in range(num_rows)],
    }
    df = pd.DataFrame(data)

    # ---- Inject dirt ----

    # 1. Price as strings with $ and ,
    df["price"] = df["price"].apply(lambda x: f"${x:,.2f}")

    # 2. Whitespace on product_name (~15% fixed — this is a format issue, not null)
    ws_mask = np.random.random(num_rows) < 0.15
    df.loc[ws_mask, "product_name"] = df.loc[ws_mask, "product_name"].apply(
        lambda x: f"  {x}  " if isinstance(x, str) else x
    )

    # 3. Nulls — apply null_percentage to each nullable column
    for col in ["product_name", "quantity", "date_sold", "category"]:
        null_mask = np.random.random(num_rows) < null_percentage
        df.loc[null_mask, col] = None

    # 4. Date format inconsistency
    # format_inconsistency=0.0 → always YYYY-MM-DD; 1.0 → full 3-format variety
    all_date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%b %d, %Y"]
    n_formats = max(1, round(1 + format_inconsistency * (len(all_date_formats) - 1)))
    date_formats = all_date_formats[:n_formats]
    df["date_sold"] = df["date_sold"].apply(
        lambda d: d.strftime(random.choice(date_formats)) if pd.notna(d) else None
    )

    # 5. Category casing issues (always injected — it's a structural issue)
    casing_funcs = [str.upper, str.lower, str.title, str.upper]
    df["category"] = df["category"].apply(
        lambda x: random.choice(casing_funcs)(x) if isinstance(x, str) else x
    )

    # 6. Duplicate rows controlled by duplicate_rate
    n_dupes = int(num_rows * duplicate_rate)
    if n_dupes > 0:
        dupe_indices = np.random.choice(num_rows, size=n_dupes, replace=True)
        dupes = df.iloc[dupe_indices].copy()
        df = pd.concat([df, dupes], ignore_index=True)

    criteria = {
        "expected_columns": ["product_name", "price", "quantity", "date_sold", "category"],
        "column_types": {"price": "float", "quantity": "numeric", "date_sold": "datetime_str"},
        "no_nulls_in": ["product_name", "price", "quantity", "date_sold", "category"],
        "no_duplicates": True,
        "category_lowercase": True,
        "date_format": "%Y-%m-%d",
        "price_is_numeric": True,
        "no_whitespace_in": ["product_name", "category"],
        "original_row_count": num_rows,
        "difficulty_params": {
            "null_percentage": null_percentage,
            "duplicate_rate": duplicate_rate,
            "format_inconsistency": format_inconsistency,
        },
    }
    return df, criteria


def generate_medium_dataset(
    seed: Optional[int] = None,
    num_users: int = 40,
    num_purchases: int = 60,
    null_percentage: float = 0.05,
    duplicate_rate: float = 0.08,
    id_format_variety: float = 0.7,
) -> Tuple[Dict[str, pd.DataFrame], dict]:
    """
    Generate two dirty related tables (users + purchases).

    Args:
        seed: Random seed
        num_users: Number of user rows before duplicates
        num_purchases: Number of purchase rows
        null_percentage: Fraction of cells to null out per nullable column
        duplicate_rate: Fraction of extra duplicate rows to inject into users table
        id_format_variety: 0.0 = all plain integers, 1.0 = max messy formats (0.0–1.0)

    Issues:
        - user_id: mixed formats ("USR-001" vs "1" vs "001") controlled by id_format_variety
        - users.name: whitespace, nulls
        - users.email: some malformed
        - users.signup_date: mixed date formats
        - users.status: inconsistent casing, nulls
        - purchases.amount: strings with "$"
        - purchases.purchase_date: mixed formats
        - Duplicate rows in users
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        fake = Faker()
        Faker.seed(seed)
    else:
        fake = Faker()

    user_ids_clean = list(range(1, num_users + 1))

    def messy_user_id(uid):
        # id_format_variety controls how often we use non-plain formats
        r = random.random()
        if r > id_format_variety:
            return str(uid)  # plain integer string
        fmt = random.choice(["padded", "prefixed"])
        return f"{uid:03d}" if fmt == "padded" else f"USR-{uid:03d}"

    # Users table
    users_data = {
        "user_id": [messy_user_id(uid) for uid in user_ids_clean],
        "name": [fake.name() for _ in range(num_users)],
        "email": [fake.email() for _ in range(num_users)],
        "signup_date": [fake.date_between(start_date="-3y", end_date="today") for _ in range(num_users)],
        "status": [random.choice(["active", "inactive"]) for _ in range(num_users)],
    }
    users_df = pd.DataFrame(users_data)

    # Name whitespace (~15% fixed structural issue)
    ws_mask = np.random.random(num_users) < 0.15
    users_df.loc[ws_mask, "name"] = users_df.loc[ws_mask, "name"].apply(
        lambda x: f"  {x}  " if isinstance(x, str) else x
    )

    # Nulls controlled by null_percentage
    for col in ["name", "status"]:
        null_mask = np.random.random(num_users) < null_percentage
        users_df.loc[null_mask, col] = None

    # Email issues (~10% fixed — structural format issue)
    bad_email_mask = np.random.random(num_users) < 0.10
    users_df.loc[bad_email_mask, "email"] = users_df.loc[bad_email_mask, "email"].apply(
        lambda x: x.replace("@", " ") if isinstance(x, str) else x
    )

    # Mixed date formats (always inject — structural issue)
    date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y"]
    users_df["signup_date"] = users_df["signup_date"].apply(
        lambda d: d.strftime(random.choice(date_formats)) if pd.notna(d) else None
    )

    # Status casing (always inject)
    users_df["status"] = users_df["status"].apply(
        lambda x: random.choice([str.upper, str.lower, str.title])(x) if isinstance(x, str) else x
    )

    # Duplicate users controlled by duplicate_rate
    n_dupes = int(num_users * duplicate_rate)
    if n_dupes > 0:
        dupe_indices = np.random.choice(num_users, size=n_dupes, replace=True)
        users_df = pd.concat([users_df, users_df.iloc[dupe_indices].copy()], ignore_index=True)

    # Purchases table
    purchases_data = {
        "purchase_id": list(range(1, num_purchases + 1)),
        "user_id": [messy_user_id(random.choice(user_ids_clean)) for _ in range(num_purchases)],
        "amount": [round(random.uniform(5.0, 500.0), 2) for _ in range(num_purchases)],
        "purchase_date": [fake.date_between(start_date="-2y", end_date="today") for _ in range(num_purchases)],
        "product": [fake.word().title() for _ in range(num_purchases)],
    }
    purchases_df = pd.DataFrame(purchases_data)

    purchases_df["amount"] = purchases_df["amount"].apply(lambda x: f"${x:,.2f}")
    purchases_df["purchase_date"] = purchases_df["purchase_date"].apply(
        lambda d: d.strftime(random.choice(date_formats)) if pd.notna(d) else None
    )

    criteria = {
        "user_id_is_integer": True,
        "tables_merged": True,
        "no_duplicates": True,
        "amount_is_numeric": True,
        "dates_standardized": True,
        "date_format": "%Y-%m-%d",
        "status_lowercase": True,
        "no_whitespace_in": ["name"],
        "only_active_users": True,
        "no_nulls_in": ["user_id", "name", "amount", "status"],
        "original_user_count": num_users,
        "original_purchase_count": num_purchases,
        "difficulty_params": {
            "null_percentage": null_percentage,
            "duplicate_rate": duplicate_rate,
            "id_format_variety": id_format_variety,
        },
    }
    return {"main": users_df, "purchases": purchases_df}, criteria


def generate_hard_dataset(
    seed: Optional[int] = None,
    num_docs: int = 30,
    pii_density: float = 0.3,
    pii_variety: float = 0.5,
) -> Tuple[pd.DataFrame, dict]:
    """
    Generate text documents with embedded PII for redaction task.

    Args:
        seed: Random seed
        num_docs: Number of documents
        pii_density: Fraction of docs that contain at least one PII item (0.0–1.0)
        pii_variety: Controls how many PII types appear per doc (0.0 = one type, 1.0 = all types)

    PII types:
        - Email addresses
        - Phone numbers (3 formats)
        - Credit card numbers
        - SSN-like numbers
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        fake = Faker()
        Faker.seed(seed)
    else:
        fake = Faker()

    topics = [
        "The latest developments in artificial intelligence are reshaping how businesses operate.",
        "Cloud computing infrastructure continues to evolve with new deployment paradigms.",
        "Machine learning models require careful validation before production deployment.",
        "Data privacy regulations are becoming stricter across different jurisdictions.",
        "Open source software communities drive innovation in developer tools.",
        "Quantum computing research has made significant progress in error correction.",
        "Cybersecurity threats are evolving with increasingly sophisticated attack vectors.",
        "The semiconductor industry faces ongoing supply chain challenges.",
        "Remote work technologies have transformed corporate communication strategies.",
        "Blockchain applications extend beyond cryptocurrency into supply chain management.",
    ]

    def random_email():
        return fake.email()

    def random_phone():
        fmt = random.choice([
            lambda: f"({random.randint(200, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
            lambda: f"{random.randint(200, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
            lambda: f"{random.randint(200, 999)}.{random.randint(100, 999)}.{random.randint(1000, 9999)}",
        ])
        return fmt()

    def random_cc():
        return f"{random.randint(4000, 4999)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"

    def random_ssn():
        return f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"

    # Base per-type probabilities scaled by pii_density and pii_variety
    # pii_density controls how likely any doc gets PII at all
    # pii_variety controls how many types per doc (higher = more types per doc)
    base_probs = {
        "email":       0.33,
        "phone":       0.27,
        "credit_card": 0.20,
        "ssn":         0.13,
    }
    # Scale all by density; variety boosts non-primary types
    def _prob(base, is_primary=True):
        variety_boost = 1.0 if is_primary else (0.3 + pii_variety * 0.7)
        return min(pii_density * (base / 0.33) * variety_boost, 1.0)

    documents = []
    pii_counts = {"email": 0, "phone": 0, "credit_card": 0, "ssn": 0}

    for i in range(num_docs):
        text = " ".join(random.sample(topics, min(random.randint(2, 3), len(topics))))

        inject_email = random.random() < _prob(base_probs["email"], is_primary=True)
        inject_phone = random.random() < _prob(base_probs["phone"], is_primary=False)
        inject_cc = random.random() < _prob(base_probs["credit_card"], is_primary=False)
        inject_ssn = random.random() < _prob(base_probs["ssn"], is_primary=False)

        if inject_email:
            text += f" For inquiries, contact {random_email()} for more information."
            pii_counts["email"] += 1

        if inject_phone:
            text += f" You can also reach us at {random_phone()} during business hours."
            pii_counts["phone"] += 1

        if inject_cc:
            text += f" Payment was processed with card number {random_cc()} on file."
            pii_counts["credit_card"] += 1

        if inject_ssn:
            text += f" The associated identification number is {random_ssn()} for records."
            pii_counts["ssn"] += 1

        documents.append({"id": i + 1, "text": text})

    df = pd.DataFrame(documents)

    criteria = {
        "pii_types_to_redact": ["email", "phone", "credit_card", "ssn"],
        "redaction_token": "[REDACTED]",
        "pii_patterns": {
            "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            "phone": r'[\(]?\d{3}[\)]?[\s\.\-]?\d{3}[\s\.\-]?\d{4}',
            "credit_card": r'\d{4}[\-\s]?\d{4}[\-\s]?\d{4}[\-\s]?\d{4}',
            "ssn": r'\d{3}\-\d{2}\-\d{4}',
        },
        "pii_counts": pii_counts,
        "total_docs": num_docs,
        "preserve_non_pii": True,
        "difficulty_params": {
            "pii_density": pii_density,
            "pii_variety": pii_variety,
        },
    }
    return df, criteria
