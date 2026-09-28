import json
import sqlite3
from pathlib import Path
from datetime import datetime


DB_PATH = Path(
    "/workspace/personal_cfo/personal_cfo_app/data/personal_cfo.db"
)

DB_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


def get_connection():
    conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False,
    )

    conn.row_factory = sqlite3.Row

    return conn


def initialize_database():
    conn = get_connection()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            transaction_id TEXT NOT NULL UNIQUE,

            date TEXT,
            posted_date TEXT,

            merchant TEXT,
            description TEXT,

            amount REAL,
            raw_amount REAL,

            source TEXT,
            source_category TEXT,

            transaction_type TEXT,

            address TEXT,
            reference_number TEXT,
            account TEXT,

            source_file TEXT,

            category TEXT,
            subcategory TEXT,
            financial_type TEXT,

            protected INTEGER,
            cut_priority TEXT,
            recurring INTEGER,
            reimbursable INTEGER,
            shared INTEGER,
            one_time_exceptional INTEGER,
            count_as_spending INTEGER,

            confidence TEXT,

            raw_classification TEXT,

            model_version TEXT,

            created_at TEXT,
            classified_at TEXT
        )
        """
    )

    conn.commit()
    conn.close()


def transaction_exists(
    transaction_id,
):
    conn = get_connection()

    row = conn.execute(
        """
        SELECT 1
        FROM transactions
        WHERE transaction_id = ?
        LIMIT 1
        """,
        (transaction_id,),
    ).fetchone()

    conn.close()

    return row is not None


def get_existing_transaction_ids(
    transaction_ids,
):
    """
    Efficiently check a batch of IDs.
    """

    if not transaction_ids:
        return set()

    conn = get_connection()

    placeholders = ",".join(
        ["?"] * len(transaction_ids)
    )

    rows = conn.execute(
        f"""
        SELECT transaction_id
        FROM transactions
        WHERE transaction_id IN ({placeholders})
        """,
        transaction_ids,
    ).fetchall()

    conn.close()

    return {
        row["transaction_id"]
        for row in rows
    }


def save_transaction(
    transaction,
    classification,
    model_version="llama-nemotron-8b-lora-v1",
):
    conn = get_connection()

    now = datetime.utcnow().isoformat()

    conn.execute(
        """
        INSERT OR IGNORE INTO transactions (

            transaction_id,

            date,
            posted_date,

            merchant,
            description,

            amount,
            raw_amount,

            source,
            source_category,

            transaction_type,

            address,
            reference_number,
            account,

            source_file,

            category,
            subcategory,
            financial_type,

            protected,
            cut_priority,
            recurring,
            reimbursable,
            shared,
            one_time_exceptional,
            count_as_spending,

            confidence,

            raw_classification,

            model_version,

            created_at,
            classified_at

        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,

        (
            transaction.get(
                "transaction_id"
            ),

            transaction.get("date"),

            transaction.get(
                "posted_date"
            ),

            transaction.get(
                "merchant"
            ),

            transaction.get(
                "description"
            ),

            transaction.get(
                "amount"
            ),

            transaction.get(
                "raw_amount"
            ),

            transaction.get(
                "source"
            ),

            transaction.get(
                "source_category"
            ),

            transaction.get(
                "transaction_type"
            ),

            transaction.get(
                "address"
            ),

            transaction.get(
                "reference_number"
            ),

            transaction.get(
                "account"
            ),

            transaction.get(
                "source_file"
            ),

            classification.get(
                "category"
            ),

            classification.get(
                "subcategory"
            ),

            classification.get(
                "financial_type"
            ),

            int(
                classification.get(
                    "protected",
                    False,
                )
            ),

            classification.get(
                "cut_priority"
            ),

            int(
                classification.get(
                    "recurring",
                    False,
                )
            ),

            int(
                classification.get(
                    "reimbursable",
                    False,
                )
            ),

            int(
                classification.get(
                    "shared",
                    False,
                )
            ),

            int(
                classification.get(
                    "one_time_exceptional",
                    False,
                )
            ),

            int(
                classification.get(
                    "count_as_spending",
                    True,
                )
            ),

            classification.get(
                "confidence"
            ),

            json.dumps(
                classification
            ),

            model_version,

            now,
            now,
        ),
    )

    conn.commit()
    conn.close()


def load_all_transactions():
    conn = get_connection()

    rows = conn.execute(
        """
        SELECT *
        FROM transactions
        ORDER BY date DESC, id DESC
        """
    ).fetchall()

    conn.close()

    results = []

    for row in rows:
        record = dict(row)

        record["protected"] = bool(
            record["protected"]
        )

        record["recurring"] = bool(
            record["recurring"]
        )

        record["reimbursable"] = bool(
            record["reimbursable"]
        )

        record["shared"] = bool(
            record["shared"]
        )

        record[
            "one_time_exceptional"
        ] = bool(
            record[
                "one_time_exceptional"
            ]
        )

        record[
            "count_as_spending"
        ] = bool(
            record[
                "count_as_spending"
            ]
        )

        record["classification"] = {
            "category":
                record["category"],

            "subcategory":
                record["subcategory"],

            "financial_type":
                record["financial_type"],

            "protected":
                record["protected"],

            "cut_priority":
                record["cut_priority"],

            "recurring":
                record["recurring"],

            "reimbursable":
                record["reimbursable"],

            "shared":
                record["shared"],

            "one_time_exceptional":
                record[
                    "one_time_exceptional"
                ],

            "count_as_spending":
                record[
                    "count_as_spending"
                ],

            "confidence":
                record["confidence"],
        }

        results.append(record)

    return results


def get_transaction_count():
    conn = get_connection()

    count = conn.execute(
        """
        SELECT COUNT(*)
        FROM transactions
        """
    ).fetchone()[0]

    conn.close()

    return count


if __name__ == "__main__":

    initialize_database()

    print(
        f"Database initialized at: "
        f"{DB_PATH}"
    )

    print(
        f"Stored transactions: "
        f"{get_transaction_count()}"
    )
