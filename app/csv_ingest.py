import hashlib
from pathlib import Path

import pandas as pd


# ============================================================
# PAYMENT FILTERING
# ============================================================

PAYMENT_PATTERNS = [
    "CAPITAL ONE AUTOPAY PYMT",
    "CAPITAL ONE ONLINE PYMT",
    "CAPITAL ONE MOBILE PYMT",
    "PAYMENT FROM CHK",
    "PAYMENT FROM CHECKING",
    "PAYMENT RECEIVED",
    "AUTOPAY PAYMENT",
    "CREDIT CARD PAYMENT",
    "ONLINE PAYMENT",
    "MOBILE PAYMENT",
    "PAYMENT THANK YOU",
]


def is_credit_card_payment(text):
    """
    Return True only for known credit-card payment patterns.

    We deliberately do NOT filter every transaction containing
    the word PAYMENT because purchases such as ATT*BILL PAYMENT
    may be legitimate spending.
    """

    if text is None:
        return False

    normalized = str(text).strip().upper()

    return any(
        pattern in normalized
        for pattern in PAYMENT_PATTERNS
    )


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_string(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    value = str(value).strip()

    if not value:
        return None

    return value


def normalize_date(value):
    """
    Normalize dates from either:
      YYYY-MM-DD
      MM/DD/YYYY

    into:
      YYYY-MM-DD
    """

    if value is None:
        return None

    try:
        parsed = pd.to_datetime(
            value,
            errors="coerce",
        )
    except Exception:
        return None

    if pd.isna(parsed):
        return None

    return parsed.strftime(
        "%Y-%m-%d"
    )


def safe_float(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def stable_transaction_id(
    source,
    date,
    merchant,
    amount,
    reference=None,
    posted_date=None,
):
    """
    Produce a deterministic transaction ID.

    The same transaction imported again should receive
    the same ID and therefore be deduplicated.
    """

    parts = [
        clean_string(source) or "",
        clean_string(date) or "",
        clean_string(posted_date) or "",
        clean_string(merchant) or "",
        f"{float(amount):.2f}",
        clean_string(reference) or "",
    ]

    raw = "|".join(parts)

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


# ============================================================
# FILE TYPE DETECTION
# ============================================================

def detect_csv_type(dataframe):
    """
    Detect one of the two supported bank formats.
    """

    columns = {
        str(column).strip()
        for column in dataframe.columns
    }

    capital_one_required = {
        "Transaction Date",
        "Posted Date",
        "Card No.",
        "Description",
        "Debit",
        "Credit",
    }

    bofa_required = {
        "Posted Date",
        "Reference Number",
        "Payee",
        "Amount",
    }

    if capital_one_required.issubset(
        columns
    ):
        return "capital_one"

    if bofa_required.issubset(
        columns
    ):
        return "bank_of_america"

    raise ValueError(
        "Unsupported CSV format. "
        f"Columns found: {sorted(columns)}"
    )


# ============================================================
# CAPITAL ONE
# ============================================================

def load_capital_one_dataframe(dataframe):
    """
    Normalize Capital One CSV format:

    Transaction Date
    Posted Date
    Card No.
    Description
    Category
    Debit
    Credit
    """

    transactions = []

    for _, row in dataframe.iterrows():

        description = clean_string(
            row.get("Description")
        )

        if not description:
            continue

        # --------------------------------------------
        # Ignore actual credit-card payments
        # --------------------------------------------

        if is_credit_card_payment(
            description
        ):
            continue

        raw_category = clean_string(
            row.get("Category")
        )

        debit = safe_float(
            row.get("Debit")
        )

        credit = safe_float(
            row.get("Credit")
        )

        transaction_date = normalize_date(
            row.get("Transaction Date")
        )

        posted_date = normalize_date(
            row.get("Posted Date")
        )

        card_no = clean_string(
            row.get("Card No.")
        )

        # --------------------------------------------
        # Capital One charges appear in Debit.
        #
        # Credits/refunds appear in Credit.
        #
        # We retain credits because things such as
        # merchant refunds may be financially useful.
        # They are marked with transaction_kind so
        # downstream logic can understand them.
        # --------------------------------------------

        if debit is not None and debit != 0:

            amount = abs(
                debit
            )

            transaction_kind = (
                "debit"
            )

        elif credit is not None and credit != 0:

            amount = abs(
                credit
            )

            transaction_kind = (
                "credit"
            )

        else:
            continue

        transaction_id = (
            stable_transaction_id(
                source="Capital One",
                date=transaction_date,
                posted_date=posted_date,
                merchant=description,
                amount=amount,
                reference=card_no,
            )
        )

        transactions.append(
            {
                "transaction_id": (
                    transaction_id
                ),
                "date": (
                    transaction_date
                    or posted_date
                ),
                "posted_date": (
                    posted_date
                ),
                "merchant": (
                    description
                ),
                "description": (
                    description
                ),
                "amount": (
                    round(
                        amount,
                        2,
                    )
                ),
                "source": (
                    "Capital One"
                ),
                "raw_category": (
                    raw_category
                ),
                "reference_number": (
                    None
                ),
                "card_no": (
                    card_no
                ),
                "address": (
                    None
                ),
                "transaction_kind": (
                    transaction_kind
                ),
            }
        )

    return transactions


# ============================================================
# BANK OF AMERICA
# ============================================================

def load_bofa_dataframe(dataframe):
    """
    Normalize Bank of America CSV format:

    Posted Date
    Reference Number
    Payee
    Address
    Amount

    BofA convention in the supplied export:
      negative amount = purchase/debit
      positive amount = credit
    """

    transactions = []

    for _, row in dataframe.iterrows():

        payee = clean_string(
            row.get("Payee")
        )

        if not payee:
            continue

        if is_credit_card_payment(
            payee
        ):
            continue

        posted_date = normalize_date(
            row.get("Posted Date")
        )

        reference_number = clean_string(
            row.get("Reference Number")
        )

        address = clean_string(
            row.get("Address")
        )

        raw_amount = safe_float(
            row.get("Amount")
        )

        if raw_amount is None:
            continue

        if raw_amount == 0:
            continue

        # --------------------------------------------
        # BofA:
        # negative = spending
        # positive = credit
        #
        # Store a positive magnitude because the
        # Personal CFO model classifies semantic type
        # separately.
        # --------------------------------------------

        if raw_amount < 0:

            transaction_kind = (
                "debit"
            )

        else:

            transaction_kind = (
                "credit"
            )

        amount = abs(
            raw_amount
        )

        transaction_id = (
            stable_transaction_id(
                source="Bank of America",
                date=posted_date,
                posted_date=posted_date,
                merchant=payee,
                amount=amount,
                reference=reference_number,
            )
        )

        description_parts = [
            payee
        ]

        if address:
            description_parts.append(
                address
            )

        description = " | ".join(
            description_parts
        )

        transactions.append(
            {
                "transaction_id": (
                    transaction_id
                ),
                "date": (
                    posted_date
                ),
                "posted_date": (
                    posted_date
                ),
                "merchant": (
                    payee
                ),
                "description": (
                    description
                ),
                "amount": (
                    round(
                        amount,
                        2,
                    )
                ),
                "source": (
                    "Bank of America"
                ),
                "raw_category": (
                    None
                ),
                "reference_number": (
                    reference_number
                ),
                "card_no": (
                    None
                ),
                "address": (
                    address
                ),
                "transaction_kind": (
                    transaction_kind
                ),
            }
        )

    return transactions


# ============================================================
# SINGLE FILE LOADER
# ============================================================

def load_csv(file_path):
    """
    Load and normalize one supported bank CSV.
    """

    file_path = Path(
        file_path
    )

    dataframe = pd.read_csv(
        file_path
    )

    csv_type = detect_csv_type(
        dataframe
    )

    if csv_type == "capital_one":

        return (
            load_capital_one_dataframe(
                dataframe
            )
        )

    if csv_type == "bank_of_america":

        return (
            load_bofa_dataframe(
                dataframe
            )
        )

    raise ValueError(
        f"Unhandled CSV type: {csv_type}"
    )


# ============================================================
# MULTI FILE LOADER
# ============================================================

def load_multiple_csvs(file_paths):
    """
    Load multiple CSV files from either supported institution.

    Returns:
        list[dict]

    This is intentional: downstream code should receive
    the same Python structure regardless of bank.
    """

    all_transactions = []

    for file_path in file_paths:

        transactions = load_csv(
            file_path
        )

        all_transactions.extend(
            transactions
        )

    return all_transactions


# ============================================================
# MODEL INPUT
# ============================================================

def build_model_input(transaction):
    """
    Produce the text given to the local Personal CFO
    classification model.
    """

    date = transaction.get(
        "date"
    )

    merchant = transaction.get(
        "merchant"
    )

    description = transaction.get(
        "description"
    )

    amount = transaction.get(
        "amount"
    )

    source = transaction.get(
        "source"
    )

    raw_category = transaction.get(
        "raw_category"
    )

    transaction_kind = (
        transaction.get(
            "transaction_kind"
        )
    )

    address = transaction.get(
        "address"
    )

    lines = [
        f"Date: {date}",
        f"Merchant: {merchant}",
        f"Amount: ${float(amount):.2f}",
        f"Source: {source}",
    ]

    if description:
        lines.append(
            f"Description: {description}"
        )

    if raw_category:
        lines.append(
            f"Bank category: {raw_category}"
        )

    if transaction_kind:
        lines.append(
            f"Transaction kind: {transaction_kind}"
        )

    if address:
        lines.append(
            f"Address: {address}"
        )

    return "\n".join(
        lines
    )
