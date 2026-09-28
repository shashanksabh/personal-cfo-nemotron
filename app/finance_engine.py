from collections import defaultdict
from datetime import datetime


# ============================================================
# HELPERS
# ============================================================

def _safe_float(value, default=0.0):
    """
    Convert a value to float safely.
    """

    if value is None:
        return default

    try:
        return float(value)

    except (TypeError, ValueError):
        return default


def _safe_bool(value, default=False):
    """
    Normalize common boolean representations.

    SQLite may return:
        0 / 1

    Model/application data may contain:
        False / True
        "false" / "true"
    """

    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):

        normalized = value.strip().lower()

        if normalized in {
            "true",
            "1",
            "yes",
            "y",
        }:
            return True

        if normalized in {
            "false",
            "0",
            "no",
            "n",
            "",
        }:
            return False

    return default


def _safe_string(
    value,
    default=None,
):
    """
    Normalize string-like values while protecting against None.
    """

    if value is None:
        return default

    value = str(value).strip()

    if not value:
        return default

    return value


def _parse_month(date_value):
    """
    Convert YYYY-MM-DD into YYYY-MM.

    Returns None if the date cannot be parsed.
    """

    if date_value is None:
        return None

    if isinstance(
        date_value,
        datetime,
    ):
        return date_value.strftime(
            "%Y-%m"
        )

    date_string = str(
        date_value
    ).strip()

    if not date_string:
        return None

    try:

        parsed = datetime.strptime(
            date_string[:10],
            "%Y-%m-%d",
        )

        return parsed.strftime(
            "%Y-%m"
        )

    except (
        TypeError,
        ValueError,
    ):
        return None


def _get_classification(transaction):
    """
    finance_engine historically expects classification metadata
    nested under transaction["classification"].

    app.py rebuilds that nested structure from SQLite.

    This helper also tolerates flat transaction dictionaries.
    """

    classification = transaction.get(
        "classification"
    )

    if isinstance(
        classification,
        dict,
    ):
        return classification

    # Defensive fallback for flattened SQLite-style records.
    return {
        "category": transaction.get(
            "category"
        ),
        "subcategory": transaction.get(
            "subcategory"
        ),
        "financial_type": transaction.get(
            "financial_type"
        ),
        "protected": transaction.get(
            "protected"
        ),
        "cut_priority": transaction.get(
            "cut_priority"
        ),
        "recurring": transaction.get(
            "recurring"
        ),
        "reimbursable": transaction.get(
            "reimbursable"
        ),
        "shared": transaction.get(
            "shared"
        ),
        "one_time_exceptional": transaction.get(
            "one_time_exceptional"
        ),
        "count_as_spending": transaction.get(
            "count_as_spending"
        ),
        "confidence": transaction.get(
            "confidence"
        ),
    }


# ============================================================
# MAIN FINANCIAL SUMMARY
# ============================================================

def calculate_financial_summary(
    transactions,
):
    """
    Calculate deterministic financial metrics.

    The LLM is responsible for semantic classification.

    This function is responsible for:
        - arithmetic
        - aggregation
        - monthly totals
        - category totals
        - protected/discretionary totals
        - reducible spending
        - reimbursement totals
        - recurring totals
        - exceptional totals
        - savings/investments/income
        - savings rate

    No financial arithmetic is delegated to an LLM.
    """

    if not transactions:
        return {
            "total_spending": 0.0,
            "protected_spending": 0.0,
            "discretionary_spending": 0.0,
            "reducible_spending": 0.0,
            "recurring_spending": 0.0,
            "reimbursable_spending": 0.0,
            "exceptional_spending": 0.0,
            "income": 0.0,
            "savings": 0.0,
            "investments": 0.0,
            "savings_rate": None,
            "spending_by_category": {},
            "spending_by_month": {},
        }

    # --------------------------------------------------------
    # TOTALS
    # --------------------------------------------------------

    total_spending = 0.0
    protected_spending = 0.0
    discretionary_spending = 0.0
    reducible_spending = 0.0

    recurring_spending = 0.0
    reimbursable_spending = 0.0
    exceptional_spending = 0.0

    income = 0.0
    savings = 0.0
    investments = 0.0

    spending_by_category = defaultdict(
        float
    )

    spending_by_month = defaultdict(
        float
    )

    # --------------------------------------------------------
    # PROCESS TRANSACTIONS
    # --------------------------------------------------------

    for transaction in transactions:

        if not isinstance(
            transaction,
            dict,
        ):
            continue

        amount = _safe_float(
            transaction.get(
                "amount"
            )
        )

        if amount < 0:
            amount = abs(amount)

        classification = (
            _get_classification(
                transaction
            )
        )

        # ====================================================
        # IMPORTANT DEFENSIVE NORMALIZATION
        # ====================================================
        #
        # Previously category=None could become a dictionary
        # key and later crash:
        #
        # sorted(spending_by_category.items())
        #
        # because Python cannot compare None and str.
        #
        # Missing categories now safely become Uncategorized.
        # ====================================================

        category = _safe_string(
            classification.get(
                "category"
            ),
            default="Uncategorized",
        )

        financial_type = _safe_string(
            classification.get(
                "financial_type"
            ),
            default="Unknown",
        )

        cut_priority = _safe_string(
            classification.get(
                "cut_priority"
            ),
            default="medium",
        )

        protected = _safe_bool(
            classification.get(
                "protected"
            )
        )

        recurring = _safe_bool(
            classification.get(
                "recurring"
            )
        )

        reimbursable = _safe_bool(
            classification.get(
                "reimbursable"
            )
        )

        one_time_exceptional = (
            _safe_bool(
                classification.get(
                    "one_time_exceptional"
                )
            )
        )

        count_as_spending = (
            _safe_bool(
                classification.get(
                    "count_as_spending"
                )
            )
        )

        # ----------------------------------------------------
        # NON-SPENDING FINANCIAL FLOWS
        # ----------------------------------------------------

        if financial_type == "Income":

            income += amount

        elif financial_type == "Savings":

            savings += amount

        elif financial_type == "Investment":

            investments += amount

        # ----------------------------------------------------
        # ORDINARY SPENDING
        # ----------------------------------------------------

        if not count_as_spending:
            continue

        total_spending += amount

        # Category aggregation
        spending_by_category[
            category
        ] += amount

        # Monthly aggregation
        month = _parse_month(
            transaction.get(
                "date"
            )
        )

        if month is not None:

            spending_by_month[
                month
            ] += amount

        # ----------------------------------------------------
        # PROTECTED / DISCRETIONARY
        # ----------------------------------------------------

        if protected:

            protected_spending += (
                amount
            )

        else:

            discretionary_spending += (
                amount
            )

        # ----------------------------------------------------
        # REDUCIBLE SPENDING
        #
        # Existing Personal CFO definition:
        #
        # spending is considered reducible when:
        #   - it is not protected
        #   - cut priority is high or very_high
        # ----------------------------------------------------

        if (
            not protected
            and cut_priority
            in {
                "high",
                "very_high",
            }
        ):

            reducible_spending += (
                amount
            )

        # ----------------------------------------------------
        # OTHER ATTRIBUTES
        # ----------------------------------------------------

        if recurring:

            recurring_spending += (
                amount
            )

        if reimbursable:

            reimbursable_spending += (
                amount
            )

        if one_time_exceptional:

            exceptional_spending += (
                amount
            )

    # --------------------------------------------------------
    # SAVINGS RATE
    # --------------------------------------------------------

    if income > 0:

        savings_rate = (
            savings + investments
        ) / income

    else:

        savings_rate = None

    # --------------------------------------------------------
    # RETURN SUMMARY
    # --------------------------------------------------------

    return {

        "total_spending": round(
            total_spending,
            2,
        ),

        "protected_spending": round(
            protected_spending,
            2,
        ),

        "discretionary_spending": round(
            discretionary_spending,
            2,
        ),

        "reducible_spending": round(
            reducible_spending,
            2,
        ),

        "recurring_spending": round(
            recurring_spending,
            2,
        ),

        "reimbursable_spending": round(
            reimbursable_spending,
            2,
        ),

        "exceptional_spending": round(
            exceptional_spending,
            2,
        ),

        "income": round(
            income,
            2,
        ),

        "savings": round(
            savings,
            2,
        ),

        "investments": round(
            investments,
            2,
        ),

        "savings_rate": (
            round(
                savings_rate,
                4,
            )
            if savings_rate
            is not None
            else None
        ),

        # ====================================================
        # IMPORTANT SAFE SORT
        # ====================================================
        #
        # Even if bad historical data somehow contains a
        # non-string key, str(item[0]) guarantees that sorting
        # cannot crash the entire Streamlit application.
        # ====================================================

        "spending_by_category": {
            k: round(
                v,
                2,
            )
            for k, v
            in sorted(
                spending_by_category.items(),
                key=lambda item: str(
                    item[0]
                ),
            )
        },

        "spending_by_month": {
            k: round(
                v,
                2,
            )
            for k, v
            in sorted(
                spending_by_month.items(),
                key=lambda item: str(
                    item[0]
                ),
            )
        },
    }
