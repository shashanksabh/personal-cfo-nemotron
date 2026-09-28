import calendar
import json
import os
import re
from datetime import date, datetime

from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

LIGHTNING_MODEL = (
    "nvidia/nemotron-3.5-lightning-30b-a3b"
)

LIGHTNING_BASE_URL = (
    "https://integrate.api.nvidia.com/v1"
)


# ============================================================
# QUERY SCHEMA
# ============================================================

VALID_CATEGORIES = {
    "Housing",
    "Utilities",
    "Groceries",
    "Workday Food",
    "Dining",
    "Personal Travel",
    "Transportation",
    "Fitness",
    "Shopping",
    "Subscriptions",
    "Health / Medical",
    "Wedding",
    "Gifts",
    "Family Support",
    "Shared Expenses",
    "Entertainment",
    "Education / Professional",
    "Taxes / Government",
    "Banking / Fees",
    "Personal Care",
    "Income",
    "Reimbursement",
    "Savings",
    "Investments",
    "Transfer",
}


ALLOWED_OPERATIONS = {
    "sum",
    "count",
    "list",
    "average",
}


ALLOWED_GROUP_BY = {
    "category",
    "subcategory",
    "merchant",
    "month",
    "protected",
    "one_time_exceptional",
}


ALLOWED_ORDER = {
    "asc",
    "desc",
}


# ============================================================
# SEMANTIC ONTOLOGY
#
# These describe financial concepts, not merchant mappings.
# ============================================================

SEMANTIC_ALIASES = {

    "rideshare": {
        "category": "Transportation",
        "subcategory": "Rideshare",
    },

    "ride share": {
        "category": "Transportation",
        "subcategory": "Rideshare",
    },

    "groceries": {
        "category": "Groceries",
        "subcategory": None,
    },

    "grocery": {
        "category": "Groceries",
        "subcategory": None,
    },

    "dining": {
        "category": "Dining",
        "subcategory": None,
    },

    "restaurants": {
        "category": "Dining",
        "subcategory": None,
    },

    "restaurant": {
        "category": "Dining",
        "subcategory": None,
    },

    "workday food": {
        "category": "Workday Food",
        "subcategory": None,
    },

    "work cafeteria": {
        "category": "Workday Food",
        "subcategory": "Work Cafeteria",
    },

    "personal travel": {
        "category": "Personal Travel",
        "subcategory": None,
    },

    "flight": {
        "category": "Personal Travel",
        "subcategory": "Flight",
    },

    "flights": {
        "category": "Personal Travel",
        "subcategory": "Flight",
    },

    "hotel": {
        "category": "Personal Travel",
        "subcategory": "Hotel",
    },

    "hotels": {
        "category": "Personal Travel",
        "subcategory": "Hotel",
    },

    "fitness": {
        "category": "Fitness",
        "subcategory": None,
    },

    "subscriptions": {
        "category": "Subscriptions",
        "subcategory": None,
    },

    "subscription": {
        "category": "Subscriptions",
        "subcategory": None,
    },

    "utilities": {
        "category": "Utilities",
        "subcategory": None,
    },

    "wedding": {
        "category": "Wedding",
        "subcategory": None,
    },

    "shopping": {
        "category": "Shopping",
        "subcategory": None,
    },

    "personal care": {
        "category": "Personal Care",
        "subcategory": None,
    },
}


# ============================================================
# MERCHANT NORMALIZATION
#
# These only normalize user-entered merchant names to strings
# likely to appear in raw bank statements.
# ============================================================

MERCHANT_ALIASES = {

    "united airlines": "United",
    "united airline": "United",

    "whole foods": "Wholefds",
    "whole foods market": "Wholefds",

    "uber technologies": "Uber",
    "uber rides": "Uber",

    "lyft rides": "Lyft",

    "airbnb": "Airbnb",
}


# ============================================================
# PLANNER PROMPT
# ============================================================

QUERY_INTENT_SYSTEM_PROMPT = """
You are the query planner for a Personal CFO application.

Convert the user's natural-language financial question into
structured database query intent.

Do NOT answer the financial question.
Do NOT write SQL.

The transactions database contains:

date
merchant
description
amount
category
subcategory
protected
one_time_exceptional
count_as_spending
source

Return ONLY one JSON object with exactly these fields:

{
  "needs_database": true,
  "operation": "sum",
  "merchant_contains": null,
  "category": null,
  "subcategory": null,
  "start_date": null,
  "end_date": null,
  "spending_only": true,
  "protected": null,
  "one_time_exceptional": null,
  "group_by": null,
  "order": "desc",
  "limit": 20
}


ALLOWED OPERATIONS

sum
count
list
average


ALLOWED GROUP BY VALUES

category
subcategory
merchant
month
protected
one_time_exceptional
null


VALID CATEGORIES

Housing
Utilities
Groceries
Workday Food
Dining
Personal Travel
Transportation
Fitness
Shopping
Subscriptions
Health / Medical
Wedding
Gifts
Family Support
Shared Expenses
Entertainment
Education / Professional
Taxes / Government
Banking / Fees
Personal Care
Income
Reimbursement
Savings
Investments
Transfer


============================================================
GENERAL RULES
============================================================

Questions about actual spending, transactions, merchants,
categories, dates, totals, counts, averages, or grouped results
require:

needs_database = true


"How much did I spend?"

normally means:

operation = "sum"
spending_only = true


"How many transactions?"

normally means:

operation = "count"


"Show me..."

normally means:

operation = "list"


"What was my average..."

normally means:

operation = "average"


============================================================
GROUPING
============================================================

"What are my biggest spending categories?"

means:

operation = "sum"
group_by = "category"
spending_only = true
order = "desc"


"Which merchants did I spend the most at?"

means:

operation = "sum"
group_by = "merchant"
spending_only = true
order = "desc"


"How much did I spend by month?"

means:

operation = "sum"
group_by = "month"
spending_only = true


============================================================
PROTECTED SPENDING
============================================================

"How much of my spending is protected?"

means:

operation = "sum"
spending_only = true
protected = true


"How much of my spending is non-protected?"

means:

operation = "sum"
spending_only = true
protected = false


"How much of my spending is reducible?"

means:

operation = "sum"
spending_only = true
protected = false


============================================================
ONE-TIME EXCEPTIONAL SPENDING
============================================================

"How much was one-time exceptional spending?"

means:

operation = "sum"
spending_only = true
one_time_exceptional = true


============================================================
MERCHANTS VS FINANCIAL CONCEPTS
============================================================

A merchant is a named company, store, airline, restaurant,
brand, or service provider.

Examples:

United Airlines
Uber
Lyft
Whole Foods
Airbnb
Turkish Airlines
Costco


A category or subcategory is a financial concept.

Examples:

Rideshare
Groceries
Dining
Personal Travel
Fitness
Subscriptions


IMPORTANT:

"How much did I spend on United Airlines?"

means:

merchant_contains = "United Airlines"
category = null
subcategory = null


"How much did I spend on Rideshare?"

means:

merchant_contains = null
category = "Transportation"
subcategory = "Rideshare"


"How much did I spend on groceries?"

means:

merchant_contains = null
category = "Groceries"
subcategory = null


"How much did I spend on flights?"

means:

merchant_contains = null
category = "Personal Travel"
subcategory = "Flight"


"Show my last 5 Personal Travel transactions"

means:

operation = "list"
category = "Personal Travel"
limit = 5


============================================================
DATES
============================================================

Dates MUST be complete ISO dates:

YYYY-MM-DD

Never return malformed or partial dates such as:

208-01
2026-08
08-01

If the user names a month without a year,
use the current year supplied in the user message.

Example:

August 2026

start_date = "2026-08-01"
end_date = "2026-08-31"


============================================================
DEFAULTS
============================================================

Unless the question requires otherwise:

protected = null
one_time_exceptional = null
group_by = null
order = "desc"
limit = 20


Return JSON only.
"""


# ============================================================
# CFO REASONING PROMPT
# ============================================================

CFO_SYSTEM_PROMPT = """
You are a Personal CFO reasoning assistant.

You receive deterministic financial calculations and/or exact
SQLite query results.

Those supplied values are authoritative.


Rules:

1. Never invent transactions, merchants, categories, dates,
   totals, or amounts.

2. Never override an exact SQLite or Python result.

3. Preserve supplied grouped values and their ordering.

4. High spending does not automatically mean overspending.
   Overspending requires a budget, goal, historical baseline,
   or relevant preference.

5. protected=true means that spending should generally not
   be targeted for cuts.

6. protected=false means the transaction is not protected,
   but that alone does not mean the user should cut it.

7. one_time_exceptional=true means the transaction was marked
   as unusual or exceptional rather than routine.

8. Do not infer affordability without sufficient information
   about income, cash flow, savings, or balances.

9. If income is missing, null, or zero in supplied context,
   treat actual income as unknown unless explicitly known.

10. If results_may_be_truncated is true, do not describe the
    returned groups as the complete data set.

11. If results_may_be_truncated is true, use wording such as
    "among the returned groups" or
    "among the top returned groups."

12. Distinguish calculations from interpretation:
    SQLite/Python calculate facts.
    You explain those facts.

Be concise and useful.
"""


# ============================================================
# HELPERS
# ============================================================

def _coerce_bool(
    value,
    default=False,
):

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        str,
    ):

        normalized = (
            value.strip().lower()
        )

        if normalized == "true":
            return True

        if normalized == "false":
            return False

    if isinstance(
        value,
        (int, float),
    ):
        return bool(
            value
        )

    return default


def _coerce_optional_bool(
    value,
):

    if value is None:
        return None

    return _coerce_bool(
        value,
        default=False,
    )


def _validate_iso_date(
    value,
):

    if value is None:
        return None

    if not isinstance(
        value,
        str,
    ):
        return None

    value = value.strip()

    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}",
        value,
    ):
        return None

    try:

        parsed = datetime.strptime(
            value,
            "%Y-%m-%d",
        )

    except ValueError:

        return None

    return parsed.strftime(
        "%Y-%m-%d"
    )


def _extract_year(
    question,
):

    match = re.search(
        r"\b(20\d{2})\b",
        question,
    )

    if match:

        return int(
            match.group(1)
        )

    return None


def _extract_month(
    question,
):

    text = question.lower()

    for month_number in range(
        1,
        13,
    ):

        full_name = (
            calendar.month_name[
                month_number
            ].lower()
        )

        abbreviation = (
            calendar.month_abbr[
                month_number
            ].lower()
        )

        if re.search(
            rf"\b{re.escape(full_name)}\b",
            text,
        ):

            return month_number

        if (
            abbreviation
            and re.search(
                rf"\b{re.escape(abbreviation)}\b",
                text,
            )
        ):

            return month_number

    return None


def _month_range(
    year,
    month,
):

    last_day = calendar.monthrange(
        year,
        month,
    )[1]

    return (
        f"{year:04d}-{month:02d}-01",
        (
            f"{year:04d}-"
            f"{month:02d}-"
            f"{last_day:02d}"
        ),
    )


def _normalize_dates(
    intent,
    question,
):

    month = _extract_month(
        question
    )

    if month is not None:

        year = (
            _extract_year(
                question
            )
            or date.today().year
        )

        (
            start_date,
            end_date,
        ) = _month_range(
            year,
            month,
        )

        intent[
            "start_date"
        ] = start_date

        intent[
            "end_date"
        ] = end_date

        return intent


    intent[
        "start_date"
    ] = _validate_iso_date(
        intent.get(
            "start_date"
        )
    )

    intent[
        "end_date"
    ] = _validate_iso_date(
        intent.get(
            "end_date"
        )
    )

    return intent


def _normalize_merchant(
    merchant,
):

    if merchant is None:
        return None

    merchant = str(
        merchant
    ).strip()

    if not merchant:
        return None

    lower = merchant.lower()

    if lower in MERCHANT_ALIASES:

        return MERCHANT_ALIASES[
            lower
        ]

    airline_match = re.fullmatch(
        r"(.+?)\s+airlines?",
        merchant,
        flags=re.IGNORECASE,
    )

    if airline_match:

        merchant = (
            airline_match
            .group(1)
            .strip()
        )

    return merchant


def _apply_semantic_alias(
    intent,
    question,
):

    text = question.lower()

    aliases = sorted(
        SEMANTIC_ALIASES.items(),
        key=lambda pair: len(
            pair[0]
        ),
        reverse=True,
    )

    for phrase, mapping in aliases:

        pattern = (
            rf"\b{re.escape(phrase)}\b"
        )

        if re.search(
            pattern,
            text,
        ):

            intent[
                "category"
            ] = mapping.get(
                "category"
            )

            intent[
                "subcategory"
            ] = mapping.get(
                "subcategory"
            )

            intent[
                "merchant_contains"
            ] = None

            break

    return intent


def _apply_query_semantics(
    intent,
    question,
):

    q = question.lower()


    # --------------------------------------------------------
    # Protected / reducible
    # --------------------------------------------------------

    if (
        "reducible"
        in q
    ):

        intent[
            "needs_database"
        ] = True

        intent[
            "operation"
        ] = "sum"

        intent[
            "spending_only"
        ] = True

        intent[
            "protected"
        ] = False


    elif (
        "protected"
        in q
    ):

        intent[
            "needs_database"
        ] = True

        intent[
            "spending_only"
        ] = True

        if (
            "non-protected"
            in q
            or "non protected"
            in q
            or "unprotected"
            in q
        ):

            intent[
                "protected"
            ] = False

        else:

            intent[
                "protected"
            ] = True

        if (
            "how much"
            in q
            or "spending"
            in q
            or "spent"
            in q
        ):

            intent[
                "operation"
            ] = "sum"


    # --------------------------------------------------------
    # Exceptional spending
    # --------------------------------------------------------

    if (
        "one-time exceptional"
        in q
        or "one time exceptional"
        in q
        or "exceptional spending"
        in q
    ):

        intent[
            "needs_database"
        ] = True

        intent[
            "operation"
        ] = "sum"

        intent[
            "spending_only"
        ] = True

        intent[
            "one_time_exceptional"
        ] = True


    # --------------------------------------------------------
    # Category grouping
    # --------------------------------------------------------

    category_group_phrases = [
        "spending by category",
        "biggest spending categories",
        "biggest categories",
        "largest spending categories",
        "largest categories",
        "top spending categories",
        "top categories",
    ]

    if any(
        phrase in q
        for phrase
        in category_group_phrases
    ):

        intent[
            "needs_database"
        ] = True

        intent[
            "operation"
        ] = "sum"

        intent[
            "group_by"
        ] = "category"

        intent[
            "spending_only"
        ] = True

        intent[
            "order"
        ] = "desc"


    # --------------------------------------------------------
    # Merchant grouping
    # --------------------------------------------------------

    merchant_group_phrases = [
        "spending by merchant",
        "top merchants",
        "biggest merchants",
        "largest merchants",
        "spend the most at",
        "spent the most at",
    ]

    if any(
        phrase in q
        for phrase
        in merchant_group_phrases
    ):

        intent[
            "needs_database"
        ] = True

        intent[
            "operation"
        ] = "sum"

        intent[
            "group_by"
        ] = "merchant"

        intent[
            "spending_only"
        ] = True

        intent[
            "order"
        ] = "desc"


    # --------------------------------------------------------
    # Month grouping
    # --------------------------------------------------------

    if (
        "spending by month"
        in q
        or "spend by month"
        in q
    ):

        intent[
            "needs_database"
        ] = True

        intent[
            "operation"
        ] = "sum"

        intent[
            "group_by"
        ] = "month"

        intent[
            "spending_only"
        ] = True


    return intent


def _normalize_query_intent(
    raw_intent,
    question,
):

    if not isinstance(
        raw_intent,
        dict,
    ):

        raise ValueError(
            "Planner output was not "
            "a JSON object."
        )


    intent = {

        "needs_database":
            _coerce_bool(
                raw_intent.get(
                    "needs_database",
                    True,
                ),
                default=True,
            ),

        "operation":
            raw_intent.get(
                "operation",
                "sum",
            ),

        "merchant_contains":
            raw_intent.get(
                "merchant_contains"
            ),

        "category":
            raw_intent.get(
                "category"
            ),

        "subcategory":
            raw_intent.get(
                "subcategory"
            ),

        "start_date":
            raw_intent.get(
                "start_date"
            ),

        "end_date":
            raw_intent.get(
                "end_date"
            ),

        "spending_only":
            _coerce_bool(
                raw_intent.get(
                    "spending_only",
                    True,
                ),
                default=True,
            ),

        "protected":
            _coerce_optional_bool(
                raw_intent.get(
                    "protected"
                )
            ),

        "one_time_exceptional":
            _coerce_optional_bool(
                raw_intent.get(
                    "one_time_exceptional"
                )
            ),

        "group_by":
            raw_intent.get(
                "group_by"
            ),

        "order":
            raw_intent.get(
                "order",
                "desc",
            ),

        "limit":
            raw_intent.get(
                "limit",
                20,
            ),
    }


    # --------------------------------------------------------
    # Operation validation
    # --------------------------------------------------------

    if (
        intent[
            "operation"
        ]
        not in ALLOWED_OPERATIONS
    ):

        intent[
            "operation"
        ] = "sum"


    # --------------------------------------------------------
    # Group-by validation
    # --------------------------------------------------------

    if (
        intent[
            "group_by"
        ]
        not in ALLOWED_GROUP_BY
    ):

        intent[
            "group_by"
        ] = None


    # --------------------------------------------------------
    # Sort validation
    # --------------------------------------------------------

    order = str(
        intent.get(
            "order",
            "desc",
        )
    ).lower()

    if (
        order
        not in ALLOWED_ORDER
    ):

        order = "desc"

    intent[
        "order"
    ] = order


    # --------------------------------------------------------
    # Limit validation
    # --------------------------------------------------------

    try:

        limit = int(
            intent[
                "limit"
            ]
        )

    except (
        TypeError,
        ValueError,
    ):

        limit = 20

    intent[
        "limit"
    ] = max(
        1,
        min(
            limit,
            100,
        ),
    )


    # --------------------------------------------------------
    # Category validation
    # --------------------------------------------------------

    category = intent.get(
        "category"
    )

    if (
        category is not None
        and category
        not in VALID_CATEGORIES
    ):

        intent[
            "category"
        ] = None


    # --------------------------------------------------------
    # Semantic concepts override accidental merchant mapping
    # --------------------------------------------------------

    intent = _apply_semantic_alias(
        intent,
        question,
    )


    # --------------------------------------------------------
    # Deterministic query semantics
    # --------------------------------------------------------

    intent = _apply_query_semantics(
        intent,
        question,
    )


    # --------------------------------------------------------
    # Merchant cleanup
    # --------------------------------------------------------

    intent[
        "merchant_contains"
    ] = _normalize_merchant(
        intent.get(
            "merchant_contains"
        )
    )


    # --------------------------------------------------------
    # Date cleanup
    # --------------------------------------------------------

    intent = _normalize_dates(
        intent,
        question,
    )


    return intent


# ============================================================
# LIGHTNING CLIENT
# ============================================================

class LightningClient:

    def __init__(
        self,
    ):

        api_key = os.getenv(
            "NVIDIA_API_KEY"
        )

        if not api_key:

            raise RuntimeError(
                "NVIDIA_API_KEY is not set."
            )


        self.model = (
            LIGHTNING_MODEL
        )


        self.client = OpenAI(
            base_url=(
                LIGHTNING_BASE_URL
            ),
            api_key=api_key,
            timeout=30.0,
            max_retries=1,
        )


    # ========================================================
    # INTERNAL PLANNER CALL
    # ========================================================

    def _planner_request(
        self,
        question,
        max_tokens,
    ):

        today = date.today()

        planner_question = (
            f"Current date: "
            f"{today.isoformat()}\n"
            f"Current year: "
            f"{today.year}\n\n"
            f"User question: "
            f"{question}"
        )


        response = (
            self.client
            .chat
            .completions
            .create(

                model=self.model,

                messages=[
                    {
                        "role":
                            "system",

                        "content":
                            QUERY_INTENT_SYSTEM_PROMPT,
                    },
                    {
                        "role":
                            "user",

                        "content":
                            planner_question,
                    },
                ],

                temperature=0,

                max_tokens=(
                    max_tokens
                ),

                response_format={
                    "type":
                        "json_object"
                },

                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking":
                            False
                    }
                },
            )
        )


        choice = (
            response.choices[0]
        )


        content = (
            choice.message.content
        )


        if not content:

            raise RuntimeError(
                "Lightning returned an empty "
                "planner response."
            )


        return (
            content,
            choice.finish_reason,
        )


    # ========================================================
    # PUBLIC: QUERY PLANNER
    # ========================================================

    def parse_query_intent(
        self,
        question,
    ):

        first_error = None


        # ----------------------------------------------------
        # First attempt
        # ----------------------------------------------------

        try:

            (
                content,
                finish_reason,
            ) = self._planner_request(
                question=question,
                max_tokens=800,
            )


            if (
                finish_reason
                == "length"
            ):

                raise RuntimeError(
                    "Planner output was truncated."
                )


            raw_intent = json.loads(
                content
            )


            intent = (
                _normalize_query_intent(
                    raw_intent,
                    question,
                )
            )


            return {
                "success": True,
                "intent": intent,
                "model": self.model,
            }


        except Exception as exc:

            first_error = exc


        # ----------------------------------------------------
        # Retry once
        # ----------------------------------------------------

        try:

            (
                content,
                finish_reason,
            ) = self._planner_request(
                question=question,
                max_tokens=1600,
            )


            if (
                finish_reason
                == "length"
            ):

                raise RuntimeError(
                    "Planner retry was truncated."
                )


            raw_intent = json.loads(
                content
            )


            intent = (
                _normalize_query_intent(
                    raw_intent,
                    question,
                )
            )


            return {
                "success": True,
                "intent": intent,
                "model": self.model,
            }


        except Exception as exc:

            return {
                "success": False,
                "intent": None,
                "model": self.model,
                "error": (
                    "Lightning planner failed. "
                    f"First attempt: "
                    f"{first_error}. "
                    f"Retry: {exc}"
                ),
            }


    # ========================================================
    # PUBLIC: CFO REASONING
    # ========================================================

    def ask_cfo(
        self,
        financial_context,
        question,
        database_context=None,
    ):

        context = {

            "financial_summary": (
                financial_context
                or {}
            ),

            "database_context": (
                database_context
                or {}
            ),
        }


        user_message = (
            f"User question:\n"
            f"{question}\n\n"
            f"Authoritative context:\n"
            f"{json.dumps(
                context,
                indent=2,
                default=str
            )}"
        )


        try:

            response = (
                self.client
                .chat
                .completions
                .create(

                    model=self.model,

                    messages=[
                        {
                            "role":
                                "system",

                            "content":
                                CFO_SYSTEM_PROMPT,
                        },
                        {
                            "role":
                                "user",

                            "content":
                                user_message,
                        },
                    ],

                    temperature=0.2,

                    max_tokens=1200,

                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking":
                                False
                        }
                    },
                )
            )


            choice = (
                response.choices[0]
            )


            answer = (
                choice.message.content
            )


            if not answer:

                raise RuntimeError(
                    "Lightning returned an "
                    "empty CFO response."
                )


            return {
                "success": True,
                "answer": answer,
                "model": self.model,
                "finish_reason": (
                    choice.finish_reason
                ),
            }


        except Exception as exc:

            return {
                "success": False,
                "answer": None,
                "model": self.model,
                "error": str(
                    exc
                ),
            }
