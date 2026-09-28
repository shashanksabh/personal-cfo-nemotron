import json
import os
import re

import requests


# ============================================================
# CONFIG
# ============================================================

VLLM_BASE_URL = os.getenv(
    "VLLM_BASE_URL",
    "http://localhost:8000",
).rstrip("/")


VLLM_MODEL = os.getenv(
    "VLLM_MODEL",
    "personal-cfo-v2",
)


# ============================================================
# PERSONAL CFO ONTOLOGY
# ============================================================

CATEGORY_NAMES = [
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
]


ALLOWED_CATEGORIES = set(
    CATEGORY_NAMES
)


# ============================================================
# CLASSIFICATION PROMPT
# ============================================================

CLASSIFICATION_SYSTEM_PROMPT = """
You are the transaction classification engine for Personal CFO.

Classify exactly one financial transaction.

Return exactly one valid JSON object and nothing else.

The JSON object must contain exactly these four model-generated fields:

{
  "category": string,
  "subcategory": string,
  "protected": boolean,
  "one_time_exceptional": boolean
}


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


IMPORTANT RULES

- Other is not a valid category.

- Work Travel is not a valid category.

- All airline/flight transactions are Personal Travel / Flight.

- Hotels and Airbnb/lodging are Personal Travel.

- Uber/Lyft/rideshare are Transportation / Rideshare.

- Uber Eats is Dining / Food Delivery.

- Workplace cafeteria and workplace vending purchases are
  Workday Food.

- A repeated purchase is NOT automatically a subscription.

- Only actual subscriptions or memberships are Subscriptions.

- ChatGPT, Claude, Perplexity, Gemini, Netflix, Hulu, Spotify,
  Microsoft 365, Zipcar memberships and similar recurring
  services are Subscriptions.

- Utilities, phone, internet, Vespaio rent, and yoga stay in
  their natural categories and are protected.

- Dining in New York City is Dining / Relationship Dining.

- Personal grooming/services are Personal Care.

- Legal fees are Taxes / Government / Legal Fees.

- protected=true ONLY for utilities/phone/internet,
  Vespaio rent, and yoga.

- Groceries are not protected.

- Health / Medical is not protected.

- Flights and hotels are not protected.

- Gifts are not protected.

- Dining is not protected.

- Shopping is not protected.

- Subscriptions are not protected.

- Wedding spending is not protected.

- Transportation is not protected.

- Workday Food is not protected.

- one_time_exceptional=true only when the transaction is
  meaningfully exceptional, unusual, or clearly one-time.

- Routine recurring transactions should normally have
  one_time_exceptional=false.

Choose a concise and useful subcategory.

Return valid JSON only.
"""


# ============================================================
# QUERY INTENT PROMPT
# ============================================================

QUERY_INTENT_SYSTEM_PROMPT = """
You convert natural-language Personal CFO questions into structured
database query intent.

You are NOT answering the user's financial question.

Your job is to decide:

1. whether SQLite transaction retrieval is required
2. what operation is required
3. whether results should be grouped
4. what filters should be applied
5. how results should be ordered

Return exactly one valid JSON object and nothing else.


DATABASE

Table:

transactions


Important transaction fields:

- date
- merchant
- description
- amount
- source
- category
- subcategory
- protected
- one_time_exceptional
- count_as_spending


VALID CATEGORY VALUES

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


SUPPORTED OPERATIONS

sum
count
list
average


SUPPORTED GROUP BY VALUES

category
subcategory
merchant
month
protected
one_time_exceptional


OUTPUT SCHEMA

{
  "needs_database": boolean,
  "operation": "sum" | "count" | "list" | "average",
  "group_by": "category" | "subcategory" | "merchant" | "month" | "protected" | "one_time_exceptional" | null,
  "merchant_contains": string | null,
  "category": string | null,
  "subcategory": string | null,
  "start_date": string | null,
  "end_date": string | null,
  "spending_only": boolean,
  "protected": boolean | null,
  "one_time_exceptional": boolean | null,
  "order": "asc" | "desc",
  "limit": integer
}


CORE RULES

1. Questions about actual transactions, spending, merchants,
   categories, dates, totals, counts, averages, or grouped spending
   require needs_database=true.

2. "How much did I spend..." normally means:
   operation="sum"
   spending_only=true

3. "How many transactions..." means:
   operation="count"

4. "Show me...", "which transactions...", and similar requests
   normally mean:
   operation="list"

5. "What was my average..." normally means:
   operation="average"

6. Questions asking for spending by category use:
   operation="sum"
   group_by="category"
   spending_only=true

7. Questions asking for top, biggest, or largest spending categories use:
   operation="sum"
   group_by="category"
   spending_only=true
   order="desc"

8. Questions asking for top or biggest merchants use:
   operation="sum"
   group_by="merchant"
   spending_only=true
   order="desc"

9. Questions asking for spending by month use:
   operation="sum"
   group_by="month"
   spending_only=true

10. Questions asking how much spending is protected use:
    operation="sum"
    spending_only=true
    protected=true

11. Questions asking about non-protected or reducible spending use:
    operation="sum"
    spending_only=true
    protected=false

12. Questions asking about one-time exceptional spending use:
    operation="sum"
    spending_only=true
    one_time_exceptional=true

13. If a valid category name appears explicitly in the question,
    place that exact value in "category".

14. If the user asks about a specific merchant,
    populate "merchant_contains".

15. Use YYYY-MM-DD for dates.

16. Do not invent filters that the user did not request.

17. Do not output SQL.

18. Default:
    group_by=null
    protected=null
    one_time_exceptional=null
    order="desc"
    limit=20


EXAMPLE 1

Question:
"How much did I spend on Personal Travel?"

Output:

{
  "needs_database": true,
  "operation": "sum",
  "group_by": null,
  "merchant_contains": null,
  "category": "Personal Travel",
  "subcategory": null,
  "start_date": null,
  "end_date": null,
  "spending_only": true,
  "protected": null,
  "one_time_exceptional": null,
  "order": "desc",
  "limit": 20
}


EXAMPLE 2

Question:
"How much of my spending is protected?"

Output:

{
  "needs_database": true,
  "operation": "sum",
  "group_by": null,
  "merchant_contains": null,
  "category": null,
  "subcategory": null,
  "start_date": null,
  "end_date": null,
  "spending_only": true,
  "protected": true,
  "one_time_exceptional": null,
  "order": "desc",
  "limit": 20
}


EXAMPLE 3

Question:
"What are my biggest spending categories?"

Output:

{
  "needs_database": true,
  "operation": "sum",
  "group_by": "category",
  "merchant_contains": null,
  "category": null,
  "subcategory": null,
  "start_date": null,
  "end_date": null,
  "spending_only": true,
  "protected": null,
  "one_time_exceptional": null,
  "order": "desc",
  "limit": 20
}


EXAMPLE 4

Question:
"Show me my subscriptions."

Output:

{
  "needs_database": true,
  "operation": "list",
  "group_by": null,
  "merchant_contains": null,
  "category": "Subscriptions",
  "subcategory": null,
  "start_date": null,
  "end_date": null,
  "spending_only": true,
  "protected": null,
  "one_time_exceptional": null,
  "order": "desc",
  "limit": 20
}


EXAMPLE 5

Question:
"Which merchants did I spend the most at?"

Output:

{
  "needs_database": true,
  "operation": "sum",
  "group_by": "merchant",
  "merchant_contains": null,
  "category": null,
  "subcategory": null,
  "start_date": null,
  "end_date": null,
  "spending_only": true,
  "protected": null,
  "one_time_exceptional": null,
  "order": "desc",
  "limit": 20
}


Return valid JSON only.
"""


# ============================================================
# CFO RESPONSE PROMPT
# ============================================================

CFO_SYSTEM_PROMPT = """
You are a Personal CFO assistant.

You may receive:

1. a deterministic financial summary calculated by Python
2. an exact database retrieval result produced by SQLite
3. the user's original question

Treat Python calculations and SQLite results as the source of truth.

Never invent financial facts.


DATABASE RESULTS

Database retrieval may contain:

- aggregate results
- transaction rows
- grouped results


AGGREGATE RESULTS

If:

"result_type": "aggregate"

then the "result" field contains the exact aggregate returned by SQLite.

If operation="sum", use that exact amount.

If operation="count", use that exact count.

If operation="average", use that exact average.

Do not reconstruct arithmetic that SQLite has already performed.


ROW RESULTS

If:

"result_type": "rows"

then the "rows" field contains exact transactions returned by SQLite.

Do not invent transactions.

If rows is empty, no matching transactions were found.


GROUPED RESULTS

If:

"result_type": "grouped"

then the rows contain exact grouped database facts.

Use those grouped facts for analysis.

Do not change their numbers.


PROTECTED SPENDING

Protected spending represents expenses that should generally not
be targeted for cuts.


ONE-TIME EXCEPTIONAL SPENDING

One-time exceptional spending represents unusual or exceptional
transactions rather than routine spending.


INCOME

If income is:

- 0
- null
- missing
- unavailable

treat actual income as UNKNOWN.

Do not say the user earns zero.


AFFORDABILITY

Do not claim the user can or cannot afford a purchase unless
sufficient information exists, including relevant income,
cash flow, liquid savings, or balances.


MISSING DATA

Do not fill missing information with assumptions.

State what is missing when necessary.


FINANCIAL CALCULATIONS

Use supplied totals exactly.

Do not invent:

- totals
- income
- savings
- balances
- cash flow
- transactions

Be concise and useful.
"""


# ============================================================
# MODEL CLIENT
# ============================================================

class PersonalCFOModel:

    def __init__(
        self,
        base_url=None,
        model_name=None,
        timeout=120,
    ):

        self.base_url = (
            base_url
            or VLLM_BASE_URL
        ).rstrip("/")

        self.model_name = (
            model_name
            or VLLM_MODEL
        )

        self.timeout = timeout

        print(
            f"Using vLLM server: "
            f"{self.base_url}"
        )

        print(
            f"Using model: "
            f"{self.model_name}"
        )

        self._check_server()

        print(
            "Personal CFO vLLM client ready."
        )


    # ========================================================
    # SERVER CHECK
    # ========================================================

    def _check_server(
        self,
    ):

        response = requests.get(
            f"{self.base_url}/v1/models",
            timeout=self.timeout,
        )

        response.raise_for_status()

        payload = response.json()

        available_models = {
            item.get(
                "id"
            )
            for item in payload.get(
                "data",
                [],
            )
        }

        if (
            self.model_name
            not in available_models
        ):

            raise RuntimeError(
                "Requested model "
                f"'{self.model_name}' "
                "is not available from vLLM. "
                "Available models: "
                f"{sorted(available_models)}"
            )


    # ========================================================
    # GENERATION
    # ========================================================

    def _generate(
        self,
        messages,
        max_new_tokens=300,
    ):

        response = requests.post(
            (
                f"{self.base_url}"
                "/v1/chat/completions"
            ),
            json={
                "model":
                    self.model_name,

                "messages":
                    messages,

                "temperature":
                    0,

                "max_tokens":
                    max_new_tokens,
            },
            timeout=self.timeout,
        )

        response.raise_for_status()

        payload = response.json()

        choices = payload.get(
            "choices",
            [],
        )

        if not choices:

            raise RuntimeError(
                "vLLM returned no choices."
            )

        content = (
            choices[0]
            .get(
                "message",
                {},
            )
            .get(
                "content"
            )
        )

        if content is None:

            raise RuntimeError(
                "vLLM returned no message content."
            )

        return str(
            content
        ).strip()


    # ========================================================
    # JSON EXTRACTION
    # ========================================================

    def _extract_json(
        self,
        text,
    ):

        if not isinstance(
            text,
            str,
        ):

            raise ValueError(
                "Model response is not text."
            )

        cleaned = text.strip()

        # Strip common fenced-code wrappers if present.
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        first = cleaned.find(
            "{"
        )

        last = cleaned.rfind(
            "}"
        )

        if (
            first == -1
            or last == -1
            or last < first
        ):

            raise ValueError(
                "Model response did not contain "
                "a JSON object."
            )

        json_text = cleaned[
            first:last + 1
        ]

        result = json.loads(
            json_text
        )

        if not isinstance(
            result,
            dict,
        ):

            raise ValueError(
                "Expected a JSON object."
            )

        return result


    # ========================================================
    # BOOLEAN NORMALIZATION
    # ========================================================

    def _coerce_bool(
        self,
        value,
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

        return False


    # ========================================================
    # QUERY INTENT
    # ========================================================

    def parse_query_intent(
        self,
        question,
    ):

        messages = [
            {
                "role": "system",
                "content":
                    QUERY_INTENT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content":
                    question,
            },
        ]

        raw_response = self._generate(
            messages,
            max_new_tokens=350,
        )

        try:

            intent = self._extract_json(
                raw_response
            )

        except Exception as error:

            return {
                "success": False,
                "error": str(
                    error
                ),
                "raw_response":
                    raw_response,
            }


        # ----------------------------------------------------
        # DEFAULT V2 QUERY SCHEMA
        # ----------------------------------------------------

        defaults = {
            "needs_database": False,
            "operation": "list",
            "group_by": None,
            "merchant_contains": None,
            "category": None,
            "subcategory": None,
            "start_date": None,
            "end_date": None,
            "spending_only": False,
            "protected": None,
            "one_time_exceptional": None,
            "order": "desc",
            "limit": 20,
        }

        for key, value in defaults.items():

            if key not in intent:

                intent[
                    key
                ] = value


        question_lower = (
            question.lower()
        )

        question_upper = (
            question.upper()
        )


        # ----------------------------------------------------
        # EXPLICIT CATEGORY GROUNDING
        # ----------------------------------------------------

        for category_name in CATEGORY_NAMES:

            if (
                category_name.upper()
                in question_upper
            ):

                intent[
                    "category"
                ] = category_name

                break


        # ----------------------------------------------------
        # QUERY-SEMANTIC GROUNDING
        #
        # These rules only interpret the user's query.
        # They do not classify transactions.
        # ----------------------------------------------------

        if "protected" in question_lower:

            intent[
                "needs_database"
            ] = True

            intent[
                "spending_only"
            ] = True

            if (
                "non-protected"
                in question_lower
                or "non protected"
                in question_lower
                or "unprotected"
                in question_lower
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
                in question_lower
                or "spending"
                in question_lower
                or "spent"
                in question_lower
            ):

                intent[
                    "operation"
                ] = "sum"


        if (
            "reducible spending"
            in question_lower
            or "reducible"
            in question_lower
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


        if (
            "one-time exceptional"
            in question_lower
            or "one time exceptional"
            in question_lower
            or "exceptional spending"
            in question_lower
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
            phrase in question_lower
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


        merchant_group_phrases = [
            "spending by merchant",
            "top merchants",
            "biggest merchants",
            "largest merchants",
            "spend the most at",
            "spent the most at",
        ]

        if any(
            phrase in question_lower
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


        if (
            "spending by month"
            in question_lower
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


        # ----------------------------------------------------
        # OPERATION VALIDATION
        # ----------------------------------------------------

        if intent.get(
            "operation"
        ) not in {
            "sum",
            "count",
            "list",
            "average",
        }:

            intent[
                "operation"
            ] = "list"


        # ----------------------------------------------------
        # GROUP BY VALIDATION
        # ----------------------------------------------------

        if intent.get(
            "group_by"
        ) not in {
            None,
            "category",
            "subcategory",
            "merchant",
            "month",
            "protected",
            "one_time_exceptional",
        }:

            intent[
                "group_by"
            ] = None


        # ----------------------------------------------------
        # ORDER VALIDATION
        # ----------------------------------------------------

        order = str(
            intent.get(
                "order",
                "desc",
            )
        ).lower()

        if order not in {
            "asc",
            "desc",
        }:

            order = "desc"

        intent[
            "order"
        ] = order


        # ----------------------------------------------------
        # CATEGORY VALIDATION
        # ----------------------------------------------------

        category = intent.get(
            "category"
        )

        if (
            category is not None
            and category
            not in ALLOWED_CATEGORIES
        ):

            intent[
                "category"
            ] = None


        # ----------------------------------------------------
        # LIMIT VALIDATION
        # ----------------------------------------------------

        try:

            intent[
                "limit"
            ] = int(
                intent.get(
                    "limit",
                    20,
                )
            )

        except Exception:

            intent[
                "limit"
            ] = 20

        intent[
            "limit"
        ] = max(
            1,
            min(
                intent[
                    "limit"
                ],
                100,
            ),
        )


        # ----------------------------------------------------
        # REQUIRED BOOLEAN NORMALIZATION
        # ----------------------------------------------------

        intent[
            "needs_database"
        ] = self._coerce_bool(
            intent.get(
                "needs_database",
                False,
            )
        )

        intent[
            "spending_only"
        ] = self._coerce_bool(
            intent.get(
                "spending_only",
                False,
            )
        )


        # ----------------------------------------------------
        # OPTIONAL BOOLEAN NORMALIZATION
        #
        # None must stay None.
        # ----------------------------------------------------

        for field in [
            "protected",
            "one_time_exceptional",
        ]:

            value = intent.get(
                field
            )

            if value is None:

                intent[
                    field
                ] = None

            else:

                intent[
                    field
                ] = self._coerce_bool(
                    value
                )


        return {
            "success": True,
            "intent": intent,
            "raw_response":
                raw_response,
        }


    # ========================================================
    # V2 CLASSIFICATION VALIDATION
    # ========================================================

    def _normalize_v2_classification(
        self,
        result,
    ):

        if not isinstance(
            result,
            dict,
        ):

            raise ValueError(
                "Classification result must "
                "be a JSON object."
            )

        category = str(
            result.get(
                "category",
                "",
            )
        ).strip()

        subcategory = str(
            result.get(
                "subcategory",
                "",
            )
        ).strip()

        if not category:

            raise ValueError(
                "Classification returned "
                "an empty category."
            )

        if (
            category
            not in ALLOWED_CATEGORIES
        ):

            raise ValueError(
                "Model returned unsupported "
                f"category '{category}'."
            )

        if not subcategory:

            raise ValueError(
                "Classification returned "
                "an empty subcategory."
            )

        protected = self._coerce_bool(
            result.get(
                "protected",
                False,
            )
        )

        one_time_exceptional = (
            self._coerce_bool(
                result.get(
                    "one_time_exceptional",
                    False,
                )
            )
        )

        return {
            "category":
                category,

            "subcategory":
                subcategory,

            "protected":
                protected,

            "one_time_exceptional":
                one_time_exceptional,
        }


    # ========================================================
    # LEGACY COMPATIBILITY
    #
    # The model predicts only four fields.
    # These fields temporarily keep the current DB/dashboard
    # schema compatible while the rest of the app is migrated.
    # ========================================================

    def _add_compatibility_fields(
        self,
        classification,
    ):

        result = dict(
            classification
        )

        category = result[
            "category"
        ]

        protected = result[
            "protected"
        ]

        non_spending_categories = {
            "Income",
            "Reimbursement",
            "Savings",
            "Investments",
            "Transfer",
        }

        result[
            "count_as_spending"
        ] = (
            category
            not in non_spending_categories
        )


        if category == "Income":

            financial_type = (
                "Income"
            )

        elif category == "Reimbursement":

            financial_type = (
                "Income"
            )

        elif category == "Savings":

            financial_type = (
                "Savings"
            )

        elif category == "Investments":

            financial_type = (
                "Investment"
            )

        elif category == "Transfer":

            financial_type = (
                "Transfer"
            )

        elif category == "Shared Expenses":

            financial_type = (
                "Shared"
            )

        elif category in {
            "Housing",
            "Utilities",
            "Groceries",
            "Health / Medical",
        }:

            financial_type = (
                "Essential"
            )

        else:

            financial_type = (
                "Discretionary"
            )


        result[
            "financial_type"
        ] = financial_type


        result[
            "cut_priority"
        ] = (
            "very_low"
            if protected
            else "medium"
        )


        # Temporary compatibility behavior.
        # Long-term recurrence should come from transaction
        # history rather than model inference.

        result[
            "recurring"
        ] = (
            category
            == "Subscriptions"
        )


        result[
            "reimbursable"
        ] = False


        result[
            "shared"
        ] = (
            category
            == "Shared Expenses"
        )


        result[
            "confidence"
        ] = "medium"


        return result


    # ========================================================
    # TRANSACTION CLASSIFICATION
    # ========================================================

    def classify_transaction(
        self,
        transaction_text,
    ):

        messages = [
            {
                "role": "system",
                "content":
                    CLASSIFICATION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content":
                    transaction_text,
            },
        ]

        try:

            raw_response = self._generate(
                messages,
                max_new_tokens=180,
            )

            parsed = self._extract_json(
                raw_response
            )

            classification = (
                self._normalize_v2_classification(
                    parsed
                )
            )

            classification = (
                self._add_compatibility_fields(
                    classification
                )
            )

            return {
                "success": True,
                "classification":
                    classification,
                "raw_response":
                    raw_response,
            }

        except Exception as error:

            return {
                "success": False,
                "error": str(
                    error
                ),
                "raw_response":
                    (
                        raw_response
                        if "raw_response"
                        in locals()
                        else None
                    ),
            }


    # ========================================================
    # CFO REASONING
    # ========================================================

    def ask_cfo(
        self,
        financial_context,
        question,
        database_context=None,
    ):

        payload = {
            "financial_context":
                financial_context,

            "database_context":
                database_context,

            "question":
                question,
        }

        messages = [
            {
                "role": "system",
                "content":
                    CFO_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content":
                    json.dumps(
                        payload,
                        indent=2,
                        default=str,
                    ),
            },
        ]

        return self._generate(
            messages,
            max_new_tokens=600,
        )
