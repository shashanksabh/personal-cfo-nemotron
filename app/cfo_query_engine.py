from lightning_client import LightningClient
from sql_query_engine import execute_query_intent


# ============================================================
# QUESTION TYPES
# ============================================================

FACTUAL_GROUPED_PHRASES = [
    "which merchants",
    "which merchant",
    "biggest merchants",
    "top merchants",
    "most at",
    "largest categories",
    "biggest categories",
    "biggest spending categories",
    "largest spending categories",
    "top spending categories",
    "top categories",
    "spending by category",
    "spending by subcategory",
    "spending by merchant",
    "spending by month",
    "which month",
]


ANALYTICAL_GROUPED_PHRASES = [
    "overspending",
    "over spending",
    "what should i cut",
    "where should i cut",
    "what can i cut",
    "where can i save",
    "how can i save",
    "patterns",
    "analyze",
    "analyse",
    "why",
]


# ============================================================
# DETERMINISTIC FORMATTING
# ============================================================

def _build_subject(intent):

    merchant = intent.get(
        "merchant_contains"
    )

    category = intent.get(
        "category"
    )

    subcategory = intent.get(
        "subcategory"
    )

    protected = intent.get(
        "protected"
    )

    exceptional = intent.get(
        "one_time_exceptional"
    )

    if merchant:
        return merchant

    if subcategory:
        return subcategory

    if category:
        return category

    if protected is True:
        return "protected spending"

    if protected is False:
        return "non-protected spending"

    if exceptional is True:
        return "one-time exceptional spending"

    if exceptional is False:
        return "non-exceptional spending"

    return None


def _build_date_text(intent):

    start_date = intent.get(
        "start_date"
    )

    end_date = intent.get(
        "end_date"
    )

    if start_date and end_date:
        return (
            f" from {start_date} "
            f"through {end_date}"
        )

    if start_date:
        return (
            f" since {start_date}"
        )

    if end_date:
        return (
            f" through {end_date}"
        )

    return ""


def _format_aggregate_answer(
    question,
    intent,
    database_context,
):

    operation = database_context.get(
        "operation"
    )

    result = database_context.get(
        "result"
    )

    subject = _build_subject(
        intent
    )

    date_text = _build_date_text(
        intent
    )

    if operation == "sum":

        amount = (
            float(result)
            if result is not None
            else 0.0
        )

        if subject:
            return (
                f"Your {subject}{date_text} "
                f"was ${amount:,.2f}."
            )

        return (
            f"You spent ${amount:,.2f}"
            f"{date_text}."
        )

    if operation == "count":

        count = (
            int(result)
            if result is not None
            else 0
        )

        if subject:
            return (
                f"You had {count} "
                f"{subject} transactions"
                f"{date_text}."
            )

        return (
            f"You had {count} matching "
            f"transactions{date_text}."
        )

    if operation == "average":

        amount = (
            float(result)
            if result is not None
            else 0.0
        )

        if subject:
            return (
                f"Your average {subject} "
                f"transaction{date_text} "
                f"was ${amount:,.2f}."
            )

        return (
            f"Your average matching "
            f"transaction{date_text} "
            f"was ${amount:,.2f}."
        )

    return str(
        result
    )


def _format_grouped_answer(
    database_context,
):

    rows = database_context.get(
        "rows",
        [],
    )

    group_by = database_context.get(
        "group_by"
    )

    operation = database_context.get(
        "operation"
    )

    truncated = database_context.get(
        "results_may_be_truncated",
        False,
    )

    if not rows:

        return (
            "No matching grouped financial "
            "data was found."
        )

    title_map = {
        "merchant":
            "Top merchants",

        "category":
            "Spending by category",

        "subcategory":
            "Spending by subcategory",

        "month":
            "Spending by month",

        "protected":
            "Spending by protection status",

        "one_time_exceptional":
            "Spending by exceptional status",
    }

    title = title_map.get(
        group_by,
        "Grouped financial results",
    )

    lines = [
        f"{title}:"
    ]

    for index, row in enumerate(
        rows,
        start=1,
    ):

        label = row.get(
            group_by
        )

        value = row.get(
            "value"
        )

        transaction_count = row.get(
            "transaction_count"
        )

        if group_by == "protected":

            label = (
                "Protected"
                if label
                else "Non-protected"
            )

        elif group_by == "one_time_exceptional":

            label = (
                "One-time exceptional"
                if label
                else "Normal"
            )

        if operation in {
            "sum",
            "average",
        }:

            value_text = (
                f"${float(value):,.2f}"
            )

        else:

            value_text = (
                f"{int(value)}"
            )

        line = (
            f"{index}. {label} — "
            f"{value_text}"
        )

        if transaction_count is not None:

            tx_word = (
                "transaction"
                if transaction_count == 1
                else "transactions"
            )

            line += (
                f" ({transaction_count} "
                f"{tx_word})"
            )

        lines.append(
            line
        )

    if truncated:

        lines.append(
            ""
        )

        lines.append(
            "Showing the top results only; "
            "additional groups may exist."
        )

    return "\n".join(
        lines
    )


# ============================================================
# QUESTION CLASSIFICATION
# ============================================================

def _is_analytical_grouped_question(
    question,
):

    q = question.lower()

    return any(
        phrase in q
        for phrase in ANALYTICAL_GROUPED_PHRASES
    )


def _is_factual_grouped_question(
    question,
):

    q = question.lower()

    return any(
        phrase in q
        for phrase in FACTUAL_GROUPED_PHRASES
    )


# ============================================================
# REASONING MODEL FALLBACKS
# ============================================================

def _local_cfo_fallback(
    cfo_model,
    financial_context,
    question,
    database_context,
):

    answer = cfo_model.ask_cfo(
        financial_context=financial_context,
        question=question,
        database_context=database_context,
    )

    return {
        "answer": answer,
        "reasoning_model":
            "local-personal-cfo",
    }


def _lightning_cfo_answer(
    lightning,
    cfo_model,
    financial_context,
    question,
    database_context,
):

    lightning_result = lightning.ask_cfo(
        financial_context=financial_context,
        question=question,
        database_context=database_context,
    )

    if lightning_result.get(
        "success",
        False,
    ):

        return {
            "answer":
                lightning_result["answer"],

            "reasoning_model":
                lightning_result.get(
                    "model",
                    "nemotron-lightning",
                ),
        }

    return _local_cfo_fallback(
        cfo_model=cfo_model,
        financial_context=financial_context,
        question=question,
        database_context=database_context,
    )


def _reason_with_best_model(
    lightning,
    cfo_model,
    financial_context,
    question,
    database_context,
):

    if lightning is not None:

        return _lightning_cfo_answer(
            lightning=lightning,
            cfo_model=cfo_model,
            financial_context=financial_context,
            question=question,
            database_context=database_context,
        )

    return _local_cfo_fallback(
        cfo_model=cfo_model,
        financial_context=financial_context,
        question=question,
        database_context=database_context,
    )


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def answer_cfo_question(
    cfo_model,
    financial_context,
    question,
):

    # --------------------------------------------------------
    # 1. Initialize hosted Lightning when available
    # --------------------------------------------------------

    try:

        lightning = LightningClient()

    except Exception:

        lightning = None


    # --------------------------------------------------------
    # 2. Query planning
    #
    # Prefer hosted Lightning.
    # Fall back to local Personal CFO model.
    # --------------------------------------------------------

    intent_result = None

    if lightning is not None:

        intent_result = (
            lightning.parse_query_intent(
                question
            )
        )

    if (
        intent_result is None
        or not intent_result.get(
            "success",
            False,
        )
    ):

        intent_result = (
            cfo_model.parse_query_intent(
                question
            )
        )


    # --------------------------------------------------------
    # 3. Planner failed completely
    # --------------------------------------------------------

    if not intent_result.get(
        "success",
        False,
    ):

        database_context = {
            "database_used": False,
            "retrieval_error":
                intent_result.get(
                    "error"
                ),
        }

        reasoning = (
            _reason_with_best_model(
                lightning=lightning,
                cfo_model=cfo_model,
                financial_context=financial_context,
                question=question,
                database_context=database_context,
            )
        )

        return {
            "answer":
                reasoning["answer"],

            "intent":
                None,

            "database_context":
                database_context,

            "rows":
                [],

            "query_planner":
                "fallback",

            "reasoning_model":
                reasoning[
                    "reasoning_model"
                ],
        }


    # --------------------------------------------------------
    # 4. Valid intent
    # --------------------------------------------------------

    intent = intent_result[
        "intent"
    ]

    planner_model = intent_result.get(
        "model",
        "local-cfo-model",
    )


    # --------------------------------------------------------
    # 5. Question does not require database
    # --------------------------------------------------------

    if not intent.get(
        "needs_database",
        False,
    ):

        database_context = {
            "database_used": False,
        }

        reasoning = (
            _reason_with_best_model(
                lightning=lightning,
                cfo_model=cfo_model,
                financial_context=financial_context,
                question=question,
                database_context=database_context,
            )
        )

        return {
            "answer":
                reasoning["answer"],

            "intent":
                intent,

            "database_context":
                database_context,

            "rows":
                [],

            "query_planner":
                planner_model,

            "reasoning_model":
                reasoning[
                    "reasoning_model"
                ],
        }


    # --------------------------------------------------------
    # 6. Execute deterministic SQL
    # --------------------------------------------------------

    database_context = (
        execute_query_intent(
            intent
        )
    )


    # --------------------------------------------------------
    # 7. SQL/retrieval failure
    # --------------------------------------------------------

    if not database_context.get(
        "success",
        False,
    ):

        reasoning = (
            _reason_with_best_model(
                lightning=lightning,
                cfo_model=cfo_model,
                financial_context=financial_context,
                question=question,
                database_context=database_context,
            )
        )

        return {
            "answer":
                reasoning["answer"],

            "intent":
                intent,

            "database_context":
                database_context,

            "rows":
                [],

            "query_planner":
                planner_model,

            "reasoning_model":
                reasoning[
                    "reasoning_model"
                ],
        }


    # --------------------------------------------------------
    # 8. Handle exact database result
    # --------------------------------------------------------

    result_type = (
        database_context.get(
            "result_type"
        )
    )


    # ========================================================
    # ROW LIST
    # ========================================================

    if result_type == "rows":

        rows = database_context.get(
            "rows",
            [],
        )

        if not rows:

            answer = (
                "No matching transactions "
                "were found."
            )

        else:

            transaction_word = (
                "transaction"
                if len(rows) == 1
                else "transactions"
            )

            answer = (
                f"Found {len(rows)} matching "
                f"{transaction_word}."
            )

        return {
            "answer":
                answer,

            "intent":
                intent,

            "database_context":
                database_context,

            "rows":
                rows,

            "query_planner":
                planner_model,

            "reasoning_model":
                "deterministic-python",
        }


    # ========================================================
    # SINGLE AGGREGATE
    # ========================================================

    if result_type == "aggregate":

        answer = (
            _format_aggregate_answer(
                question=question,
                intent=intent,
                database_context=database_context,
            )
        )

        return {
            "answer":
                answer,

            "intent":
                intent,

            "database_context":
                database_context,

            "rows":
                [],

            "query_planner":
                planner_model,

            "reasoning_model":
                "deterministic-python",
        }


    # ========================================================
    # GROUPED RESULT
    # ========================================================

    if result_type == "grouped":

        rows = database_context.get(
            "rows",
            [],
        )

        if not rows:

            return {
                "answer":
                    (
                        "No matching grouped "
                        "financial data was found."
                    ),

                "intent":
                    intent,

                "database_context":
                    database_context,

                "rows":
                    [],

                "query_planner":
                    planner_model,

                "reasoning_model":
                    "deterministic-python",
            }


        analytical = (
            _is_analytical_grouped_question(
                question
            )
        )

        factual = (
            _is_factual_grouped_question(
                question
            )
        )


        # ----------------------------------------------------
        # Pure factual grouped question:
        #
        # SQL already has the answer.
        # Do NOT ask an LLM to reconstruct it.
        # ----------------------------------------------------

        if factual and not analytical:

            answer = (
                _format_grouped_answer(
                    database_context
                )
            )

            return {
                "answer":
                    answer,

                "intent":
                    intent,

                "database_context":
                    database_context,

                "rows":
                    rows,

                "query_planner":
                    planner_model,

                "reasoning_model":
                    "deterministic-python",
            }


        # ----------------------------------------------------
        # Analytical grouped question:
        #
        # SQL facts are passed to Lightning/local model.
        # Model interprets; it does not calculate the facts.
        # ----------------------------------------------------

        reasoning = (
            _reason_with_best_model(
                lightning=lightning,
                cfo_model=cfo_model,
                financial_context=financial_context,
                question=question,
                database_context=database_context,
            )
        )

        return {
            "answer":
                reasoning["answer"],

            "intent":
                intent,

            "database_context":
                database_context,

            "rows":
                rows,

            "query_planner":
                planner_model,

            "reasoning_model":
                reasoning[
                    "reasoning_model"
                ],
        }


    # --------------------------------------------------------
    # 9. Unexpected result type
    # --------------------------------------------------------

    reasoning = (
        _reason_with_best_model(
            lightning=lightning,
            cfo_model=cfo_model,
            financial_context=financial_context,
            question=question,
            database_context=database_context,
        )
    )

    return {
        "answer":
            reasoning["answer"],

        "intent":
            intent,

        "database_context":
            database_context,

        "rows":
            [],

        "query_planner":
            planner_model,

        "reasoning_model":
            reasoning[
                "reasoning_model"
            ],
    }
