import sqlite3

from database import get_connection


# ============================================================
# ALLOWED QUERY OPERATIONS
# ============================================================

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
# FILTER BUILDING
# ============================================================

def _build_filters(intent):
    """
    Build a parameterized WHERE clause from a validated
    structured query intent.

    No user-provided value is interpolated directly into SQL.
    """

    where_clauses = []
    params = []

    merchant_contains = intent.get(
        "merchant_contains"
    )

    category = intent.get(
        "category"
    )

    subcategory = intent.get(
        "subcategory"
    )

    start_date = intent.get(
        "start_date"
    )

    end_date = intent.get(
        "end_date"
    )

    spending_only = intent.get(
        "spending_only",
        False,
    )

    protected = intent.get(
        "protected"
    )

    one_time_exceptional = intent.get(
        "one_time_exceptional"
    )

    # --------------------------------------------------------
    # Merchant
    # --------------------------------------------------------

    if merchant_contains:
        where_clauses.append(
            "UPPER(merchant) LIKE UPPER(?)"
        )

        params.append(
            f"%{merchant_contains}%"
        )

    # --------------------------------------------------------
    # Category / subcategory
    # --------------------------------------------------------

    if category:
        where_clauses.append(
            "category = ?"
        )

        params.append(
            category
        )

    if subcategory:
        where_clauses.append(
            "subcategory = ?"
        )

        params.append(
            subcategory
        )

    # --------------------------------------------------------
    # Date range
    # --------------------------------------------------------

    if start_date:
        where_clauses.append(
            "date >= ?"
        )

        params.append(
            start_date
        )

    if end_date:
        where_clauses.append(
            "date <= ?"
        )

        params.append(
            end_date
        )

    # --------------------------------------------------------
    # Spending
    # --------------------------------------------------------

    if spending_only:
        where_clauses.append(
            "count_as_spending = 1"
        )

    # --------------------------------------------------------
    # Protected spending
    # --------------------------------------------------------

    if protected is not None:
        where_clauses.append(
            "protected = ?"
        )

        params.append(
            int(bool(protected))
        )

    # --------------------------------------------------------
    # One-time exceptional spending
    # --------------------------------------------------------

    if one_time_exceptional is not None:
        where_clauses.append(
            "one_time_exceptional = ?"
        )

        params.append(
            int(
                bool(
                    one_time_exceptional
                )
            )
        )

    # --------------------------------------------------------
    # Final WHERE
    # --------------------------------------------------------

    if where_clauses:
        where_sql = (
            " WHERE "
            + " AND ".join(
                where_clauses
            )
        )

    else:
        where_sql = ""

    return (
        where_sql,
        params,
    )


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize_limit(intent):
    limit = intent.get(
        "limit",
        20,
    )

    try:
        limit = int(
            limit
        )

    except (
        TypeError,
        ValueError,
    ):
        limit = 20

    return max(
        1,
        min(
            limit,
            100,
        ),
    )


def _normalize_order(intent):
    order = str(
        intent.get(
            "order",
            "desc",
        )
    ).lower()

    if order not in ALLOWED_ORDER:
        order = "desc"

    return order


# ============================================================
# AGGREGATE QUERY
# ============================================================

def _execute_aggregate_query(
    connection,
    intent,
):

    operation = intent[
        "operation"
    ]

    (
        where_sql,
        params,
    ) = _build_filters(
        intent
    )

    if operation == "sum":

        select_sql = (
            "COALESCE(SUM(amount), 0)"
        )

    elif operation == "count":

        select_sql = (
            "COUNT(*)"
        )

    elif operation == "average":

        select_sql = (
            "COALESCE(AVG(amount), 0)"
        )

    else:

        return {
            "success": False,
            "database_used": True,
            "error": (
                "Unsupported aggregate "
                f"operation: {operation}"
            ),
            "intent": intent,
        }

    sql = (
        f"SELECT {select_sql} AS result "
        f"FROM transactions"
        f"{where_sql}"
    )

    cursor = connection.execute(
        sql,
        params,
    )

    row = cursor.fetchone()

    result = (
        row["result"]
        if row is not None
        else 0
    )

    if isinstance(
        result,
        float,
    ):
        result = round(
            result,
            2,
        )

    return {
        "success": True,
        "database_used": True,
        "result_type": "aggregate",
        "operation": operation,
        "result": result,
        "intent": intent,
    }


# ============================================================
# LIST QUERY
# ============================================================

def _execute_list_query(
    connection,
    intent,
):

    (
        where_sql,
        params,
    ) = _build_filters(
        intent
    )

    limit = _normalize_limit(
        intent
    )

    order = _normalize_order(
        intent
    )

    order_sql = (
        "DESC"
        if order == "desc"
        else "ASC"
    )

    sql = f"""
        SELECT
            date,
            merchant,
            description,
            amount,
            category,
            subcategory,
            protected,
            one_time_exceptional,
            count_as_spending,
            source
        FROM transactions
        {where_sql}
        ORDER BY date {order_sql}, transaction_id {order_sql}
        LIMIT ?
    """

    params_with_limit = (
        params
        + [limit]
    )

    cursor = connection.execute(
        sql,
        params_with_limit,
    )

    rows = [
        dict(row)
        for row
        in cursor.fetchall()
    ]

    return {
        "success": True,
        "database_used": True,
        "result_type": "rows",
        "operation": "list",
        "result": None,
        "rows": rows,
        "returned_rows": len(
            rows
        ),
        "limit": limit,
        "results_may_be_truncated": (
            len(rows)
            == limit
        ),
        "intent": intent,
    }


# ============================================================
# GROUP BY
# ============================================================

def _get_group_expression(
    group_by,
):

    if group_by == "category":

        return (
            "category",
            "category",
        )

    if group_by == "subcategory":

        return (
            "subcategory",
            "subcategory",
        )

    if group_by == "merchant":

        return (
            "merchant",
            "merchant",
        )

    if group_by == "month":

        return (
            "substr(date, 1, 7)",
            "month",
        )

    if group_by == "protected":

        return (
            "protected",
            "protected",
        )

    if group_by == "one_time_exceptional":

        return (
            "one_time_exceptional",
            "one_time_exceptional",
        )

    return (
        None,
        None,
    )


def _execute_grouped_query(
    connection,
    intent,
):

    operation = intent.get(
        "operation"
    )

    group_by = intent.get(
        "group_by"
    )

    if group_by not in ALLOWED_GROUP_BY:

        return {
            "success": False,
            "database_used": True,
            "error": (
                "Unsupported group_by: "
                f"{group_by}"
            ),
            "intent": intent,
        }

    if operation not in {
        "sum",
        "count",
        "average",
    }:

        return {
            "success": False,
            "database_used": True,
            "error": (
                "Grouped queries support "
                "only sum, count, or average."
            ),
            "intent": intent,
        }

    (
        where_sql,
        params,
    ) = _build_filters(
        intent
    )

    limit = _normalize_limit(
        intent
    )

    order = _normalize_order(
        intent
    )

    order_sql = (
        "DESC"
        if order == "desc"
        else "ASC"
    )

    (
        group_expression,
        group_alias,
    ) = _get_group_expression(
        group_by
    )

    if group_expression is None:

        return {
            "success": False,
            "database_used": True,
            "error": (
                "Unsupported group_by: "
                f"{group_by}"
            ),
            "intent": intent,
        }

    if operation == "sum":

        aggregate_expression = (
            "COALESCE(SUM(amount), 0)"
        )

    elif operation == "count":

        aggregate_expression = (
            "COUNT(*)"
        )

    else:

        aggregate_expression = (
            "COALESCE(AVG(amount), 0)"
        )

    sql = f"""
        SELECT
            {group_expression} AS group_value,
            {aggregate_expression} AS value,
            COUNT(*) AS transaction_count
        FROM transactions
        {where_sql}
        GROUP BY {group_expression}
        ORDER BY value {order_sql}
        LIMIT ?
    """

    params_with_limit = (
        params
        + [limit]
    )

    cursor = connection.execute(
        sql,
        params_with_limit,
    )

    raw_rows = cursor.fetchall()

    rows = []

    for row in raw_rows:

        value = row[
            "value"
        ]

        if isinstance(
            value,
            float,
        ):

            value = round(
                value,
                2,
            )

        group_value = row[
            "group_value"
        ]

        # SQLite stores booleans as 0/1.
        if group_by in {
            "protected",
            "one_time_exceptional",
        }:

            group_value = bool(
                group_value
            )

        rows.append(
            {
                group_alias:
                    group_value,

                "value":
                    value,

                "transaction_count":
                    row[
                        "transaction_count"
                    ],
            }
        )

    return {
        "success": True,
        "database_used": True,
        "result_type": "grouped",
        "operation": operation,
        "group_by": group_by,
        "order": order,
        "rows": rows,
        "returned_groups": len(
            rows
        ),
        "limit": limit,
        "results_may_be_truncated": (
            len(rows)
            == limit
        ),
        "intent": intent,
    }


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def execute_query_intent(
    intent,
):

    try:

        needs_database = intent.get(
            "needs_database",
            False,
        )

        if not needs_database:

            return {
                "success": True,
                "database_used": False,
                "result_type": None,
                "intent": intent,
            }

        operation = intent.get(
            "operation"
        )

        group_by = intent.get(
            "group_by"
        )

        if operation not in ALLOWED_OPERATIONS:

            return {
                "success": False,
                "database_used": True,
                "error": (
                    "Unsupported operation: "
                    f"{operation}"
                ),
                "intent": intent,
            }

        connection = get_connection()

        try:

            if group_by is not None:

                return _execute_grouped_query(
                    connection,
                    intent,
                )

            if operation == "list":

                return _execute_list_query(
                    connection,
                    intent,
                )

            return _execute_aggregate_query(
                connection,
                intent,
            )

        finally:

            connection.close()

    except sqlite3.Error as exc:

        return {
            "success": False,
            "database_used": True,
            "error": (
                f"SQLite error: {exc}"
            ),
            "intent": intent,
        }

    except Exception as exc:

        return {
            "success": False,
            "database_used": True,
            "error": str(
                exc
            ),
            "intent": intent,
        }
