import os
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


from csv_ingest import (
    load_multiple_csvs,
    build_model_input,
)

from model_backend import (
    PersonalCFOModel,
)

from finance_engine import (
    calculate_financial_summary,
)

from database import (
    initialize_database,
    get_existing_transaction_ids,
    save_transaction,
    load_all_transactions,
    get_transaction_count,
)

from cfo_query_engine import (
    answer_cfo_question,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Personal CFO",
    page_icon="💰",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_VERSION = (
    "personal-cfo-v2"
)


# ============================================================
# DATABASE
# ============================================================

initialize_database()


# ============================================================
# MODEL CLIENT
# ============================================================

@st.cache_resource
def get_cfo_model():

    return PersonalCFOModel()


cfo_model = get_cfo_model()


# ============================================================
# HELPERS
# ============================================================

def normalize_transactions_to_records(
    transactions,
):

    if transactions is None:
        return []

    if isinstance(
        transactions,
        pd.DataFrame,
    ):

        if transactions.empty:
            return []

        return transactions.to_dict(
            orient="records"
        )

    if isinstance(
        transactions,
        list,
    ):
        return transactions

    if isinstance(
        transactions,
        tuple,
    ):
        return list(
            transactions
        )

    raise TypeError(
        "Unsupported transaction container "
        "returned by csv_ingest: "
        f"{type(transactions).__name__}"
    )


def spending_transactions(
    dataframe,
):

    if dataframe.empty:
        return dataframe.copy()

    if (
        "count_as_spending"
        not in dataframe.columns
    ):
        return dataframe.copy()

    return dataframe[
        dataframe[
            "count_as_spending"
        ] == 1
    ].copy()


def build_finance_engine_records(
    dataframe,
):

    if dataframe.empty:
        return []

    records = []

    classification_fields = [
        "category",
        "subcategory",
        "financial_type",
        "protected",
        "cut_priority",
        "recurring",
        "reimbursable",
        "shared",
        "one_time_exceptional",
        "count_as_spending",
        "confidence",
    ]

    for _, row in dataframe.iterrows():

        record = row.to_dict()

        for date_field in [
            "date",
            "posted_date",
        ]:

            value = record.get(
                date_field
            )

            if value is None:
                continue

            try:

                if pd.isna(
                    value
                ):

                    record[
                        date_field
                    ] = None

                    continue

            except (
                TypeError,
                ValueError,
            ):

                pass

            if isinstance(
                value,
                pd.Timestamp,
            ):

                record[
                    date_field
                ] = value.strftime(
                    "%Y-%m-%d"
                )

            else:

                record[
                    date_field
                ] = str(
                    value
                )

        classification = {}

        for field in (
            classification_fields
        ):

            if field not in record:
                continue

            value = record.get(
                field
            )

            try:

                if pd.isna(
                    value
                ):
                    value = None

            except (
                TypeError,
                ValueError,
            ):

                pass

            classification[
                field
            ] = value

        record[
            "classification"
        ] = classification

        records.append(
            record
        )

    return records


def classify_new_transactions(
    transactions,
):

    transactions = (
        normalize_transactions_to_records(
            transactions
        )
    )

    if len(
        transactions
    ) == 0:

        return {
            "uploaded": 0,
            "existing": 0,
            "new": 0,
            "saved": 0,
            "failed": 0,
        }

    transaction_ids = [
        tx.get(
            "transaction_id"
        )
        for tx in transactions
        if tx.get(
            "transaction_id"
        )
    ]

    existing_ids = set(
        get_existing_transaction_ids(
            transaction_ids
        )
    )

    new_transactions = []

    existing_count = 0

    for tx in transactions:

        transaction_id = (
            tx.get(
                "transaction_id"
            )
        )

        if (
            transaction_id
            and transaction_id
            in existing_ids
        ):

            existing_count += 1

        else:

            new_transactions.append(
                tx
            )

    if not new_transactions:

        return {
            "uploaded":
                len(
                    transactions
                ),

            "existing":
                existing_count,

            "new":
                0,

            "saved":
                0,

            "failed":
                0,
        }

    progress_bar = (
        st.progress(
            0
        )
    )

    status_text = (
        st.empty()
    )

    saved_count = 0
    failed_count = 0

    total_new = len(
        new_transactions
    )

    for index, tx in enumerate(
        new_transactions,
        start=1,
    ):

        merchant = (
            tx.get(
                "merchant"
            )
            or tx.get(
                "description"
            )
            or "transaction"
        )

        status_text.text(
            f"Classifying "
            f"{index}/{total_new}: "
            f"{merchant}"
        )

        model_input = (
            build_model_input(
                tx
            )
        )

        result = (
            cfo_model
            .classify_transaction(
                model_input
            )
        )

        if not result.get(
            "success",
            False,
        ):

            failed_count += 1

            progress_bar.progress(
                index / total_new
            )

            continue

        classification = (
            result[
                "classification"
            ]
        )

        save_transaction(
            transaction=tx,
            classification=classification,
            model_version=MODEL_VERSION,
        )

        saved_count += 1

        progress_bar.progress(
            index / total_new
        )

    status_text.empty()
    progress_bar.empty()

    return {
        "uploaded":
            len(
                transactions
            ),

        "existing":
            existing_count,

        "new":
            total_new,

        "saved":
            saved_count,

        "failed":
            failed_count,
    }


def process_uploaded_files(
    uploaded_files,
):

    temp_paths = []

    try:

        for uploaded_file in (
            uploaded_files
        ):

            suffix = Path(
                uploaded_file.name
            ).suffix

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp_file:

                temp_file.write(
                    uploaded_file
                    .getbuffer()
                )

                temp_paths.append(
                    temp_file.name
                )

        transactions = (
            load_multiple_csvs(
                temp_paths
            )
        )

        return (
            normalize_transactions_to_records(
                transactions
            )
        )

    finally:

        for temp_path in (
            temp_paths
        ):

            try:

                os.remove(
                    temp_path
                )

            except OSError:

                pass


def load_transactions_dataframe():

    records = (
        load_all_transactions()
    )

    if not records:

        return (
            pd.DataFrame()
        )

    dataframe = (
        pd.DataFrame(
            records
        )
    )

    if (
        "date"
        in dataframe.columns
    ):

        dataframe[
            "date"
        ] = pd.to_datetime(
            dataframe[
                "date"
            ],
            errors="coerce",
        )

    if (
        "amount"
        in dataframe.columns
    ):

        dataframe[
            "amount"
        ] = pd.to_numeric(
            dataframe[
                "amount"
            ],
            errors="coerce",
        ).fillna(
            0.0
        )

    for boolean_column in [
        "protected",
        "one_time_exceptional",
        "count_as_spending",
    ]:

        if (
            boolean_column
            in dataframe.columns
        ):

            dataframe[
                boolean_column
            ] = pd.to_numeric(
                dataframe[
                    boolean_column
                ],
                errors="coerce",
            ).fillna(
                0
            ).astype(
                int
            )

    return dataframe


def build_v2_financial_summary(
    dataframe,
):

    spending_df = (
        spending_transactions(
            dataframe
        )
    )

    if spending_df.empty:

        return {
            "total_spending":
                0.0,

            "protected_spending":
                0.0,

            "reducible_spending":
                0.0,

            "one_time_exceptional_spending":
                0.0,

            "subscription_spending":
                0.0,
        }

    total_spending = float(
        spending_df[
            "amount"
        ].sum()
    )

    if (
        "protected"
        in spending_df.columns
    ):

        protected_spending = float(
            spending_df.loc[
                spending_df[
                    "protected"
                ] == 1,
                "amount",
            ].sum()
        )

    else:

        protected_spending = 0.0

    reducible_spending = (
        total_spending
        - protected_spending
    )

    if (
        "one_time_exceptional"
        in spending_df.columns
    ):

        exceptional_spending = float(
            spending_df.loc[
                spending_df[
                    "one_time_exceptional"
                ] == 1,
                "amount",
            ].sum()
        )

    else:

        exceptional_spending = 0.0

    if (
        "category"
        in spending_df.columns
    ):

        subscription_spending = float(
            spending_df.loc[
                spending_df[
                    "category"
                ] == "Subscriptions",
                "amount",
            ].sum()
        )

    else:

        subscription_spending = 0.0

    return {
        "total_spending":
            round(
                total_spending,
                2,
            ),

        "protected_spending":
            round(
                protected_spending,
                2,
            ),

        "reducible_spending":
            round(
                reducible_spending,
                2,
            ),

        "one_time_exceptional_spending":
            round(
                exceptional_spending,
                2,
            ),

        "subscription_spending":
            round(
                subscription_spending,
                2,
            ),
    }


# ============================================================
# HEADER
# ============================================================

st.title(
    "💰 Personal CFO"
)

st.caption(
    "Fine-tuned local Nemotron for transaction classification, "
    "Nemotron 3.5 Lightning for semantic query planning and "
    "financial reasoning, and SQLite/Python for exact financial facts."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Upload Transactions"
    )

    uploaded_files = (
        st.file_uploader(
            (
                "Upload Bank of America "
                "or Capital One CSVs"
            ),
            type=[
                "csv"
            ],
            accept_multiple_files=True,
        )
    )

    if uploaded_files:

        upload_signature = tuple(
            (
                uploaded_file.name,
                uploaded_file.size,
            )
            for uploaded_file
            in uploaded_files
        )

        previous_signature = (
            st.session_state.get(
                "upload_signature"
            )
        )

        if (
            upload_signature
            != previous_signature
        ):

            with st.spinner(
                (
                    "Reading and classifying "
                    "new transactions..."
                )
            ):

                try:

                    uploaded_transactions = (
                        process_uploaded_files(
                            uploaded_files
                        )
                    )

                    stats = (
                        classify_new_transactions(
                            uploaded_transactions
                        )
                    )

                    st.session_state[
                        "upload_signature"
                    ] = (
                        upload_signature
                    )

                    st.success(
                        (
                            f"Processed "
                            f"{stats['uploaded']} "
                            f"transactions."
                        )
                    )

                    st.write(
                        (
                            f"Already stored: "
                            f"{stats['existing']}"
                        )
                    )

                    st.write(
                        (
                            f"New transactions: "
                            f"{stats['new']}"
                        )
                    )

                    st.write(
                        (
                            f"Saved: "
                            f"{stats['saved']}"
                        )
                    )

                    if (
                        stats[
                            "failed"
                        ]
                        > 0
                    ):

                        st.warning(
                            (
                                f"Classification "
                                f"failed for "
                                f"{stats['failed']} "
                                f"transactions."
                            )
                        )

                except Exception as exc:

                    st.error(
                        (
                            "Upload failed: "
                            f"{exc}"
                        )
                    )

        else:

            st.info(
                (
                    "These files have already "
                    "been processed in this "
                    "session."
                )
            )

    st.divider()

    st.caption(
        (
            f"SQLite transactions: "
            f"{get_transaction_count()}"
        )
    )

    st.caption(
        (
            "Classifier: "
            "personal-cfo-v2"
        )
    )

    st.caption(
        (
            "Planner: "
            "Nemotron 3.5 Lightning"
        )
    )


# ============================================================
# LOAD DATABASE
# ============================================================

transactions_df = (
    load_transactions_dataframe()
)


if transactions_df.empty:

    st.info(
        "Upload transaction CSVs to begin."
    )

    st.stop()


# ============================================================
# FINANCIAL CONTEXT
# ============================================================

finance_engine_records = (
    build_finance_engine_records(
        transactions_df
    )
)


legacy_financial_summary = (
    calculate_financial_summary(
        finance_engine_records
    )
    if finance_engine_records
    else {}
)


v2_summary = (
    build_v2_financial_summary(
        transactions_df
    )
)


financial_summary = dict(
    legacy_financial_summary
)

financial_summary.update(
    v2_summary
)


# ============================================================
# COMMON DATA
# ============================================================

spending_df = (
    spending_transactions(
        transactions_df
    )
)


if (
    not spending_df.empty
    and "date"
    in spending_df.columns
):

    spending_df[
        "month"
    ] = (
        spending_df[
            "date"
        ]
        .dt.to_period(
            "M"
        )
        .astype(
            str
        )
    )


# ============================================================
# TABS
# ============================================================

(
    dashboard_tab,
    transactions_tab,
    ask_cfo_tab,
) = st.tabs(
    [
        "Dashboard",
        "Transactions",
        "Ask CFO",
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

with dashboard_tab:

    st.subheader(
        "Financial Overview"
    )


    # --------------------------------------------------------
    # PRIMARY METRICS
    # --------------------------------------------------------

    (
        metric_1,
        metric_2,
        metric_3,
        metric_4,
    ) = st.columns(
        4
    )


    metric_1.metric(
        "Total Spending",
        (
            f"${v2_summary['total_spending']:,.2f}"
        ),
    )


    metric_2.metric(
        "Protected Spending",
        (
            f"${v2_summary['protected_spending']:,.2f}"
        ),
    )


    metric_3.metric(
        "Reducible Spending",
        (
            f"${v2_summary['reducible_spending']:,.2f}"
        ),
    )


    metric_4.metric(
        "One-Time Exceptional",
        (
            f"${v2_summary['one_time_exceptional_spending']:,.2f}"
        ),
    )


    st.caption(
        (
            "Reducible spending is defined as "
            "counted spending that is not marked protected. "
            "It is an analytical bucket, not a recommendation "
            "to cut every included expense."
        )
    )


    st.divider()


    if not spending_df.empty:


        # ====================================================
        # CATEGORY + MONTH FILTER
        # ====================================================

        st.subheader(
            "Spending by Category"
        )


        available_months = sorted(
            spending_df[
                "month"
            ]
            .dropna()
            .unique()
            .tolist()
        )


        selected_month = (
            st.selectbox(
                (
                    "Filter dashboard "
                    "by month"
                ),
                options=[
                    "All months",
                    *available_months,
                ],
                key="dashboard_month",
            )
        )


        if (
            selected_month
            != "All months"
        ):

            filtered_spending_df = (
                spending_df[
                    spending_df[
                        "month"
                    ]
                    == selected_month
                ].copy()
            )

        else:

            filtered_spending_df = (
                spending_df.copy()
            )


        grouped_categories = (
            filtered_spending_df
            .groupby(
                "category",
                dropna=False,
            )[
                "amount"
            ]
            .sum()
            .reset_index()
            .sort_values(
                "amount",
                ascending=False,
            )
        )


        category_fig = (
            px.bar(
                grouped_categories,
                x="category",
                y="amount",
                labels={
                    "category":
                        "Category",

                    "amount":
                        "Spending ($)",
                },
            )
        )


        st.plotly_chart(
            category_fig,
            use_container_width=True,
        )


        if (
            selected_month
            != "All months"
        ):

            selected_total = (
                filtered_spending_df[
                    "amount"
                ].sum()
            )

            st.caption(
                (
                    f"Total spending in "
                    f"{selected_month}: "
                    f"${selected_total:,.2f}"
                )
            )


        # ====================================================
        # MONTHLY + TOP MERCHANTS
        # ====================================================

        left_col, right_col = (
            st.columns(
                2
            )
        )


        with left_col:

            st.subheader(
                "Monthly Spending"
            )

            monthly_spending = (
                spending_df
                .groupby(
                    "month"
                )[
                    "amount"
                ]
                .sum()
                .reset_index()
                .sort_values(
                    "month"
                )
            )

            monthly_fig = (
                px.line(
                    monthly_spending,
                    x="month",
                    y="amount",
                    markers=True,
                    labels={
                        "month":
                            "Month",

                        "amount":
                            "Spending ($)",
                    },
                )
            )

            st.plotly_chart(
                monthly_fig,
                use_container_width=True,
            )


        with right_col:

            st.subheader(
                "Top Merchants"
            )

            top_merchants = (
                filtered_spending_df
                .groupby(
                    "merchant",
                    dropna=False,
                )[
                    "amount"
                ]
                .sum()
                .reset_index()
                .sort_values(
                    "amount",
                    ascending=False,
                )
                .head(
                    10
                )
            )

            merchant_fig = (
                px.bar(
                    top_merchants,
                    x="amount",
                    y="merchant",
                    orientation="h",
                    labels={
                        "merchant":
                            "Merchant",

                        "amount":
                            "Spending ($)",
                    },
                )
            )

            merchant_fig.update_layout(
                yaxis={
                    "categoryorder":
                        "total ascending"
                }
            )

            st.plotly_chart(
                merchant_fig,
                use_container_width=True,
            )


        # ====================================================
        # SUBSCRIPTIONS + EXCEPTIONAL
        # ====================================================

        subscription_col, exceptional_col = (
            st.columns(
                2
            )
        )


        with subscription_col:

            st.subheader(
                "Subscriptions"
            )

            subscription_df = (
                spending_df[
                    spending_df[
                        "category"
                    ]
                    == "Subscriptions"
                ]
                .copy()
                .sort_values(
                    "date",
                    ascending=False,
                )
            )


            st.metric(
                "Subscription Spending",
                (
                    f"${v2_summary['subscription_spending']:,.2f}"
                ),
            )


            if (
                subscription_df.empty
            ):

                st.info(
                    (
                        "No subscription "
                        "transactions found."
                    )
                )

            else:

                subscription_columns = [
                    column
                    for column in [
                        "date",
                        "merchant",
                        "amount",
                        "subcategory",
                    ]
                    if column
                    in subscription_df.columns
                ]

                st.dataframe(
                    subscription_df[
                        subscription_columns
                    ],
                    use_container_width=True,
                    hide_index=True,
                )


        with exceptional_col:

            st.subheader(
                "One-Time Exceptional"
            )


            if (
                "one_time_exceptional"
                in spending_df.columns
            ):

                exceptional_df = (
                    spending_df[
                        spending_df[
                            "one_time_exceptional"
                        ]
                        == 1
                    ]
                    .copy()
                    .sort_values(
                        "amount",
                        ascending=False,
                    )
                )

            else:

                exceptional_df = (
                    pd.DataFrame()
                )


            st.metric(
                "Exceptional Spending",
                (
                    f"${v2_summary['one_time_exceptional_spending']:,.2f}"
                ),
            )


            if exceptional_df.empty:

                st.info(
                    (
                        "No transactions are "
                        "currently marked "
                        "one-time exceptional."
                    )
                )

            else:

                exceptional_columns = [
                    column
                    for column in [
                        "date",
                        "merchant",
                        "amount",
                        "category",
                        "subcategory",
                    ]
                    if column
                    in exceptional_df.columns
                ]

                st.dataframe(
                    exceptional_df[
                        exceptional_columns
                    ],
                    use_container_width=True,
                    hide_index=True,
                )


# ============================================================
# TRANSACTION HISTORY
# ============================================================

with transactions_tab:

    st.subheader(
        "Transaction History"
    )


    (
        filter_col_1,
        filter_col_2,
        filter_col_3,
        filter_col_4,
    ) = st.columns(
        4
    )


    sources = sorted(
        transactions_df[
            "source"
        ]
        .dropna()
        .astype(
            str
        )
        .unique()
        .tolist()
    )


    categories = sorted(
        transactions_df[
            "category"
        ]
        .dropna()
        .astype(
            str
        )
        .unique()
        .tolist()
    )


    with filter_col_1:

        selected_source = (
            st.selectbox(
                "Source",
                [
                    "All",
                    *sources,
                ],
                key=(
                    "transaction_source"
                ),
            )
        )


    with filter_col_2:

        selected_category = (
            st.selectbox(
                "Category",
                [
                    "All",
                    *categories,
                ],
                key=(
                    "transaction_category"
                ),
            )
        )


    with filter_col_3:

        protected_filter = (
            st.selectbox(
                "Protection",
                [
                    "All",
                    "Protected",
                    "Non-protected",
                ],
            )
        )


    with filter_col_4:

        exceptional_filter = (
            st.selectbox(
                "Exceptional",
                [
                    "All",
                    "Exceptional only",
                    "Normal only",
                ],
            )
        )


    filtered_df = (
        transactions_df.copy()
    )


    if (
        selected_source
        != "All"
    ):

        filtered_df = (
            filtered_df[
                filtered_df[
                    "source"
                ]
                == selected_source
            ]
        )


    if (
        selected_category
        != "All"
    ):

        filtered_df = (
            filtered_df[
                filtered_df[
                    "category"
                ]
                == selected_category
            ]
        )


    if (
        protected_filter
        == "Protected"
        and "protected"
        in filtered_df.columns
    ):

        filtered_df = (
            filtered_df[
                filtered_df[
                    "protected"
                ]
                == 1
            ]
        )


    elif (
        protected_filter
        == "Non-protected"
        and "protected"
        in filtered_df.columns
    ):

        filtered_df = (
            filtered_df[
                filtered_df[
                    "protected"
                ]
                == 0
            ]
        )


    if (
        exceptional_filter
        == "Exceptional only"
        and "one_time_exceptional"
        in filtered_df.columns
    ):

        filtered_df = (
            filtered_df[
                filtered_df[
                    "one_time_exceptional"
                ]
                == 1
            ]
        )


    elif (
        exceptional_filter
        == "Normal only"
        and "one_time_exceptional"
        in filtered_df.columns
    ):

        filtered_df = (
            filtered_df[
                filtered_df[
                    "one_time_exceptional"
                ]
                == 0
            ]
        )


    st.caption(
        (
            f"{len(filtered_df)} "
            f"matching transactions"
        )
    )


    display_columns = [
        column
        for column in [
            "date",
            "merchant",
            "amount",
            "category",
            "subcategory",
            "protected",
            "one_time_exceptional",
            "count_as_spending",
            "source",
            "model_version",
        ]
        if column
        in filtered_df.columns
    ]


    display_df = (
        filtered_df[
            display_columns
        ]
        .sort_values(
            "date",
            ascending=False,
        )
    )


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ASK CFO
# ============================================================

with ask_cfo_tab:

    st.subheader(
        "Ask Your CFO"
    )


    st.caption(
        (
            "Nemotron 3.5 Lightning interprets "
            "your question. SQLite and Python "
            "calculate exact financial facts. "
            "The local fine-tuned Nemotron model "
            "provides fallback planning and reasoning."
        )
    )


    example_col_1, example_col_2 = (
        st.columns(
            2
        )
    )


    with example_col_1:

        st.markdown(
            """
**Try factual questions**
- How much did I spend in total?
- How much of my spending is protected?
- What are my biggest spending categories?
- Show me my subscriptions.
"""
        )


    with example_col_2:

        st.markdown(
            """
**Try analytical questions**
- Where can I reduce spending?
- What patterns do you see in my spending?
- Why is my travel spending high?
- What should I pay attention to?
"""
        )


    question = (
        st.text_input(
            "Ask a financial question",
            placeholder=(
                "What are my biggest "
                "spending categories?"
            ),
        )
    )


    ask_button = (
        st.button(
            "Ask CFO",
            type="primary",
        )
    )


    if (
        ask_button
        and question.strip()
    ):

        with st.spinner(
            "Analyzing..."
        ):

            try:

                result = (
                    answer_cfo_question(
                        cfo_model=(
                            cfo_model
                        ),

                        financial_context=(
                            financial_summary
                        ),

                        question=(
                            question.strip()
                        ),
                    )
                )


                # --------------------------------------------
                # ANSWER
                # --------------------------------------------

                st.markdown(
                    "### Answer"
                )

                st.markdown(
                    result.get(
                        "answer",
                        (
                            "No answer "
                            "returned."
                        ),
                    )
                )


                result_rows = (
                    result.get(
                        "rows",
                        [],
                    )
                )


                database_context = (
                    result.get(
                        "database_context",
                        {},
                    )
                )


                result_type = (
                    database_context.get(
                        "result_type"
                    )
                )


                # --------------------------------------------
                # ROW RESULTS
                # --------------------------------------------

                if (
                    result_rows
                    and result_type
                    == "rows"
                ):

                    st.markdown(
                        (
                            "### Matching "
                            "Transactions"
                        )
                    )

                    st.dataframe(
                        pd.DataFrame(
                            result_rows
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )


                # --------------------------------------------
                # GROUPED RESULTS
                # --------------------------------------------

                elif (
                    result_rows
                    and result_type
                    == "grouped"
                ):

                    st.markdown(
                        (
                            "### Retrieved "
                            "Groups"
                        )
                    )

                    grouped_result_df = (
                        pd.DataFrame(
                            result_rows
                        )
                    )

                    st.dataframe(
                        grouped_result_df,
                        use_container_width=True,
                        hide_index=True,
                    )


                # --------------------------------------------
                # EXECUTION METADATA
                # --------------------------------------------

                st.markdown(
                    "### Execution"
                )


                execution_col_1, execution_col_2 = (
                    st.columns(
                        2
                    )
                )


                with execution_col_1:

                    st.caption(
                        "Query Planner"
                    )

                    st.code(
                        result.get(
                            "query_planner",
                            "unknown",
                        )
                    )


                with execution_col_2:

                    st.caption(
                        "Answer / Reasoning"
                    )

                    st.code(
                        result.get(
                            "reasoning_model",
                            "unknown",
                        )
                    )


                with st.expander(
                    (
                        "Technical retrieval "
                        "details"
                    )
                ):

                    st.markdown(
                        "**Result type**"
                    )

                    st.code(
                        str(
                            result_type
                        )
                    )


                    if (
                        "results_may_be_truncated"
                        in database_context
                    ):

                        st.markdown(
                            (
                                "**Results may "
                                "be truncated**"
                            )
                        )

                        st.code(
                            str(
                                database_context.get(
                                    "results_may_be_truncated"
                                )
                            )
                        )


                    st.markdown(
                        (
                            "**Structured "
                            "query intent**"
                        )
                    )

                    st.json(
                        result.get(
                            "intent"
                        )
                        or {}
                    )


                    st.markdown(
                        (
                            "**Database "
                            "context**"
                        )
                    )

                    st.json(
                        database_context
                    )


            except Exception as exc:

                st.error(
                    (
                        "Ask CFO failed: "
                        f"{exc}"
                    )
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    (
        f"Local SQLite database: "
        f"{get_transaction_count()} "
        f"transactions · "
        f"Classifier: personal-cfo-v2 · "
        f"Planner: Nemotron 3.5 Lightning · "
        f"Financial truth layer: SQLite/Python"
    )
)
