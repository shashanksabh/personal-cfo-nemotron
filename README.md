# Personal CFO

A fun weekend project - Personal finance assistant built on NVIDIA DGX Spark.

It classifies bank transactions with a locally fine-tuned Nemotron model, stores the results in SQLite, and answers natural-language finance questions using Nemotron 3.5 Lightning for query planning and Python/SQLite for exact calculations.

## Stack

- NVIDIA DGX Spark
- `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`
- Unsloth + QLoRA
- vLLM
- Nemotron 3.5 Lightning
- SQLite + Python
- Streamlit

## Models

### Local transaction classifier

Base model: `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`

I fine-tuned the 8B model with Unsloth using QLoRA, so only lightweight LoRA adapter weights are trained rather than updating the full model. The resulting `personal-cfo-v2` model runs locally on DGX Spark through vLLM.

For each transaction, it predicts:

- category
- subcategory
- protected
- one-time exceptional

On the held-out test set, category and subcategory accuracy were 91%, while the two boolean fields were 100%. Real bank merchant strings were more varied than the training set, so I kept the existing historical labels instead of automatically rewriting them with the new model.

### Query planner and reasoner

`nvidia/nemotron-3.5-lightning-30b-a3b` is accessed through the NVIDIA API.

It converts natural-language questions into structured query intent and can reason over retrieved financial context. Exact totals, filters, and aggregations are still computed by Python and SQLite.

## Architecture

LLMs handle semantics. SQLite and Python provide the financial source of truth. LLMs explain the retrieved facts.

## Run

1. Copy `.env.example` to `.personal_cfo.env` and add your NVIDIA API key.
2. Start the local vLLM server.
3. Start the Streamlit app.
4. Open port `8501`.

This repository excludes real bank exports, API keys, model weights, and personal financial data.
