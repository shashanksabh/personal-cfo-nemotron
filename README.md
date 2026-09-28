# Personal CFO

Personal finance assistant built on NVIDIA DGX Spark.

It classifies bank transactions with a fine-tuned Nemotron model, stores the results in SQLite, and answers natural-language questions using Nemotron 3.5 Lightning for query planning and Python/SQLite for exact calculations.

## Stack

- `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`
- Unsloth QLoRA
- vLLM on DGX Spark
- Nemotron 3.5 Lightning
- SQLite + Python
- Streamlit

## Model

The local classifier predicts:

- category
- subcategory
- protected
- one-time exceptional

Held-out evaluation:

- Category accuracy: 91%
- Subcategory accuracy: 91%
- Protected accuracy: 100%
- One-time exceptional accuracy: 100%
- Exact match across all four fields: 91%

Production merchant strings exposed additional distribution shift, so historical labels were preserved rather than overwritten by the new model.

## Architecture

LLMs handle semantics. SQLite and Python provide the financial source of truth. LLMs explain the retrieved facts.

## Run

1. Copy `.env.example` to `.personal_cfo.env` and add your NVIDIA API key.
2. Start the local vLLM server.
3. Start the Streamlit app.
4. Open port `8501`.

This repository intentionally excludes real bank exports, API keys, model weights, and personal financial data.
