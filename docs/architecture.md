# Architecture

```mermaid
flowchart LR
    A[Bank of America / Capital One CSVs] --> B[CSV ingestion + deduplication]
    B --> C[personal-cfo-v2 classifier]

    subgraph Spark[NVIDIA DGX Spark]
        C
        D[nvidia/Llama-3.1-Nemotron-Nano-8B-v1]
        E[Unsloth + QLoRA]
        F[vLLM]
        G[SQLite + Python]
        H[Streamlit]
        E -->|LoRA adapter| C
        D --> E
        C --> F
        F --> G
        G --> H
    end

    H --> I[Natural-language question]
    I --> J[Nemotron 3.5 Lightning
NVIDIA API]
    J -->|structured query intent| G
    G -->|exact rows / aggregates| J
    J -->|answer / explanation| H
```

The local classifier is a LoRA adapter fine-tuned from `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` with Unsloth + QLoRA and served with vLLM on DGX Spark.

Nemotron 3.5 Lightning handles semantic query planning and higher-level reasoning. Python and SQLite perform exact financial calculations.
