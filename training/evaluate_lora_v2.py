import json
from pathlib import Path

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from peft import PeftModel


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "/workspace/personal_cfo/models/nemotron8b"

ADAPTER_PATH = (
    "/workspace/personal_cfo/"
    "models/personal-cfo-lora-v2-clean-unsloth/"
    "final_adapter"
)

TEST_FILE = (
    "/workspace/personal_cfo/lora_v2/"
    "personal_cfo_lora_v2_clean_test.jsonl"
)

MAX_SEQ_LENGTH = 1024

MAX_NEW_TOKENS = 160


# ============================================================
# HELPERS
# ============================================================

FIELDS = [
    "category",
    "subcategory",
    "protected",
    "one_time_exceptional",
]


def normalize_bool(value):

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() == "true"

    return bool(value)


def normalize_prediction(prediction):

    return {
        "category":
            str(
                prediction.get(
                    "category",
                    "",
                )
            ).strip(),

        "subcategory":
            str(
                prediction.get(
                    "subcategory",
                    "",
                )
            ).strip(),

        "protected":
            normalize_bool(
                prediction.get(
                    "protected",
                    False,
                )
            ),

        "one_time_exceptional":
            normalize_bool(
                prediction.get(
                    "one_time_exceptional",
                    False,
                )
            ),
    }


def extract_gold(example):

    assistant_messages = [
        message
        for message in example["messages"]
        if message["role"] == "assistant"
    ]

    if not assistant_messages:
        raise ValueError(
            "No assistant/gold message found."
        )

    gold_text = assistant_messages[-1]["content"]

    return normalize_prediction(
        json.loads(
            gold_text
        )
    )


def build_inference_messages(example):

    return [
        message
        for message in example["messages"]
        if message["role"] != "assistant"
    ]


def parse_model_json(text):

    text = text.strip()

    # First try: model returned clean JSON.
    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    # Second try:
    # find first {...} block if the model added text.
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:

        candidate = text[
            start:end + 1
        ]

        try:
            return json.loads(
                candidate
            )

        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not parse JSON: {text}"
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

print()
print("=" * 80)
print("Loading held-out test set")
print("=" * 80)

dataset = load_dataset(
    "json",
    data_files={
        "test": TEST_FILE,
    },
)

test_dataset = dataset["test"]

print(
    "Test examples:",
    len(test_dataset),
)


# ============================================================
# LOAD BASE MODEL
# ============================================================

print()
print("=" * 80)
print("Loading Nemotron 8B")
print("=" * 80)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)


# ============================================================
# LOAD LoRA
# ============================================================

print()
print("=" * 80)
print("Loading Personal CFO LoRA v2")
print("=" * 80)

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH,
)

FastLanguageModel.for_inference(
    model
)

model.eval()


# ============================================================
# EVALUATION COUNTERS
# ============================================================

field_correct = {
    field: 0
    for field in FIELDS
}

joint_correct = 0
parse_failures = 0

mistakes = []


# ============================================================
# RUN TEST SET
# ============================================================

print()
print("=" * 80)
print("Running held-out evaluation")
print("=" * 80)

for index, example in enumerate(
    test_dataset
):

    gold = extract_gold(
        example
    )

    messages = build_inference_messages(
        example
    )

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(
        model.device
    )

    with torch.inference_mode():

        output = model.generate(
            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=False,

            temperature=None,

            top_p=None,

            use_cache=True,
        )

    generated_tokens = output[
        0,
        inputs["input_ids"].shape[1]:
    ]

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    try:

        raw_prediction = parse_model_json(
            generated_text
        )

        prediction = normalize_prediction(
            raw_prediction
        )

    except Exception:

        parse_failures += 1

        prediction = {
            "category": "",
            "subcategory": "",
            "protected": False,
            "one_time_exceptional": False,
        }

    example_correct = True

    wrong_fields = []

    for field in FIELDS:

        if prediction[field] == gold[field]:

            field_correct[field] += 1

        else:

            example_correct = False

            wrong_fields.append(
                field
            )

    if example_correct:

        joint_correct += 1

    else:

        user_message = next(
            (
                message["content"]
                for message
                in example["messages"]
                if message["role"] == "user"
            ),
            "",
        )

        mistakes.append(
            {
                "index": index,
                "input": user_message,
                "gold": gold,
                "prediction": prediction,
                "wrong_fields": wrong_fields,
                "raw_output": generated_text,
            }
        )

    if (
        (index + 1) % 10 == 0
        or index == 0
    ):

        print(
            f"Completed "
            f"{index + 1}/"
            f"{len(test_dataset)}"
        )


# ============================================================
# RESULTS
# ============================================================

total = len(
    test_dataset
)

print()
print("=" * 80)
print("FINAL HELD-OUT RESULTS")
print("=" * 80)

for field in FIELDS:

    correct = field_correct[
        field
    ]

    accuracy = (
        correct
        / total
        * 100
    )

    print(
        f"{field:24s}: "
        f"{correct:3d}/{total} "
        f"= {accuracy:6.2f}%"
    )


joint_accuracy = (
    joint_correct
    / total
    * 100
)

print()
print(
    f"ALL 4 FIELDS CORRECT     : "
    f"{joint_correct:3d}/{total} "
    f"= {joint_accuracy:6.2f}%"
)

print(
    f"JSON parse failures      : "
    f"{parse_failures}"
)

print(
    f"Examples with any error  : "
    f"{len(mistakes)}"
)


# ============================================================
# SAVE ERROR ANALYSIS
# ============================================================

ERROR_FILE = Path(
    "/workspace/personal_cfo/"
    "lora_v2/"
    "lora_v2_test_errors.json"
)

with ERROR_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        mistakes,
        file,
        indent=2,
    )


print()
print(
    "Error analysis saved to:"
)

print(
    ERROR_FILE
)


# ============================================================
# PRINT FIRST 20 ERRORS
# ============================================================

print()
print("=" * 80)
print("FIRST 20 ERRORS")
print("=" * 80)

for mistake in mistakes[:20]:

    print()
    print(
        "-" * 80
    )

    print(
        "Test index:",
        mistake["index"],
    )

    print(
        mistake["input"]
    )

    print(
        "Gold:"
    )

    print(
        json.dumps(
            mistake["gold"],
            indent=2,
        )
    )

    print(
        "Prediction:"
    )

    print(
        json.dumps(
            mistake["prediction"],
            indent=2,
        )
    )

    print(
        "Wrong fields:",
        mistake["wrong_fields"],
    )
