import json
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "/workspace/personal_cfo/models/nemotron8b"

ADAPTER = (
    "/workspace/personal_cfo/nemo_training/output/"
    "nemotron8b_run50/epoch_1_step_49/model"
)

TEST_FILE = "/workspace/personal_cfo/test.jsonl"

OUTPUT_FILE = "/workspace/personal_cfo/eval_8b_fair_results.json"


FIELDS = [
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


SYSTEM_PROMPT = """
You are a transaction-classification system for a Personal CFO application.

The user will provide a transaction that they are explicitly asking you to classify.
Do not access external systems, bank accounts, or private records.
Use only the transaction information provided in the prompt.

Return exactly one valid JSON object and nothing else.

The JSON must contain exactly these fields:

{
  "category": string,
  "subcategory": string,
  "financial_type": string,
  "protected": boolean,
  "cut_priority": string,
  "recurring": boolean,
  "reimbursable": boolean,
  "shared": boolean,
  "one_time_exceptional": boolean,
  "count_as_spending": boolean,
  "confidence": string
}

Allowed category values:

Housing
Utilities
Groceries
Workday Food
Dining
Personal Travel
Work Travel
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
Other
Income
Reimbursement
Savings
Investments
Transfer

Allowed financial_type values:

Essential
Discretionary
Savings
Investment
Transfer
Income
Reimbursable
Shared

Allowed cut_priority values:

very_low
low
medium
high
very_high

Allowed confidence values:

high
medium
low

Definitions:

protected:
True when this expense should generally be preserved rather than targeted for cuts.

cut_priority:
How aggressively this expense should be considered for reduction.

recurring:
True when the transaction represents a repeating expense.

reimbursable:
True when someone else, such as an employer, is expected to reimburse it.

shared:
True when the expense is shared with another person.

one_time_exceptional:
True for an unusual or exceptional one-time expense rather than normal ongoing spending.

count_as_spending:
False for transfers, savings movements, investments, income, reimbursements,
or credit-card payments that would otherwise double count spending.

subcategory:
Choose a concise, specific description of the transaction.

Classify based on the information provided.
Do not invent facts that are not supported by the transaction.
"""


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


def load_base():
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    return model.eval()


def load_tuned():
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    model = PeftModel.from_pretrained(
        base,
        ADAPTER,
    )

    return model.eval()


def generate(model, transaction_text):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": transaction_text,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()


def parse_json(text):

    try:
        start = text.find("{")
        end = text.rfind("}") + 1

        if start == -1 or end <= start:
            return None

        return json.loads(text[start:end])

    except Exception:
        return None


def score(expected, predicted):

    if predicted is None:
        return 0, len(FIELDS), []

    correct = 0
    details = []

    for field in FIELDS:

        expected_value = expected.get(field)
        predicted_value = predicted.get(field)

        is_correct = expected_value == predicted_value

        correct += int(is_correct)

        details.append(
            {
                "field": field,
                "expected": expected_value,
                "predicted": predicted_value,
                "correct": is_correct,
            }
        )

    return correct, len(FIELDS), details


with open(TEST_FILE) as f:
    examples = [
        json.loads(line)
        for line in f
        if line.strip()
    ]


print(f"\nLoaded {len(examples)} held-out examples.")


def evaluate(model, label):

    total_correct = 0
    total_fields = 0
    results = []

    print("\n" + "=" * 75)
    print(label)
    print("=" * 75)

    for i, example in enumerate(examples, 1):

        transaction = example["messages"][0]["content"]

        expected = json.loads(
            example["messages"][1]["content"]
        )

        raw_output = generate(
            model,
            transaction,
        )

        predicted = parse_json(raw_output)

        correct, total, details = score(
            expected,
            predicted,
        )

        total_correct += correct
        total_fields += total

        accuracy = 100 * correct / total

        print(
            f"\nExample {i}: "
            f"{correct}/{total} = {accuracy:.1f}%"
        )

        print("\nTransaction:")
        print(transaction)

        print("\nExpected:")
        print(
            json.dumps(
                expected,
                indent=2,
            )
        )

        print("\nPredicted:")

        if predicted is not None:
            print(
                json.dumps(
                    predicted,
                    indent=2,
                )
            )
        else:
            print(raw_output)

        mismatches = [
            x for x in details
            if not x["correct"]
        ]

        if mismatches:

            print("\nMismatched fields:")

            for mismatch in mismatches:

                print(
                    f"  {mismatch['field']}: "
                    f"expected={mismatch['expected']!r}, "
                    f"predicted={mismatch['predicted']!r}"
                )

        results.append(
            {
                "example": i,
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
                "expected": expected,
                "predicted": predicted,
                "raw": raw_output,
                "details": details,
            }
        )

    overall_accuracy = (
        100 * total_correct / total_fields
    )

    print("\n" + "-" * 75)

    print(
        f"{label} OVERALL: "
        f"{total_correct}/{total_fields} "
        f"= {overall_accuracy:.1f}%"
    )

    return overall_accuracy, results


print("\nLoading BASE model...")

base_model = load_base()

base_accuracy, base_results = evaluate(
    base_model,
    "BASE NEMOTRON 8B",
)


del base_model

torch.cuda.empty_cache()


print("\nLoading TUNED model...")

tuned_model = load_tuned()

tuned_accuracy, tuned_results = evaluate(
    tuned_model,
    "TUNED NEMOTRON 8B + LoRA",
)


print("\n" + "=" * 75)
print("FINAL FAIR COMPARISON")
print("=" * 75)

print(
    f"Base accuracy:   {base_accuracy:.1f}%"
)

print(
    f"Tuned accuracy:  {tuned_accuracy:.1f}%"
)

print(
    f"Improvement:     "
    f"{tuned_accuracy - base_accuracy:+.1f} percentage points"
)


output = {
    "evaluation": "shared_explicit_schema_prompt",
    "num_examples": len(examples),
    "fields_per_example": len(FIELDS),
    "base_accuracy": base_accuracy,
    "tuned_accuracy": tuned_accuracy,
    "improvement_pp": tuned_accuracy - base_accuracy,
    "base_results": base_results,
    "tuned_results": tuned_results,
}


with open(
    OUTPUT_FILE,
    "w",
) as f:

    json.dump(
        output,
        f,
        indent=2,
    )


print(
    f"\nResults saved to {OUTPUT_FILE}"
)
