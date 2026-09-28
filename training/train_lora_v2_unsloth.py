import os
import torch

from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "/workspace/personal_cfo/models/nemotron8b"

TRAIN_FILE = (
    "/workspace/personal_cfo/lora_v2/"
    "personal_cfo_lora_v2_clean_train.jsonl"
)

VALIDATION_FILE = (
    "/workspace/personal_cfo/lora_v2/"
    "personal_cfo_lora_v2_clean_validation.jsonl"
)

OUTPUT_DIR = (
    "/workspace/personal_cfo/"
    "models/personal-cfo-lora-v2-clean-unsloth"
)

MAX_SEQ_LENGTH = 1024
SEED = 3407


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("Loading dataset")
print("=" * 70)

dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VALIDATION_FILE,
    },
)

print(dataset)
print("Train examples:", len(dataset["train"]))
print("Validation examples:", len(dataset["validation"]))


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("Loading Nemotron 8B")
print("=" * 70)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)


# ============================================================
# ADD LoRA
# ============================================================

print()
print("=" * 70)
print("Adding LoRA")
print("=" * 70)

model = FastLanguageModel.get_peft_model(
    model,
    r=8,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],

    lora_alpha=32,
    lora_dropout=0,
    bias="none",

    use_gradient_checkpointing="unsloth",

    random_state=SEED,

    use_rslora=False,
    loftq_config=None,
)


# ============================================================
# FORMAT CHAT DATA
# ============================================================

def format_example(example):

    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

    return {
        "text": text
    }


print()
print("=" * 70)
print("Formatting examples")
print("=" * 70)

train_dataset = dataset["train"].map(
    format_example,
    remove_columns=dataset["train"].column_names,
)

validation_dataset = dataset["validation"].map(
    format_example,
    remove_columns=dataset["validation"].column_names,
)

print()
print("Sample formatted record:")
print()
print(train_dataset[0]["text"])


# ============================================================
# TRAINING CONFIG
# ============================================================

training_args = SFTConfig(

    output_dir=OUTPUT_DIR,

    max_seq_length=MAX_SEQ_LENGTH,

    per_device_train_batch_size=2,

    per_device_eval_batch_size=2,

    gradient_accumulation_steps=4,

    num_train_epochs=2,

    learning_rate=2e-4,

    warmup_ratio=0.05,

    lr_scheduler_type="cosine",

    optim="adamw_8bit",

    weight_decay=0.01,

    logging_steps=5,

    eval_strategy="epoch",

    save_strategy="epoch",

    save_total_limit=2,

    bf16=torch.cuda.is_bf16_supported(),

    fp16=not torch.cuda.is_bf16_supported(),

    seed=SEED,

    report_to="none",

    packing=False,
)


# ============================================================
# TRAINER
# ============================================================

trainer = SFTTrainer(
    model=model,

    tokenizer=tokenizer,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    dataset_text_field="text",

    args=training_args,
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("Starting clean LoRA v2 training")
print("=" * 70)

train_result = trainer.train()

print()
print("=" * 70)
print("Training complete")
print("=" * 70)

print(train_result)


# ============================================================
# SAVE FINAL ADAPTER
# ============================================================

FINAL_ADAPTER_DIR = os.path.join(
    OUTPUT_DIR,
    "final_adapter",
)

print()
print(
    "Saving adapter to:",
    FINAL_ADAPTER_DIR,
)

model.save_pretrained(
    FINAL_ADAPTER_DIR
)

tokenizer.save_pretrained(
    FINAL_ADAPTER_DIR
)

print()
print("=" * 70)
print("Saved")
print("=" * 70)

print(FINAL_ADAPTER_DIR)
