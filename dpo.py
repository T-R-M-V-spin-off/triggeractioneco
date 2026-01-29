import torch
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model

# Sostituisci o aggiungi DPOConfig
from trl import DPOTrainer, DPOConfig

# ==================================================
# CONFIG
# ==================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./eco_dpo_model"

MAX_LEN = 256
EPOCHS = 3
LR = 2e-5

# ==================================================
# LOAD DATASET
# ==================================================

print("Loading dataset...")

dataset = load_dataset(
    "csv",
    data_files="output_rules_modified.csv"
)["train"]

def format_example(example):
    return {
        "prompt": example["rule"],
        "chosen": example["eco_rule"],
        "rejected": example["non_eco_rule"],
    }

dataset = dataset.map(
    format_example,
    remove_columns=dataset.column_names,
)

print("Dataset size:", len(dataset))

# ==================================================
# TOKENIZER
# ==================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = None  # pulisce warning template

# ==================================================
# 8-BIT CONFIG
# ==================================================

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

# ==================================================
# LOAD MODEL
# ==================================================

print("Loading model (8-bit)...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# ==================================================
# LORA
# ==================================================

print("Applying LoRA...")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# ==================================================
# TRAINING ARGUMENTS
# ==================================================

print("Preparing training args...")

print("Preparing training args...")

training_args = DPOConfig( # <--- Usa DPOConfig qui
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    gradient_checkpointing=True,
    disable_tqdm=False,
    max_length=MAX_LEN, # DPOConfig preferisce avere i parametri di lunghezza qui
    max_prompt_length=MAX_LEN // 2,
)


# ==================================================
# DPO TRAINER
# ==================================================

print("Building DPO trainer...")

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    train_dataset=dataset,
    args=training_args
)

# ==================================================
# TRAIN
# ==================================================

print("\n🚀 Starting DPO training...\n")

trainer.train()

# ==================================================
# SAVE
# ==================================================

print("\nSaving model...")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n✅ Training completato con successo!")