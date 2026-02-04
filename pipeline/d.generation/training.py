import torch
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    set_seed,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

# ============================
# CONFIGURAZIONE
# ============================

MODEL_NAME = "google/flan-t5-base"
DATA_PATH = "dpo_dataset.jsonl"
OUTPUT_DIR = "./flan_t5_dpo_eco"

# Configurazione Hardware (RTX 2060 6GB)
BATCH_SIZE = 2  # Aumentato leggermente
GRAD_ACC = 16  # Ridotto per aggiornamenti più frequenti (batch effettivo = 32)
LR = 5e-5  # Aumentato per convergenza più veloce
EPOCHS = 2  # Aumentato per 200k samples
MAX_LEN = 256
MAX_PROMPT_LEN = 128
SEED = 42

USE_8BIT = True
WARMUP_RATIO = 0.1  # AGGIUNTO: Warmup per stabilità

set_seed(SEED)

# ============================
# 1. LOAD DATASET
# ============================

print("📂 Loading dataset...")
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
print(f"✅ Loaded {len(dataset)} samples")

# SPLIT TRAIN/VALIDATION (CRITICO)
dataset = dataset.train_test_split(test_size=0.1, seed=SEED)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]
print(f"📊 Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

# ============================
# 2. TOKENIZER
# ============================

print("🔤 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ============================
# 3. LOAD MODEL
# ============================

print("🧠 Loading model...")

bnb_config = None
if USE_8BIT:
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        print("❄️  8-bit quantization enabled.")
    except Exception as e:
        print(f"⚠️  BitsAndBytes error: {e}. Loading in FP16/FP32.")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

if USE_8BIT:
    model = prepare_model_for_kbit_training(model)

model.config.use_cache = False

# ============================
# 4. LORA CONFIG
# ============================

print("🧩 Applying LoRA...")
lora_config = LoraConfig(
    r=32,  # AUMENTATO da 16
    lora_alpha=64,  # AUMENTATO proporzionalmente
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================
# 5. DPO CONFIGURATION
# ============================

dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,

    # Parametri DPO
    beta=0.4,  # AUMENTATO da 0.1 (critico!)
    max_length=MAX_LEN,
    max_prompt_length=MAX_PROMPT_LEN,

    # Parametri Training
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,  # AGGIUNTO
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    warmup_ratio=WARMUP_RATIO,  # AGGIUNTO

    # Evaluation
    eval_strategy="steps",  # AGGIUNTO
    eval_steps=250,  # AGGIUNTO

    # Ottimizzazioni
    fp16=True,
    gradient_checkpointing=True,
    logging_steps=50,
    save_strategy="steps",
    save_steps=250,  # RIDOTTO da 500
    save_total_limit=3,  # AUMENTATO
    load_best_model_at_end=True,  # AGGIUNTO
    metric_for_best_model="eval_loss",  # AGGIUNTO
    greater_is_better=False,  # AGGIUNTO
    remove_unused_columns=False,
    report_to="tensorboard",  # CAMBIATO da "none"
    optim="paged_adamw_32bit"
)

# ============================
# 6. TRAINER
# ============================

print("🚀 Initializing DPO Trainer...")

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_config,
    train_dataset=train_dataset,  # CAMBIATO
    eval_dataset=eval_dataset,  # AGGIUNTO
    processing_class=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # AGGIUNTO
)

# ============================
# 7. TRAINING
# ============================

print("🏋️ Training started...")
trainer.train()

# ============================
# 8. SAVE
# ============================

print("💾 Saving final model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Training completed! Model saved in {OUTPUT_DIR}")