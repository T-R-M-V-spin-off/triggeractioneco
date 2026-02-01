import torch
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    set_seed
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
BATCH_SIZE = 1  # Basso per risparmiare VRAM
GRAD_ACC = 32  # Alto per compensare il batch size (Batch effettivo = 32)
LR = 1e-5  # DPO richiede learning rate bassi e stabili
EPOCHS = 1  # DPO converge in fretta, spesso 1 epoca basta
MAX_LEN = 256
MAX_PROMPT_LEN = 128
SEED = 42

# Windows workaround per BitsAndBytes
# Se hai installato: pip install bitsandbytes-windows
USE_8BIT = True

set_seed(SEED)

# ============================
# 1. LOAD DATASET
# ============================

print("📂 Loading dataset...")
# DPO richiede colonne: "prompt", "chosen", "rejected"
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
print(f"✅ Loaded {len(dataset)} samples")

# ============================
# 2. TOKENIZER
# ============================

print("🔤 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ============================
# 3. LOAD MODEL
# ============================

print("🧠 Loading model...")

# Configurazione quantizzazione (opzionale ma consigliata per 6GB VRAM)
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

try:
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

# Preparazione per il training (importante per LoRA + Quantizzazione)
if USE_8BIT:
    model = prepare_model_for_kbit_training(model)

# Disabilita cache per risparmiare VRAM durante il training (riattivala in inferenza)
model.config.use_cache = False

# ============================
# 4. LORA CONFIG
# ============================

print("🧩 Applying LoRA...")
lora_config = LoraConfig(
    r=16,  # Aumentato leggermente per migliore capacità
    lora_alpha=32,
    target_modules=["q", "v"],  # Corretto per T5
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# == == == == == == == == == == == == == ==
# 5. DPO CONFIGURATION
# ============================

dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,

    # Parametri DPO
    beta=0.1,
    max_length=MAX_LEN,
    max_prompt_length=MAX_PROMPT_LEN,
    # RIMOSSO: is_encoder_decoder=True (non va qui!)

    # Parametri Training
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    num_train_epochs=EPOCHS,

    # Ottimizzazioni
    fp16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none",
    optim="paged_adamw_32bit"
)

# ============================
# 6. TRAINER
# ============================

print("🚀 Initializing DPO Trainer...")

# FIX per trl >= 0.12.0:
# 1. 'tokenizer' diventa 'processing_class'
# 2. Rimuoviamo 'is_encoder_decoder' (lo rileva automaticamente dal modello)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer  # <--- CAMBIATO QUI (era tokenizer=tokenizer)
)

# ============================
# 7. TRAINING
# ============================

print("🏋️ Training started...")
trainer.train()

# ============================
# 8. SAVE
# ============================

print("💾 Saving adapters...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Training completed! Model saved in {OUTPUT_DIR}")