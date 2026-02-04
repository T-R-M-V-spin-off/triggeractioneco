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

# --- Configurazione Hardware (Ottimizzata per RTX 2060 6GB) ---
BATCH_SIZE = 2  # Teniamo basso per sicurezza VRAM
GRAD_ACC = 32  # Accumuliamo molto per avere un batch effettivo stabile
LR = 5e-5  # Learning rate standard per LoRA
EPOCHS = 3  # Un'epoca in più per compensare il batch piccolo
MAX_LEN = 256  # Ridotto da 256: le tue regole sono brevi, questo salva VRAM
MAX_PROMPT_LEN = 128
SEED = 42

USE_8BIT = True
WARMUP_RATIO = 0.1  # AGGIUNTO: Warmup per stabilità

set_seed(SEED)

# ============================
# 1. LOAD & PREPROCESS DATASET
# ============================

print("📂 Loading dataset...")
# Carica il dataset grezzo
raw_dataset = load_dataset("json", data_files=DATA_PATH, split="train")
print(f"✅ Loaded {len(raw_dataset)} raw samples")


# --- MODIFICA CRITICA: AGGIUNTA ISTRUZIONE ---
def add_instruction(samples):
    # Aggiungiamo un prefisso chiaro affinché T5 sappia cosa fare
    instruction = "Rewrite the following smart home rule to be eco-friendly and energy efficient:\n\n"

    return {
        "prompt": [instruction + p for p in samples["prompt"]],
        "chosen": samples["chosen"],
        "rejected": samples["rejected"]
    }


print("🔧 Injecting instructions into prompts...")
dataset = raw_dataset.map(add_instruction, batched=True)

# Esempio di controllo per vedere se l'istruzione è stata aggiunta
print(f"👀 Example input: {dataset[0]['prompt'][:100]}...")

# Split Train/Test
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
# 3. LOAD MODEL (Quantized)
# ============================

print("🧠 Loading model...")

bnb_config = None
if USE_8BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# Preparazione per il training k-bit (riduce l'uso di memoria)
if USE_8BIT:
    model = prepare_model_for_kbit_training(model)

# Disabilita cache per il training (risparmia VRAM)
model.config.use_cache = False

# ============================
# 4. LORA CONFIG
# ============================

print("🧩 Applying LoRA...")
lora_config = LoraConfig(
    r=32,  # Ridotto a 16 per salvare memoria sulla 2060
    lora_alpha=64,
    target_modules=["q", "v"],  # Target standard per T5
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM  # Importante per T5
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================
# 5. DPO CONFIGURATION
# ============================

dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,

    # --- PARAMETRI CRITICI PER T5 ---
    # INDISPENSABILE: dice al trainer che non è un GPT
    beta=0.25,  # Valore standard, permette al modello di imparare lo stile
    # --------------------------------

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
    eval_steps=125,  # AGGIUNTO

    # Ottimizzazioni Hardware
    fp16=True,  # Mixed Precision
    gradient_checkpointing=True,  # CRITICO per 6GB VRAM: rallenta un po' ma salva molta memoria

    # Logging & Saving
    logging_steps=50,
    save_strategy="steps",
    save_steps=125,  # RIDOTTO da 500
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
try:
    trainer.train()
except Exception as e:
    print(f"\n❌ ERRORE DURANTE IL TRAINING: {e}")
    print("Suggerimento: Se è un errore di memoria (OOM), prova ad abbassare MAX_LEN a 64 o r a 8.")
    exit()

# ============================
# 8. SAVE
# ============================

print("💾 Saving final model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Training completed! Model saved in {OUTPUT_DIR}")
print("Per usare il modello, ricorda di usare lo stesso prefisso nel prompt:")
print("'Rewrite the following smart home rule to be eco-friendly and energy efficient: IF ...'")