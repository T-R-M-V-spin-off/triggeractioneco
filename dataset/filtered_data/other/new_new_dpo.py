import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

# ================= CONFIGURAZIONE =================
INPUT_FILE = "final_with_variants.csv"
MODEL_SAVE_DIR = "flan_t5_dpo_final"  # Dove salvare il modello finito
PRETRAINED_MODEL = "google/flan-t5-base"
BATCH_SIZE = 4   # Abbassa a 1 o 2 se hai errori di memoria (CUDA OOM)
EPOCHS = 3
LR = 1e-6        # Learning rate basso per DPO
MAX_LEN = 128

# ================= CARICAMENTO DATI =================
df = pd.read_csv(INPUT_FILE)

# Pulizia: rimuoviamo righe che hanno valori nulli in una delle 3 colonne
df = df.dropna(subset=["neutral_rule", "eco_rule", "noneco_rule"])

# Mappatura per DPO:
# Prompt   -> La regola di partenza (neutral)
# Chosen   -> La regola che vogliamo (eco)
# Rejected -> La regola che NON vogliamo (noneco)
dataset_dict = {
    "prompt": df["neutral_rule"].tolist(),
    "chosen": df["eco_rule"].tolist(),
    "rejected": df["noneco_rule"].tolist(),
}
hf_dataset = Dataset.from_dict(dataset_dict)

print(f"Dataset caricato: {len(hf_dataset)} esempi.")

# ================= MODELLO & TOKENIZER =================
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAINED_MODEL)

# ================= CONFIG TRAINER =================
training_args = DPOConfig(
    output_dir=MODEL_SAVE_DIR,
    beta=0.1,                        # Quanto pesare la differenza tra chosen e rejected
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    gradient_accumulation_steps=4,   # Accumula gradienti per simulare batch più grandi
    save_strategy="epoch",           # Salva alla fine di ogni epoca
    logging_steps=10,
    fp16=torch.cuda.is_available(),  # Usa FP16 se hai la GPU
    remove_unused_columns=False,
    is_encoder_decoder=True,         # OBBLIGATORIO per T5
)

# ================= AVVIO TRAINER =================
trainer = DPOTrainer(
    model=model,
    ref_model=None,             # Crea automaticamente una copia del modello originale come riferimento
    args=training_args,
    train_dataset=hf_dataset,
    tokenizer=tokenizer,
    max_length=MAX_LEN,
    max_prompt_length=MAX_LEN,
    max_target_length=MAX_LEN,
)

print("Inizio addestramento DPO...")
trainer.train()

# ================= SALVATAGGIO =================
print(f"Salvataggio modello in {MODEL_SAVE_DIR}...")
trainer.save_model(MODEL_SAVE_DIR)
tokenizer.save_pretrained(MODEL_SAVE_DIR)
print("Fatto!")