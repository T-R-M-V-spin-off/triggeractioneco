import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ================= CONFIG =================
MODEL_PATH = "flan_t5_dpo_final"  # La cartella creata dallo script precedente
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Caricamento modello da {MODEL_PATH} su {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

# ================= REGOLE DA TESTARE =================
# Qui metti delle regole NEUTRE per vedere come le trasforma
test_rules = [
    "IF temperature drops below 20 THEN turn on heater",
    "IF motion is detected THEN turn on lights",
    "IF window opens THEN close AC",
    "IF humidity rises above 70 THEN turn on dehumidifier",
    "IF presence is detected THEN turn on fan"
]

print("\n=== RISULTATI GENERAZIONE ===")

for neutral in test_rules:
    # Preparazione input
    inputs = tokenizer(neutral, return_tensors="pt", max_length=128, truncation=True).to(DEVICE)

    # Generazione
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=5,  # Beam search per qualità migliore
            do_sample=False,  # Deterministico (meglio per riproducibilità)
            early_stopping=True,
            repetition_penalty=1.2  # Evita che ripeta parole
        )

    # Decodifica output
    generated_eco = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Neutral:  {neutral}")
    print(f"Eco (Gen): {generated_eco}")
    print("-" * 50)