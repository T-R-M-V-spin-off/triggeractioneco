import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


# =========================
# CONFIG
# =========================

BASE_MODEL = "google/flan-t5-base"   # o large se usavi quello
LORA_PATH = "./flan_t5_dpo_eco"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# LOAD MODEL
# =========================

print("🔄 Loading base model...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="auto"
)

print("🔄 Loading LoRA adapters...")

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH
)

model.eval()

print("✅ Model + LoRA loaded")


# =========================
# PROMPT
# =========================

def build_prompt(rule):

    return f"""
You are an expert in eco-friendly IoT automation.

Rewrite the following rule to be more energy efficient.

Original rule:
{rule}

Eco-friendly version:
""".strip()


# =========================
# GENERATION
# =========================

def generate(rule):

    prompt = build_prompt(rule)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=256,
        truncation=True
    ).to(model.device)

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1
        )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )


# =========================
# TEST RULES
# =========================

NEUTRAL_RULES = [

    "IF motion is detected THEN turn on the living room lights",
    "IF temperature is above 26 degrees THEN turn on the air conditioner",
    "IF it is 7 AM THEN start the coffee machine",
    "IF door is opened THEN turn on hallway lights",
    "IF humidity is below 30 percent THEN turn on the humidifier",
    "IF sunset occurs THEN turn on garden lights",
    "IF smoke is detected THEN activate ventilation",
    "IF window is opened THEN turn off heater",
    "IF brightness is low THEN turn on desk lamp",
    "IF washing machine finishes THEN send notification",

    "IF presence is detected in bedroom THEN turn on fan",
    "IF rain is detected THEN close roof windows",
    "IF power consumption exceeds threshold THEN send alert",
    "IF TV is turned off THEN turn off sound system",
    "IF alarm is triggered THEN turn on all lights",

    "IF air quality is poor THEN start air purifier",
    "IF fridge door is open THEN send reminder",
    "IF battery is low THEN enable power saving",
    "IF user arrives home THEN turn on heating",
    "IF midnight is reached THEN turn off devices"
]


# =========================
# RUN
# =========================

print("\n🌱 Running tests...\n")

for i, rule in enumerate(NEUTRAL_RULES, 1):

    eco = generate(rule)

    print(f"========== TEST {i} ==========")
    print("NEUTRAL:")
    print(rule)

    print("\nECO:")
    print(eco)

    print("=" * 60 + "\n")


print("✅ Done")
