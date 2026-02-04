import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

MODEL_PATH = "./flan_t5_dpo_eco"
BASE_MODEL = "google/flan-t5-base"

print("🔄 Caricamento del tuo modello ECO...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()


def generate_eco_rule(rule):
    instruction = "Rewrite the following smart home rule to be eco-friendly and energy efficient:\n\n"
    input_text = instruction + rule

    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


test_suite = {
    "NON_ECO": [
        "IF the window is open, THEN set the heater to 28 degrees.",
        "IF the room is empty, THEN keep the 4K TV on for background noise.",
        "IF it is raining, THEN start the garden sprinklers at maximum pressure.",
        "IF the sun is shining directly inside, THEN turn on the AC to 16 degrees instead of closing curtains.",
        "IF I leave for a 2-week vacation, THEN keep all water heaters at maximum temperature.",
        "IF the dishwasher is 20% full, THEN start a high-temperature intensive wash.",
        "IF it is midnight, THEN turn on all architectural outdoor floodlights until morning.",
        "IF the smartphone is at 100% battery, THEN continue fast charging all night.",
        "IF the balcony door is open, THEN set the air purifier to Turbo mode.",
        "IF the hallway is empty, THEN keep the lights at 100% brightness indefinitely."
    ],
    "NEUTRE": [
        "IF the motion sensor detects a person, THEN turn on the light.",
        "IF the temperature is below 18 degrees, THEN turn on the heating.",
        "IF it is 7:00 AM, THEN open the motorized curtains.",
        "IF the smoke detector is activated, THEN sound the alarm.",
        "IF the doorbell rings, THEN show the camera feed on the smart display.",
        "IF the indoor CO2 level is high, THEN open the ventilation valve.",
        "IF it is sunset, THEN close the shutters.",
        "IF the washing machine finishes its cycle, THEN send a notification to the phone.",
        "IF the water leak sensor is triggered, THEN shut off the main valve.",
        "IF the humidity is above 60%, THEN turn on the dehumidifier."
    ]
}

print("--- 🧪 INIZIO VALIDAZIONE MODELLO ---")

for category, rules in test_suite.items():
    print(f"\n=== CATEGORIA: {category} ===")
    for rule in rules:
        result = generate_eco_rule(rule)
        print(f"\n[IN] : {rule}")
        print(f"[OUT]: {result}")

print("\n--- ✅ TEST COMPLETATO ---")