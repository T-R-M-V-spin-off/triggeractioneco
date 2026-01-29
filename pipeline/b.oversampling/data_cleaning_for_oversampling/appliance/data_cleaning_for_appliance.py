import pandas as pd
import os
import re


# ===============================
# CONFIG
# ===============================

INPUT_FILE = "../../../../dataset/filtered_data/device_category_data/Appliance.csv"


# ===============================
# NORMALIZATION RULES
# ===============================

TRIGGER_RULES = [
    (r".*device.*turned on.*|device turned on", "IF a device is turned on"),
    (r".*electricity price.*lowest.*", "IF electricity prices are low"),
    (r".*electricity.*cheap.*", "IF electricity prices are low"),
    (r".*(smoke detector|smoke alarm|dangerous smoke).*", "IF a smoke detector detects smoke"),
    (r".*every.*day.*specific time.*", "IF every day at a specified time"),
    (r".*button.*pressed.*|.*press.*button.*|flic.*", "IF a button is pressed"),
    (r".*to do list.*edit.*|.*edit.*to do list.*", "IF a to-do list item is edited"),
    (r".*enter or exit.*area.*", "IF the user enters or exits a specified area"),
    (r".*enter.*area.*", "IF the user enters a specified area"),
    (r".*exit.*area.*", "IF the user exits a specified area"),
    (r".*(alexa trigger|ok google|google assistant).*", "IF a voice assistant is activated with a specific phrase"),
    (r".*cookit.*finished.*", "IF a smart cooking device finishes successfully"),
    (r".*cookit.*turned on.*", "IF a smart cooking device is turned on"),
    (r".*binary switch.*turned on.*", "IF a binary switch is turned on"),
    (r".*freezer.*door.*open.*", "IF a freezer door is opened"),
    (r".*sleep.*out of bed.*", "IF a sleep tracking device detects the user leaving the bed"),
    (r".*specific days.*week.*time.*", "IF on specific days of the week at a specified time"),
    (r".*(homeseer|knx).*turned on.*", "IF a connected device is turned on."),
    (r".*security panel.*armed|disarmed.*", "IF a security system is armed or disarmed"),
    (r".*co2.*intensity.*lowest.*", "IF carbon intensity is lowest within a time window"),
    (r".*exchanged power.*threshold.*", "IF power usage satisfies a threshold condition"),
    (r".*(telegram|text message|ifttt).*", "IF a message containing a key phrase is sent"),
    (r".*google calendar.*event.*", "IF shortly before a calendar event with a specific keyword"),
    (r".*sunrise.*", "IF shortly before sunrise at the user's location"),
    (r".*lawnmower.*button.*", "IF a lawn mower button is pressed")
]

ACTION_RULES = [
    (r".*deactivate.*eco mode.*", ", THEN deactivates eco mode on the appliance."),
    (r".*deactivate.*fresh mode.*", ", THEN deactivates fresh mode on the appliance."),
    (r".*deactivate.*vacation mode.*", ", THEN deactivates vacation mode on the appliance."),
    (r".*preheat.*oven.*hot air.*fast preheat.*", ", THEN preheats the oven using the hot air program with fast preheat."),
    (r".*start.*dishwasher.*currently selected.*", ", THEN starts the currently selected dishwasher program."),
    (r".*start.*coffee milk mix.*selected.*", ", THEN starts the selected coffee milk mix program with preferred settings."),
    (r".*start.*coffee program.*selected.*", ", THEN starts the selected coffee program with preferred settings."),
    (r".*start.*coffee world.*selected.*", ", THEN starts the selected coffee world program with preferred settings."),
    (r".*start.*dishwasher.*selected.*", ", THEN starts the selected dishwasher program."),
    (r".*start.*hood.*selected.*", ", THEN starts the selected hood program."),
    (r".*turn off.*coffee machine.*", ", THEN turns off a coffee machine."),
    (r".*turn off.*dishwasher.*", ", THEN turns off the dishwasher."),
    (r".*turn off.*hood.*", ", THEN turns off the hood."),
    (r".*turn off.*oven.*", ", THEN turns off then oven."),
    (r".*turn on.*coffee machine.*", ", THEN turns on the coffee machine."),
    (r".*turn on.*dishwasher.*", ", THEN turns on the dishwasher."),
    (r".*turn on.*hood.*default program.*", ", THEN turns on the hood with the default program."),
    (r".*turn on.*oven.*", ", THEN turns on the oven.")
]

# ===============================
# NORMALIZATION FUNCTION
# ===============================

def normalize_trigger(text):

    if pd.isna(text):
        return text

    text = str(text).lower().strip()

    for pattern, normalized in TRIGGER_RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return normalized

    # Non matchato
    return text.capitalize()


# ===============================
# PROCESS
# ===============================

df = pd.read_csv(INPUT_FILE)

if "triggerDesc" not in df.columns:
    raise ValueError("Colonna 'triggerDesc' non trovata")

# Backup per controllo
original_triggers = df["triggerDesc"].copy()

# Normalizza
df["triggerDesc"] = df["triggerDesc"].apply(normalize_trigger)

# Output file
folder = os.path.dirname(INPUT_FILE)
base = os.path.basename(INPUT_FILE).replace(".csv", "")

output_file = os.path.join(
    folder,
    f"{base}_normalized_for_oversampling.csv"
)

df.to_csv(output_file, index=False)

print(f"File creato: {output_file}")


# ===============================
# VALIDATION
# ===============================

# Frasi normalizzate valide
valid_normalized = set([rule[1] for rule in TRIGGER_RULES])

# Trigger finali unici
final_triggers = set(df["triggerDesc"].dropna().unique())

# Trigger NON normalizzati
not_normalized = final_triggers - valid_normalized

print("\n========== CONTROLLO NORMALIZZAZIONE ==========")

if not not_normalized:
    print("Tutti i trigger sono stati normalizzati correttamente ✅")
else:
    print(f"Trovati {len(not_normalized)} trigger NON normalizzati ❌:\n")

    for t in sorted(not_normalized):
        print(f"- {t}")

print("==============================================")
