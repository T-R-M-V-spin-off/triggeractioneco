import pandas as pd
import os
import re


# ===============================
# CONFIG
# ===============================

INPUT_FILE = "../../../../dataset/filtered_data/device_category_data/Air_Climate.csv"


# ===============================
# NORMALIZATION RULES
# ===============================

TRIGGER_RULES = [
    (r".*calendar.*event.*specific keyword.*minutes before.*", "IF a set number of minutes before a calendar event with a specific keyword"),
    (r".*calendar.*event.*specific keyword.*", "IF a set number of minutes before a calendar event with a specific keyword"),
    (r".*before.*time-of-day.*peak rates.*start.*", "IF before time-of-day peak rates start"),
    (r".*every day at a specified time.*", "IF every day at a specified time"),
    (r".*message.*key phrase.*", "IF a message containing a key phrase is sent"),
    (r".*home.*set to away.*", "IF every time no one is at home"),
    (r".*home.*set to home.*", "IF every time someone is at home"),
    (r".*user enters a specified area.*", "IF the user enters a specified area"),
    (r".*user exits a specified area.*", "IF the user exits a specified area"),
    (r".*button.*pressed.*", "IF a button is pressed"),
    (r".*voice assistant.*activated.*", "IF a voice assistant is activated with a specific phrase"),
    (r".*a/c unit.*temperature.*above.*value.*", "IF an AC unit detects temperature above a specified value"),
    (r".*a/c unit.*temperature.*below.*value.*", "IF an AC unit detects temperature below a specified value"),
    (r".*android device.*connects to wifi.*", "IF an Android device connects to a specified Wi-Fi network"),
    (r".*android device.*disconnects from wifi.*", "IF an Android device disconnects from a specified Wi-Fi network"),
    (r".*fitbit.*logs.*sleep.*", "IF a device logs new sleep data"),
    (r".*myfox.*security system.*partially armed.*", "IF a security system is partially armed"),
    (r".*smartthings.*device.*switched off.*", "IF a smart device is switched off"),
    (r".*smartthings.*device.*switched on.*", "IF a smart device is switched on"),
    (r".*routine.*activated.*", "IF a routine is activated"),
    (r".*air purifier.*air quality.*", "IF an air purifier detects a specific air quality level"),
    (r".*room temperature.*condition.*threshold.*", "IF the room temperature satisfies a threshold condition"),
    (r".*solar power.*drops below.*", "IF solar power drops below a specified value"),
    (r".*temperature.*specific device.*exceeds.*threshold.*", "IF a device temperature exceeds a threshold"),
    (r".*temperature.*satisfies.*threshold.*", "IF the room temperature satisfies a threshold condition"),
    (r".*device.*temperature.*above.*threshold.*", "IF a device detects temperature above a specified threshold"),
    (r".*device.*temperature.*below.*threshold.*", "IF a device detects temperature below a specified threshold"),
    (r".*indoor temperature.*dropping below.*threshold.*", "IF indoor temperature drops below a specified threshold"),
    (r".*indoor temperature.*rising above.*threshold.*", "IF indoor temperature rises above a specified threshold"),
    (r".*sunrise.*", "IF a set number of minutes before sunrise"),
    (r".*sunset.*", "IF a set number of minutes before sunset"),
    (r".*unit.*turned on.*", "IF a set number of minutes after a unit has been turned on"),
    (r".*current weather condition.*rain|snow|cloudy|clear.*", "IF the weather conditions change"),
    (r".*local humidity.*above.*value.*", "IF local humidity and fires when above a specified value"),
    (r".*local temperature.*drops below.*value.*", "IF local temperature and fires when below a specified value"),
    (r".*local temperature.*rises above.*value.*", "IF local temperature and fires when above a specified value"),
    (r".*room enters manual mode.*", "IF every time a room enters manual mode")
]


ACTION_RULES = [
    (r".*changes modes.*auto.*sleep.*", ", THEN change the AC unit mode to sleep."),
    (r".*turns off.*air purifier.*", ", THEN turns off the air purifier."),
    (r".*turns on.*air purifier.*", ", THEN turns on the air purifier."),
    (r".*turns.*display.*on or off.*", ", THEN turns the display on."),
    (r".*turns.*fan.*speed.*low.*medium.*high.*", ", THEN sets the fan to a specified speed."),
    (r".*disable.*timer.*", ", THEN disable the indicated timer."),
    (r".*enable.*timer.*", ", THEN enable the indicated timer."),
    (r".*econo mode.*enable|disable.*", ", THEN disable eco mode on the AC unit."),
    (r".*holiday mode.*enable|disable.*", ", THEN disable holiday mode on AC units."),
    (r".*execute.*scene.*", ", THEN set the mode of the air conditioner to turbo."),
    (r".*set.*mode.*air conditioner.*", ", THEN set the mode of the air conditioner to eco."),
    (r".*turn off.*air conditioner.*", ", THEN turn off the air conditioner."),
    (r".*turn off.*a/c.*specified room.*", ", THEN turn off the air conditioner in a specified room."),
    (r".*turn off.*daikin.*ac unit.*", ", THEN turn off the AC unit."),
    (r".*turn on.*air conditioner.*", ", THEN turn on the air conditioner."),
    (r".*turn on.*a/c.*specified room.*comfort mode.*", ", THEN turn on the air conditioner in a specified room in comfort mode."),
    (r".*turn on.*daikin.*ac unit.*", ", THEN turn on the AC unit."),
    (r".*turn.*intesishome.*a/c.*off.*", ", THEN turn off the AC unit."),
    (r".*turn.*intesishome.*a/c.*on.*", ", THEN turn on the AC unit.")
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
