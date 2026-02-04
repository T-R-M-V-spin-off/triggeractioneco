# ================= CONFIG =================
import os
import pandas as pd

# ================= CONFIG =================
import os
import pandas as pd

INPUT_FOLDER = "../../dataset/filtered_data/device_category_data/"
OUTPUT_FILE = "../../dataset/filtered_data/ultra_mega_final.csv"

CSV_FILES = [
    "Air_Climate_normalized_for_oversampling_oversampled_final.csv",
    "Appliance_normalized_for_oversampling_oversampled_final.csv",
    "Generic_Other_normalized_for_oversampling_oversampled_final.csv",
    "Light_normalized_for_oversampling_oversampled_final.csv",
    "Thermostat_Heating_normalized_for_oversampling_oversampled_final.csv",
]

# ================= ACTION SETS =================

# Questi sono esempi, li puoi adattare alle tue liste complete
ACTION_X = [
    ["eco mode", "comfort mode", "turbo mode"],
    ["low power mode", "default mode", "maximum power"],
    ["energy saving mode", "auto mode", "boost mode"]
]

ACTION_Y = [
    ["for a specific amount of time", "for a while", "for the entire day"],
    ["for a limited time", "until stopped", "indefinitely"],
    ["temporarily", "until manually changed", "continuously"],
    ["temporarily", "indefinitely", "continuously"],
    ["for a limited time", "until stopped", "until manually changed"],
    ["for the entire day", "for a specific amount of time", "for a fixed delay"],

]

ACTION_Z = [
    ["at low value", "at medium value", "at maximum value"],
    ["at minimum value ", "at default value", "at high value"],
    ["to power saving mode", "at moderate value", "at boosted value"],
    ["low", "medium", "high"],
    ["minimum", "moderate", "maximum"],
    ["silent", "default", "boost"],
    ["at low power", "at medium power", "at maximum power"],
    ["at minimum power ", "at default power", "at high power"],
    ["to power saving mode", "at moderate power", "at boosted power"]
]

ACTION_W = [
    ["after a specified amount of time", "after a while", "after a large amount of time"],
    ["after a fixed delay", "without delay", "after several hours"],
    ["after a defined period ", "immediately", "much later"]
]

# ================= FUNCTIONS =================

def classify_eco(row, action_sets):
    for action_name in ['X', 'Y', 'Z', 'W']:
        if action_name not in action_sets:
            continue
        for keywords in action_sets[action_name]:
            if len(keywords) < 3:
                continue  # sicurezza

            eco_kw = keywords[0].lower()
            neutral_kw = keywords[1].lower()
            noneco_kw = keywords[2].lower()

            desc = str(row['actionDesc']).lower()

            if eco_kw in desc:
                return "ECO"
            elif neutral_kw in desc:
                return "NEUTRAL"
            elif noneco_kw in desc:
                return "NON_ECO"
    return "UNKNOWN"

dfs = []

for file in CSV_FILES:
    path = os.path.join(INPUT_FOLDER, file)

    if not os.path.exists(path):
        print(f"[WARNING] File not found: {file}")
        continue

    df = pd.read_csv(path)
    print(f"Loaded {file}: {len(df)} rows")

    # ================= Set ACTION sets per categoria =================
    if "Air_Climate" in file:
        action_sets = {'X': ACTION_X, 'Y': ACTION_Y, 'Z': ACTION_Z, 'W': ACTION_W}
    elif "Appliance" in file:
        action_sets = {'X': ACTION_X, 'Y': ACTION_Y, 'W': ACTION_W}
    elif "Generic_Other" in file:
        action_sets = {'X': ACTION_X, 'Y': ACTION_Y, 'Z': ACTION_Z, 'W': ACTION_W}
    elif "Light" in file:
        action_sets = {'X': ACTION_X, 'Y': ACTION_Y, 'Z': ACTION_Z, 'W': ACTION_W}
    elif "Thermostat" in file:
        action_sets = {'X': ACTION_X, 'Y': ACTION_Y, 'Z': ACTION_Z, 'W': ACTION_W}
    else:
        action_sets = {}

    # ================= CLASSIFICATIONS =================
    df['isRuleEco'] = df.apply(lambda row: classify_eco(row, action_sets), axis=1)

    dfs.append(df)

# ================= CONCAT & SAVE =================
if dfs:
    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Final CSV saved: {OUTPUT_FILE}, total rows: {len(final_df)}")
else:
    print("No CSVs loaded, nothing to save.")