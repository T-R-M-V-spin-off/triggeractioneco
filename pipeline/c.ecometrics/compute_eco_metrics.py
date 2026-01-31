# ================= CONFIG =================
import os

import pandas as pd

INPUT_FOLDER = "../../dataset/filtered_data/device_category_data/"
OUTPUT_FILE = "../../dataset/filtered_data/ultra_mega_final.csv"

# List of csv files to aggregate
CSV_FILES = [
    "Air_Climate_normalized_for_oversampling_oversampled_final.csv",
    "Appliance_normalized_for_oversampling_oversampled_final.csv",
    "Generic_Other_normalized_for_oversampling_oversampled_final.csv",
    "Light_normalized_final.csv",
    "Thermostat_Heating_normalized_for_oversampling_oversampled_final.csv",
]


# ================= LOAD FILES =================

dfs = []

for file in CSV_FILES:

    path = os.path.join(INPUT_FOLDER, file)

    if not os.path.exists(path):
        print(f"[WARNING] File not found: {file}")
        continue

    df = pd.read_csv(path)

    print(f"Loaded {file}: {len(df)} rows")

    dfs.append(df)


# ================= AGGREGATION =================

# Unione di tutti i dataframe
final_df = pd.concat(dfs, ignore_index=True)

print(f"\nAggregated dataset: {len(final_df)} rows")


# ================= SAVE =================


final_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved file in: {OUTPUT_FILE}")


# SPECIFIED (max 0.1) = ["specific","specified","chosen"]

###

# Ogni riga ha come punteggio di base 0.5

# AUTOMATION = 0.6 ["time","time-of-day", "detects"]
# OCCUPANCY_AWARNESS = 0.6 = ["enter","enters","exit","exits","leaving","arrive","arrives","arrived","left","leave","leaves"]

# LOAD_SHIFTING = 0.6 = ["low"]
#---------------------
# SPECIFIED (max 0.06) = ["specific","specified","chosen", "indicated"]
# ENERGY_SAVING = 0.6 = ["eco", "sleep", "timed", "for a specified time"]
# USER_PREFERENCE (max 0.06) = ["you","preferred"]
# ENERGY_WASTE = 0.2 = ["turbo"]

## --- Configurazione parole chiave e punteggi ---
TRIGGER_KEYWORDS = {
    "AUTOMATION": {"keywords": ["time", "time-of-day", "detects"], "score": 0.6},
    "OCCUPANCY_AWARENESS": {"keywords": ["enter","enters","exit","exits","leaving","arrive","arrives","arrived","left","leave","leaves"], "score": 0.6},
    "LOAD_SHIFTING": {"keywords": ["low"], "score": 0.6}
}

ACTION_KEYWORDS = {
    "SPECIFIED": {"keywords": ["specific","specified","chosen","indicated"], "score": 0.06},
    "ENERGY_SAVING": {"keywords": ["eco", "sleep", "timed", "for a specified time"], "score": 0.6},
    "USER_PREFERENCE": {"keywords": ["you","preferred"], "score": 0.06},
    "ENERGY_WASTE": {"keywords": ["turbo"], "score": 0.2}
}

NEUTRAL_SCORE = 0.5


def calculate_score(text, keyword_dict):
    text_lower = str(text).lower()
    score = 0.5  # valore neutro

    found_energy = False  # flag per capire se compare ENERGY_SAVING o ENERGY_WASTE

    for metric, info in keyword_dict.items():
        # Controllo ENERGY_SAVING e ENERGY_WASTE
        if metric in ["ENERGY_SAVING", "ENERGY_WASTE", "AUTOMATION", "OCCUPANCY_AWARENESS", "LOAD_SHIFTING"]:
            for kw in info["keywords"]:
                if kw in text_lower:
                    score = info["score"]  # sovrascrive il neutro
                    found_energy = True

    # Controllo SPECIFIED e USER_PREFERENCE
    for metric in ["SPECIFIED", "USER_PREFERENCE"]:
        info = keyword_dict.get(metric)
        if info:
            for kw in info["keywords"]:
                if kw in text_lower:
                    score += info["score"]  # aggiunge allo score attuale

    return score

# --- Funzione per aggiungere extra score da SPECIFIED e USER_PREFERENCE ---
def add_specified_user_score(text, score):
    text_lower = str(text).lower()
    if any(kw in text_lower for kw in ACTION_KEYWORDS["SPECIFIED"]["keywords"]):
        score += ACTION_KEYWORDS["SPECIFIED"]["score"]
    if any(kw in text_lower for kw in ACTION_KEYWORDS["USER_PREFERENCE"]["keywords"]):
        score += ACTION_KEYWORDS["USER_PREFERENCE"]["score"]
    return score

# --- Leggi CSV ---
df = pd.read_csv(OUTPUT_FILE)  # sostituisci con il tuo file

# --- Crea nuove colonne ---
df["isTriggerEco"] = "NEUTRAL"
df["isActionEco"] = "NEUTRAL"
df["isRuleEco"] = "NEUTRAL"

# --- Calcola punteggi per triggerDesc ---
for idx, row in df.iterrows():
    trigger_score = calculate_score(row.get("triggerDesc", ""), TRIGGER_KEYWORDS)
    if trigger_score > NEUTRAL_SCORE:
        df.at[idx, "isTriggerEco"] = "ECO"

# --- Calcola punteggi per actionDesc ---
for idx, row in df.iterrows():
    action_score = calculate_score(row.get("actionDesc", ""), ACTION_KEYWORDS)
    action_score = add_specified_user_score(row.get("actionDesc", ""), action_score)
    if action_score > NEUTRAL_SCORE:
        df.at[idx, "isActionEco"] = "ECO"
    elif action_score == 0.5:
        df.at[idx, "isActionEco"] = "NEUTRAL"
    elif action_score < 0.5:
        df.at[idx, "isActionEco"] = "NON-ECO"

# --- Regola isRuleEco: se trigger o action è ECO, la regola è ECO ---
df.loc[(df["isTriggerEco"]=="ECO") & (df["isActionEco"]=="ECO"), "isRuleEco"] = "ECO"
df.loc[(df["isTriggerEco"]=="ECO") & (df["isActionEco"]=="NON-ECO"), "isRuleEco"] = "NEUTRAL"
df.loc[(df["isTriggerEco"]=="ECO") & (df["isActionEco"]=="NEUTRAL"), "isRuleEco"] = "ECO"
df.loc[(df["isTriggerEco"]=="NEUTRAL") & (df["isActionEco"]=="ECO"), "isRuleEco"] = "ECO"
df.loc[(df["isTriggerEco"]=="NEUTRAL") & (df["isActionEco"]=="NON-ECO"), "isRuleEco"] = "NON-ECO"
df.loc[(df["isTriggerEco"]=="NEUTRAL") & (df["isActionEco"]=="NEUTRAL"), "isRuleEco"] = "NEUTRAL"

# --- Salva CSV aggiornato ---
df.to_csv(OUTPUT_FILE, index=False)
print("CSV aggiornato salvato come output.csv")
