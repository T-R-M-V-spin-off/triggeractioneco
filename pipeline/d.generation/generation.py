import pandas as pd

# ===============================
# CONFIG
# ===============================
INPUT_FILE = "../../dataset/filtered_data/ultra_mega_final.csv"

OUTPUT_ECO = "ultra_mega_final_eco.csv"
OUTPUT_NEUTRAL = "ultra_mega_final_neutral.csv"
OUTPUT_NON_ECO = "ultra_mega_final_non_eco.csv"

# ===============================
# LOAD CSV
# ===============================
df = pd.read_csv(INPUT_FILE)

# ===============================
# FILTER
# ===============================
df_eco = df[df["isRuleEco"] == "ECO"]
df_neutral = df[df["isRuleEco"] == "NEUTRAL"]
df_non_eco = df[df["isRuleEco"] == "NON-ECO"]

# ===============================
# SAVE
# ===============================
df_eco.to_csv(OUTPUT_ECO, index=False)
df_neutral.to_csv(OUTPUT_NEUTRAL, index=False)
df_non_eco.to_csv(OUTPUT_NON_ECO, index=False)

print("✅ File creati con successo!")
print(f"- {OUTPUT_ECO}: {len(df_eco)} righe")
print(f"- {OUTPUT_NEUTRAL}: {len(df_neutral)} righe")
print(f"- {OUTPUT_NON_ECO}: {len(df_non_eco)} righe")



# Per ogni csv creato aggiungi due colonne : se il csv è quello eco aggiuni neutralVariant e nonEcoVariant
# se il csv è non eco aggiungi ecoVariant e neutralVariant
# se il csv è neutral aggiungi ecoVariant e nonEcoVariant

# Partendo dal csv non eco nella colonna neutralVariant copia prima il conenuto della colonna actionDesc
# e poi sostituisci tutto con ", THEN set the mode of the air conditioner.",
# mentre nella colonna ecoVariant sostituisci con ", THEN set the mode of the air conditioner to eco mode.",


# Poi per il csv eco prendi la colonna triggerDesc e sostituisci ogni espressione legata al tempo con all time o se non
# ci sta alcun riferimento temporale aggiungi for a long period of time

# Poi per il csv neutro


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

# eco -> neutra

# turbo -> eco    trasformare una non eco in eco # posso trasfromare una neutra in eco
# turbo -> sleep trasformare una non eco in eco  # posso trasfromare una non eco in neutra
# + for a specified amount of time and using user prefernces trasfromare una neutra in eco

ACTIONS_FROM_NONECO_TO_ECO = {
    "ENERGY_WASTE": {"keywords": ["eco","sleep"], "score": 0.6},
}