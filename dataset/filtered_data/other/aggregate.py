import os
import pandas as pd
import random
import re

# ================= CONFIG =================

INPUT_FOLDER = ""   # <-- cartella dove stanno i CSV
OUTPUT_FILE = "final_aggregated_dataset.csv"

# Lista file da aggregare
CSV_FILES = [
    "Air_Climate.csv",
    "Air_Climate_Synthetic.csv",
    "Appliance.csv",
    "Appliance_Synthetic.csv",
    "Generic_Other.csv",
    "Light.csv",
    "Thermostat_Heating.csv",
    "Thermostat_Heating_Synthetic.csv"
]

LIGHT_FILE = "../device_category_data/Light.csv"
LIGHT_REMOVE_N = 6000

# ================= FUNCTIONS =================

def clean_description(text):
    """
    Rimuove NOTE e tutto dopo il primo punto.
    """
    if pd.isna(text):
        return text

    # Rimuove "Note:" e simili
    text = re.sub(r"Note:.*", "", text, flags=re.IGNORECASE)

    # Taglia dopo il primo punto
    text = text.split(".")[0] + "."

    return text.strip()


def build_ifttt_rule(row):
    """
    Crea regola testuale:
    IF ... THEN ...
    """
    trigger = row["triggerDesc"]
    action = row["actionDesc"]

    return f'IF {trigger} THEN {action}'


# ================= LOAD FILES =================

dfs = []

for file in CSV_FILES:

    path = os.path.join(INPUT_FOLDER, file)

    if not os.path.exists(path):
        print(f"[WARNING] File non trovato: {file}")
        continue

    df = pd.read_csv(path)

    print(f"Caricato {file}: {len(df)} righe")

    # Caso speciale: Light.csv → rimuovi 6k righe con Aura
    if file == LIGHT_FILE:

        if "actionChannelTitle" in df.columns:

            aura_rows = df[
                df["actionChannelTitle"]
                .str.contains("Aura", case=False, na=False)
            ]

            print(f"Righe Aura in Light: {len(aura_rows)}")

            # Campiona max 6000 righe Aura
            to_remove = aura_rows.sample(
                min(LIGHT_REMOVE_N, len(aura_rows)),
                random_state=42
            )

            df = df.drop(to_remove.index)

            print(f"Rimosse {len(to_remove)} righe da Light.csv")

    dfs.append(df)


# ================= AGGREGATION =================

# Unione di tutti i dataframe
final_df = pd.concat(dfs, ignore_index=True)

print(f"\nDataset aggregato: {len(final_df)} righe totali")


# ================= CLEANING =================

# Pulisce descrizioni
for col in ["triggerDesc", "actionDesc"]:

    if col in final_df.columns:
        final_df[col] = final_df[col].apply(clean_description)


# ================= ADD IFTTT RULE =================

final_df["ifttt_rule"] = final_df.apply(build_ifttt_rule, axis=1)

# Versione con virgolette
final_df["ifttt_rule_quoted"] = final_df["ifttt_rule"].apply(
    lambda x: f'"{x}"'
)


# ================= SAVE =================


final_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nFile finale salvato in: {OUTPUT_FILE}")


# ================= STATS =================

print("\n--- STATISTICHE FINALI ---")
print("Totale righe:", len(final_df))
print("Categorie:")
print(final_df["deviceCategory"].value_counts())
