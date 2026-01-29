import pandas as pd
import os

# Impostazioni pandas (testi completi)
pd.set_option("display.max_colwidth", None)

# Percorsi
INPUT_FILE = "../dataset/filtered_data/augmented_dataset.csv"
OUTPUT_FOLDER = "../dataset/sets/"

# Crea cartella output se non esiste
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Carica dataset
df = pd.read_csv(INPUT_FILE)

# Colonne richieste
required_columns = [
    "deviceCategory",
    "triggerDesc",
    "triggerType",
    "actionDesc",
    "actionType"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Colonna mancante: {col}")

# Categorie device
device_categories = df["deviceCategory"].dropna().unique()

print(f"Trovate {len(device_categories)} categorie.\n")

# Funzione per rendere i nomi file sicuri
def clean_filename(name):
    return str(name).replace(" ", "_").replace("/", "_").lower()


# Per ogni deviceCategory
for device in device_categories:

    print(f"Processing: {device}")

    subset = df[df["deviceCategory"] == device]

    safe_device = clean_filename(device)

    # ==================================================
    # TRIGGER CSV
    # ==================================================

    trigger_group = (
        subset
        .groupby(["triggerDesc", "triggerType"])
        .size()
        .reset_index(name="count")
    )

    total_triggers = trigger_group["count"].sum()

    trigger_group["percentage"] = (
        trigger_group["count"] / total_triggers * 100
    ).round(2)

    trigger_csv = trigger_group[[
        "triggerDesc",
        "percentage",
        "triggerType"
    ]]

    trigger_filename = f"{safe_device}_triggers.csv"
    trigger_path = os.path.join(OUTPUT_FOLDER, trigger_filename)

    trigger_csv.to_csv(trigger_path, index=False)

    # ==================================================
    # ACTION CSV
    # ==================================================

    action_group = (
        subset
        .groupby(["actionDesc", "actionType"])
        .size()
        .reset_index(name="count")
    )

    total_actions = action_group["count"].sum()

    action_group["percentage"] = (
        action_group["count"] / total_actions * 100
    ).round(2)

    action_csv = action_group[[
        "actionDesc",
        "percentage",
        "actionType"
    ]]

    action_filename = f"{safe_device}_actions.csv"
    action_path = os.path.join(OUTPUT_FOLDER, action_filename)

    action_csv.to_csv(action_path, index=False)

    # ==================================================

    print(f"  -> Creati:")
    print(f"     {trigger_filename}")
    print(f"     {action_filename}\n")

print("Completato.")
