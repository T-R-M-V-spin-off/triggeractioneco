import os
import pandas as pd

# ================= CONFIG =================

INPUT_FOLDER = "../../dataset/filtered_data/device_category_data/"
OUTPUT_FILE = "../../dataset/filtered_data/aggregated_dataset.csv"

# List of csv files to aggregate
CSV_FILES = [
    "Air_Climate.csv",
    "Appliance.csv",
    "Generic_Other.csv",
    "Light.csv",
    "Thermostat_Heating.csv",
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