import pandas as pd

# Percorsi file
input_file = "../dataset/filtered_data/aggregated_dataset.csv"      # CSV originale

# Carica il dataset
df = pd.read_csv(input_file)

filtered_df = df

filtered_df = filtered_df[
    ~filtered_df["triggerDesc"].str.contains(
        "Particulate Matter|Ce trigger|price is lowest|virtual device turned on",
        case=False,
        na=False
    )
]

filtered_df = filtered_df[
    ~filtered_df["actionDesc"].str.contains(
        "Comfort feedback|will clean|curtain|this action will arm the blue by adt security system",
        case=False,
        na=False
    )
]

# Salva il nuovo CSV
filtered_df.to_csv(input_file, index=False)

print(f"File salvato: {input_file}")
print(f"Righe originali: {len(df)}")
print(f"Righe dopo filtro: {len(filtered_df)}")
