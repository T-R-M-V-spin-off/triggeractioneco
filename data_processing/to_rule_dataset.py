import pandas as pd
import re

# ================= LOAD =================

INPUT_CSV = "../dataset/action_channel_filtered_dataset.csv"
OUTPUT_TRANSFORMED = "transformed_dataset.csv"
OUTPUT_ADDITIONAL = "unmodified_with_content_after_dot.csv"

df_original = pd.read_csv(INPUT_CSV)

# Dataset di lavoro
df = df_original[['id', 'triggerDesc', 'actionDesc', 'deviceCategory']].copy()

initial_rows = len(df)

# ================= FILTER TRIGGERS =================

trigger_regex = re.compile(r"^this trigger fires", re.IGNORECASE)

df = df[df['triggerDesc'].fillna("").str.match(trigger_regex)].copy()

final_rows = len(df)
removed_rows = initial_rows - final_rows

print(f"\nRighe eliminate perché non iniziano con 'This trigger fires': {removed_rows}")
print(f"Righe rimanenti: {final_rows}")

# ================= DEVICE ELIMINATI PER CATEGORIA =================

remaining_ids = set(df['id'])

removed_df = df_original[~df_original['id'].isin(remaining_ids)]

device_removed_counts = (
    removed_df
    .groupby('deviceCategory')
    .size()
    .sort_values(ascending=False)
)

print("\nDevice eliminati per tipologia (deviceCategory):")
print(device_removed_counts)


# ================= TRANSFORM triggerDesc =================

def transform_trigger_desc(desc):
    if pd.isna(desc):
        return "WHEN", ""

    desc = desc.strip()

    # Rimuovi all'inizio 'This trigger fires', 'when' o 'if', case-insensitive
    desc = re.sub(r'^(this trigger fires when|this trigger fires if|this trigger fires|when|if)\b\s*', '', desc, flags=re.IGNORECASE)

    # Trova posizione del primo punto
    dot_pos = desc.find('.')
    if dot_pos != -1:
        first_part = desc[:dot_pos]  # prima del punto
        rest_part = desc[dot_pos + 1:].strip()  # tutto dopo il punto
    else:
        first_part = desc
        rest_part = ""

    # Aggiungi WHEN all'inizio
    transformed = f"WHEN {first_part}" if first_part else "WHEN"

    return transformed, rest_part


# Applica la trasformazione
trigger_results = df['triggerDesc'].apply(transform_trigger_desc)
df['triggerDesc'] = trigger_results.apply(lambda x: x[0])
removed_trigger_content = trigger_results.apply(lambda x: x[1])

# Conta quante righe hanno perso contenuto reale
trigger_rows_lost_mask = removed_trigger_content != ""
trigger_rows_lost = trigger_rows_lost_mask.sum()
total_trigger_rows = len(df)

print(
    f"\nRighe triggerDesc che hanno perso contenuto dopo il primo punto: "
    f"{trigger_rows_lost} su {total_trigger_rows} "
    f"({trigger_rows_lost / total_trigger_rows * 100:.2f}%)"
)

# ================= TRANSFORM actionDesc =================

def transform_action_desc(action_desc):
    if pd.isna(action_desc):
        return ", THEN", ""

    # Se actionDesc inizia con "When" (case-insensitive), NON tagliare nulla
    if re.match(r'^when\b', action_desc.strip(), flags=re.IGNORECASE):
        return action_desc, ""

    # Trova posizione del primo punto
    dot_pos = action_desc.find('.')
    if dot_pos != -1:
        first_part = action_desc[:dot_pos]
        rest_part = action_desc[dot_pos + 1:].strip()
    else:
        first_part = action_desc
        rest_part = ""

    # Se inizia con "This Action will"/"This action well"/"This action" applica sostituzione
    if re.match(r'^(this action will|this action well|this action)\b', first_part, flags=re.IGNORECASE):
        transformed = re.sub(
            r"this action will|this action well|this action",
            ", THEN",
            first_part,
            flags=re.IGNORECASE
        )
    else:
        # Aggiunge THEN all'inizio
        transformed = f", THEN {first_part}" if first_part else ", THEN"

    # Tutto minuscolo tranne THEN
    transformed = transformed.lower()
    transformed = re.sub(r'\bthen\b', 'THEN', transformed)

    # Rimuovi eventuale punto finale
    transformed = transformed.rstrip('.')

    return transformed, rest_part

results = df['actionDesc'].apply(transform_action_desc)
df['actionDesc'] = results.apply(lambda x: x[0])
removed_content = results.apply(lambda x: x[1])

# ================= STATS =================

rows_lost_content_mask = removed_content != ""
rows_lost_content = rows_lost_content_mask.sum()
total_rows = len(df)

print(
    f"\nRighe che hanno perso contenuto dopo il primo punto: "
    f"{rows_lost_content} su {total_rows} "
    f"({rows_lost_content / total_rows * 100:.2f}%)"
)

rows_with_note = removed_content[rows_lost_content_mask] \
    .str.contains(r'\bNOTE\b', case=False, na=False).sum()

if rows_lost_content > 0:
    print(
        f"Tra queste, righe che contenevano la parola NOTE: "
        f"{rows_with_note} "
        f"({rows_with_note / rows_lost_content * 100:.2f}%)"
    )

# ================= FINAL DATASET =================

final_df = df[['id', 'triggerDesc', 'actionDesc']]
final_df.to_csv(OUTPUT_TRANSFORMED, index=False)

# ================= ADDITIONAL CSV =================

mask_has_content_after_dot = removed_content != ""
mask_no_note = ~removed_content.str.contains(r'\bNOTE\b', case=False, na=False)
mask_additional_csv = mask_has_content_after_dot & mask_no_note

additional_csv_df = df.loc[mask_additional_csv].copy()
additional_csv_df['original_content_after_dot'] = removed_content[mask_additional_csv]

additional_csv_df.to_csv(OUTPUT_ADDITIONAL, index=False)

print(f"\nCSV finale salvato: {OUTPUT_TRANSFORMED}")
print(f"CSV aggiuntivo generato con {len(additional_csv_df)} righe: {OUTPUT_ADDITIONAL}")
