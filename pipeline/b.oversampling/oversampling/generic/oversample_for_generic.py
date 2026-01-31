import random
import pandas as pd

# ===============================
# CONFIG
# ===============================

INPUT_FILE_NORMALIZED = "../../../../dataset/filtered_data/device_category_data/Generic_Other_normalized_for_oversampling.csv"

OUTPUT_FILE_OVERSAMPLED = INPUT_FILE_NORMALIZED.replace(".csv", "_oversampled.csv")
OUTPUT_FILE_FINAL = INPUT_FILE_NORMALIZED.replace(".csv", "_oversampled_final.csv")

# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv(INPUT_FILE_NORMALIZED)

# ===============================
# STEP 1: freeOrLocked (manteniamo [FREE] per la validazione)
# ===============================

def classify_free_locked(trigger_text):
    if pd.isna(trigger_text):
        return "LOCKED", trigger_text
    if "[FREE]" in trigger_text:
        return "FREE", trigger_text.strip()  # mantieni [FREE] per ora
    return "LOCKED", trigger_text

df[["freeOrLocked", "triggerDesc"]] = df["triggerDesc"].apply(
    lambda x: pd.Series(classify_free_locked(x))
)

# ===============================
# STEP 2: Prepare oversampling + synonym augmentation
# ===============================

free_triggers = df[df["freeOrLocked"] == "FREE"]["triggerDesc"].unique()
locked_triggers = df[df["freeOrLocked"] == "LOCKED"]["triggerDesc"].unique()
unique_actions = df["actionDesc"].unique()

# Dizionari di sinonimi
trigger_synonyms = {
    "start": ["turn on", "activate"],
    "turn on": ["start", "activate"],
    "specified": ["chosen"],
    "turned on": ["started", "activated"],
}

action_synonyms_on = ["turned on", "started", "turn on", "start", "activate"]
action_synonyms_off = ["turned off", "stopped", "turn off", "stop", "deactivate"]
action_synonyms_specified = ["specified", "chosen"]

def augment_text(text, synonyms_dict):
    words = text.split()
    new_words = []
    for w in words:
        lw = w.lower().strip(",.")  # minuscolo senza punteggiatura
        if lw in synonyms_dict:
            new_words.append(random.choice(synonyms_dict[lw]))
        else:
            new_words.append(w)
    return " ".join(new_words)


# ===============================
# STEP 2a: augment FREE triggers
# ===============================

def generate_synonym_variants(text, synonyms_dict, n_variants=2):
    """
    Genera n_variants varianti di `text` sostituendo parole con sinonimi.
    """
    variants = set()
    variants.add(text)  # includi l'originale

    for _ in range(n_variants):
        new_words = []
        for w in text.split():
            lw = w.lower().strip(",.")  # minuscolo senza punteggiatura
            if lw in synonyms_dict:
                new_words.append(random.choice(synonyms_dict[lw]))
            else:
                new_words.append(w)
        variants.add(" ".join(new_words))

    return list(variants)


# Nuovi trigger generati dai FREE triggers
augmented_free_triggers = []
for trig in free_triggers:
    variants = generate_synonym_variants(trig, trigger_synonyms, n_variants=3)
    augmented_free_triggers.extend(variants)

# Rimuoviamo duplicati
free_triggers = list(set(list(free_triggers) + augmented_free_triggers))

# ===============================
# Trigger / Action originali
# ===============================
free_triggers_original = df[df["freeOrLocked"] == "FREE"]["triggerDesc"].unique()
locked_triggers = df[df["freeOrLocked"] == "LOCKED"]["triggerDesc"].unique()
unique_actions_original = df["actionDesc"].unique()

print("=== PRIMA DELL'AUGMENTATION ===")
print(f"Trigger FREE originali: {len(free_triggers_original)}")
print(f"Trigger FREE originali lista: {free_triggers_original}")
print(f"Action uniche originali: {len(unique_actions_original)}")
print(f"Action uniche originali lista: {unique_actions_original}")

# ===============================
# Dizionari / liste di sinonimi
# ===============================
trigger_synonyms = {
    "start": ["turn on", "activate", "enable"],
    "turn on": ["start", "activate", "enable"],
    "specified": ["chosen", "specific"],
    "turned on": ["started", "activated", "enabled"],
    "binary switch": ["switch", "toggle switch"]
}

action_synonyms_on_1 = ["turned on", "started", "activated", "enabled"]
action_synonyms_on_2 = ["turn on", "start", "activate", "enable"]
action_synonyms_off_1 = ["turned off", "stopped", "deactivated", "disabled"]
action_synonyms_off_2 = ["turn off", "stop", "deactivate", "disable"]
action_synonyms_specified = ["specified", "chosen", "specific"]
action_synonyms_device = ["oven", "dryer", "microwave"]


# ===============================
# Funzioni di augmentation
# ===============================
def generate_synonym_variants(text, synonyms_dict, n_variants=3):
    variants = set()
    variants.add(text)

    for _ in range(n_variants):
        new_words = []
        for w in text.split():
            lw = w.lower().strip(",.")
            if lw in synonyms_dict:
                new_words.append(random.choice(synonyms_dict[lw]))
            else:
                new_words.append(w)
        variants.add(" ".join(new_words))

    return list(variants)


def augment_action(action_text):
    new_text = action_text
    for w in action_synonyms_on_1:
        new_text = new_text.replace(w, random.choice(action_synonyms_on_1))
    for w in action_synonyms_off_1:
        new_text = new_text.replace(w, random.choice(action_synonyms_off_1))
    for w in action_synonyms_on_2:
        new_text = new_text.replace(w, random.choice(action_synonyms_on_2))
    for w in action_synonyms_off_2:
        new_text = new_text.replace(w, random.choice(action_synonyms_off_2))
    for w in action_synonyms_specified:
        new_text = new_text.replace(w, random.choice(action_synonyms_specified))
    for w in action_synonyms_device:
        new_text = new_text.replace(w, random.choice(action_synonyms_device))
    return new_text


# ===============================
# Augmentation
# ===============================
augmented_free_triggers = []
for trig in free_triggers_original:
    variants = generate_synonym_variants(trig, trigger_synonyms, n_variants=3)
    augmented_free_triggers.extend(variants)

free_triggers = list(set(list(free_triggers_original) + augmented_free_triggers))

augmented_actions = []
for act in unique_actions_original:
    for _ in range(3):
        augmented_actions.append(augment_action(act))

unique_actions = list(set(list(unique_actions_original) + augmented_actions))

# ===============================
# Stampa dopo augmentation
# ===============================
print("\n=== DOPO L'AUGMENTATION ===")
print(f"Trigger FREE dopo augmentation: {len(free_triggers)}")
print(f"Trigger FREE lista dopo augmentation: {free_triggers}")
print(f"Action uniche dopo augmentation: {len(unique_actions)}")
print(f"Action uniche lista dopo augmentation: {unique_actions}")

oversampled_rows = []

# -------------------------------
# FREE triggers
# -------------------------------
for trig in free_triggers:
    for act in unique_actions:
        for _ in range(5):
            oversampled_rows.append({
                "triggerDesc": trig,
                "actionDesc": act,
                "freeOrLocked": "FREE"
            })

# -------------------------------
# LOCKED triggers
# -------------------------------
for trig in locked_triggers:
    associated_actions = df[df["triggerDesc"] == trig]["actionDesc"].unique()
    for act in associated_actions:
        for _ in range(5):
            oversampled_rows.append({
                "triggerDesc": trig,
                "actionDesc": act,
                "freeOrLocked": "LOCKED"
            })

# ===============================
# STEP 3: Creazione DataFrame oversampled
# ===============================

oversampled_df = pd.DataFrame(oversampled_rows)

# ===============================
# STEP 4: VALIDATION / PAIR CHECK
# ===============================

# Coppie uniche del dataset originale
original_pairs = df[["triggerDesc", "actionDesc"]].drop_duplicates()

# Coppie uniche del dataset oversampled
oversampled_pairs = oversampled_df[["triggerDesc", "actionDesc"]].drop_duplicates()

# Coppie originali mancanti
missing_in_oversampled = original_pairs.merge(
    oversampled_pairs,
    on=["triggerDesc", "actionDesc"],
    how="left",
    indicator=True
).query('_merge == "left_only"')

# Coppie extra
extra_in_oversampled = oversampled_pairs.merge(
    original_pairs,
    on=["triggerDesc", "actionDesc"],
    how="left",
    indicator=True
).query('_merge == "left_only"')

# Distinguo quelle aggiunte dovute ai trigger FREE
extra_free = extra_in_oversampled[extra_in_oversampled["triggerDesc"].str.contains(r"\[FREE\]")]
extra_locked = extra_in_oversampled[~extra_in_oversampled["triggerDesc"].str.contains(r"\[FREE\]")]

print(f"Tutte le coppie originali sono presenti: {len(missing_in_oversampled) == 0}")
print(f"Coppie aggiuntive nell'oversampled: {len(extra_in_oversampled)}")
print(f"- Dipendono da trigger FREE: {len(extra_free)}")
print(f"- Dipendono da trigger LOCKED: {len(extra_locked)}")

# ===============================
# STEP 5: Rimuovo [FREE] per il CSV finale
# ===============================

oversampled_df["triggerDesc"] = oversampled_df["triggerDesc"].str.replace(r" ?\[FREE\]", "", regex=True).str.strip()

oversampled_df.to_csv(OUTPUT_FILE_FINAL, index=False)

print(f"Oversampled dataset finale salvato in: {OUTPUT_FILE_FINAL}")
print(f"Numero righe finali: {len(oversampled_df)}")
