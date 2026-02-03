import pandas as pd
import re
from collections import Counter

# ===============================
# CONFIG
# ===============================

INPUT_FILE = "../../../../dataset/filtered_data/device_category_data/Air_Climate_normalized_for_oversampling.csv"          # dataset originale
OUTPUT_FILE = "../../../../dataset/filtered_data/device_category_data/Air_Climate_normalized_for_oversampling_oversampled_final.csv"    # dataset generato

# ===============================
# ACTION MATRICES
# ===============================

ACTION_X = [
    ["eco mode", "comfort mode", "turbo mode"],
    ["low power mode", "default mode", "maximum power"],
    ["energy saving mode", "auto mode", "boost mode"]
]

ACTION_Y = [
    ["temporarily", "indefinitely", "continuously"],
    ["for a limited time", "until stopped", "until manually changed"],
    ["for the entire day", "for a specific amount of time", "for a fixed delay"]
]

ACTION_Z = [
    ["low", "medium", "high"],
    ["minimum", "moderate", "maximum"],
    ["silent", "default", "boost"]
]

ACTION_W = [
    ["immediately", "without delay", "much later"],
    ["after several hours", "after a large amount of time", "after a fixed delay"],
    ["after a defined period", "after a specified amount of time", "after some time"]
]

PLACEHOLDER_MAP = {
    "ACTION_X": ACTION_X,
    "ACTION_Y": ACTION_Y,
    "ACTION_Z": ACTION_Z,
    "ACTION_W": ACTION_W
}

df = pd.read_csv(INPUT_FILE)

unique_triggers = df["triggerDesc"].unique()
unique_actions = df["actionDesc"].unique()

def augmented_trigger_syn(unique_triggers) :
    trigger_synonyms = {
        "start": ["turn on", "activate"],
        "turn on": ["start", "activate"],
        "specified": ["chosen"],
        "turned on": ["started", "activated"],
    }

    new_unique_triggers = []

    for trigger in unique_triggers :
        new_unique_triggers.append(trigger)
        for key in trigger_synonyms :
            for syn in trigger_synonyms[key] :
                if syn in trigger :
                    new_unique_triggers.append(trigger.replace(key, syn))

    return new_unique_triggers

unique_triggers = augmented_trigger_syn(unique_triggers)

print(unique_triggers)

# Funzione per creare varianti con sinonimi per trigger ed actions

# Funzione per sostituire i placeholder con i valori


# ===============================
# ACTION COLUMN STATS
# ===============================

action_stats = {
    "ACTION_X": [0, 0, 0],
    "ACTION_Y": [0, 0, 0],
    "ACTION_Z": [0, 0, 0],
    "ACTION_W": [0, 0, 0]
}

# ===============================
# FUNCTION TO EXPAND ACTION
# ===============================

def expand_action(action_text):
    """
    Espande un placeholder in un'azione
    Ritorna lista di action finali
    Aggiorna action_stats per colonna
    """
    for key, matrix in PLACEHOLDER_MAP.items():
        pattern = rf"\[{key}\]"
        if re.search(pattern, action_text):
            expanded = []
            rows = len(matrix)
            cols = len(matrix[0])
            for col in range(cols):
                for row in range(rows):
                    value = matrix[row][col]
                    new_action = re.sub(pattern, value, action_text)
                    new_action = " ".join(new_action.split())
                    expanded.append(new_action)
                    action_stats[key][col] += 1
            return expanded
    return [action_text]

# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv(INPUT_FILE)

# Controllo se trigger FREE / LOCKED
df["triggerType"] = df["triggerDesc"].apply(lambda x: "FREE" if "[FREE]" in x else "LOCKED")

# Trigger FREE e LOCKED prima
free_triggers_before = df[df["triggerType"] == "FREE"]["triggerDesc"].unique()
locked_triggers_before = df[df["triggerType"] == "LOCKED"]["triggerDesc"].unique()

print("Trigger FREE prima:", free_triggers_before)
print("Trigger LOCKED prima:", locked_triggers_before)

# ===============================
# GENERATION
# ===============================

generated_rows = []

for _, row in df.iterrows():
    trigger = row["triggerDesc"]
    action = row["actionDesc"]
    ttype = row["triggerType"]

    expanded_actions = expand_action(action)

    for new_action in expanded_actions:
        generated_rows.append({
            "triggerDesc": trigger,
            "actionDesc": new_action,
            "triggerType": ttype
        })

gen_df = pd.DataFrame(generated_rows)

# Trigger FREE e LOCKED dopo
free_triggers_after = gen_df[gen_df["triggerType"] == "FREE"]["triggerDesc"].unique()
locked_triggers_after = gen_df[gen_df["triggerType"] == "LOCKED"]["triggerDesc"].unique()

print("\nTrigger FREE dopo:", free_triggers_after)
print("Trigger LOCKED dopo:", locked_triggers_after)

# ===============================
# FREQUENZA ACTION
# ===============================

print("\n=== FREQUENZA ACTION ===")

action_freq = Counter(gen_df["actionDesc"])
for act, cnt in action_freq.most_common():
    print(f"{act}  -->  {cnt}")

# ===============================
# VALIDATION
# ===============================

print("\n=== VALIDATION ===")

original_pairs = set(zip(df["triggerDesc"], df["actionDesc"]))
generated_pairs = set(zip(gen_df["triggerDesc"], gen_df["actionDesc"]))

# Tutte presenti?
all_present = original_pairs.issubset(generated_pairs)
print("Tutte le coppie originali presenti:", all_present)

# Missing
missing = original_pairs - generated_pairs
if missing:
    print("\nCoppie mancanti:")
    print(pd.DataFrame(list(missing), columns=["triggerDesc", "actionDesc"]))

# Extra
extra = generated_pairs - original_pairs
free_extra = sum(1 for t, _ in extra if t in free_triggers_after)
locked_extra = sum(1 for t, _ in extra if t in locked_triggers_after)

print("\nCoppie extra:", len(extra))
print("- Da FREE:", free_extra)
print("- Da LOCKED:", locked_extra)

# ===============================
# PLACEHOLDER CHECK
# ===============================

bad = gen_df[gen_df["actionDesc"].str.contains(r"\[ACTION_", regex=True)]
print("\nPlaceholder rimasti:", len(bad))

# ===============================
# ACTION COLUMN REPORT
# ===============================

print("\n=== ACTION COLUMN STATS ===")
for key, counts in action_stats.items():
    print(f"\n{key}")
    print(f"  Colonna 0: {counts[0]}")
    print(f"  Colonna 1: {counts[1]}")
    print(f"  Colonna 2: {counts[2]}")

# ===============================
# SAVE DATA
# ===============================

gen_df.to_csv(OUTPUT_FILE, index=False)
print("\nDataset salvato in:", OUTPUT_FILE)
