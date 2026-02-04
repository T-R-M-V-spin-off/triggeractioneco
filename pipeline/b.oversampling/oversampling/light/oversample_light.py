import pandas as pd

# ===============================
# CONFIG
# ===============================

INPUT_FILE = "../../../../dataset/filtered_data/device_category_data/Light_normalized_for_oversampling.csv"          # dataset originale
OUTPUT_FILE = "../../../../dataset/filtered_data/device_category_data/Light_normalized_for_oversampling_oversampled_final.csv"    # dataset generato

# ===============================
# ACTION MATRICES
# ===============================

ACTION_X = [
    ["eco mode", "comfort mode", "turbo mode"],
    ["low power mode", "default mode", "maximum power"],
    ["energy saving mode", "auto mode", "boost mode"]
]

ACTION_Y = [
    ["for a specific amount of time", "for a while", "for the entire day"],
    ["for a limited time", "until stopped", "indefinitely"],
    ["temporarily", "until manually changed", "continuously"]
]

ACTION_Z = [
    ["at low power", "at medium power", "at maximum power"],
    ["at minimum power ", "at default power", "at high power"],
    ["to power saving mode", "at moderate power", "at boosted power"]
]

ACTION_W = [
    ["after a specified amount of time", "after a while", "after a large amount of time"],
    ["after a fixed delay", "without delay", "after several hours"],
    ["after a defined period ", "immediately", "much later"]
]

PLACEHOLDER_MAP = {
    "[ACTION_X]": ACTION_X,
    "[ACTION_Y]": ACTION_Y,
    "[ACTION_Z]": ACTION_Z,
    "[ACTION_W]": ACTION_W
}


# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv(INPUT_FILE)

unique_triggers = df["triggerDesc"].unique()
unique_actions = df["actionDesc"].unique()

# ===============================
# SYNONYMS AUGMENTATION
# ===============================

trigger_synonyms = {

}

action_synonyms = {

}

def augmented_syn(unique_list, synonyms, is_trigger) :

    new_unique_list = []

    for element in unique_list :
        new_unique_list.append(element)

        if (is_trigger and ("[FREE]" in element.lower())) or not is_trigger:
            for key in synonyms :
                for syn in synonyms[key] :
                    if key.lower() in element.lower() :
                        new_unique_list.append(element.replace(key, syn))

    return new_unique_list

#unique_triggers = augmented_syn(unique_triggers, trigger_synonyms, True)
#unique_actions = augmented_syn(unique_actions, action_synonyms, False)

print("UNIQUE TRIGGER LIST: ")
for trigger in unique_triggers:
    print("TRIGGER: ", trigger)

print("UNIQUE ACTION LIST: ")
for action in unique_actions:
    print("ACTION: ", action)

# ===============================
# ACTION MATRICES
# ===============================

def replace_placeholder(unique_actions, PLACEHOLDER_MAP) :

    new_unique_actions = []

    for action in unique_actions :
        for key in PLACEHOLDER_MAP :
            if key in action:
                for sublist in PLACEHOLDER_MAP[key]:
                    for mode in sublist:
                        new_unique_actions.append(action.replace(key, mode))

    return new_unique_actions


unique_actions = replace_placeholder(unique_actions, PLACEHOLDER_MAP)

print("UNIQUE ACTION LIST WITH PLACEHOLDERS REPLACED: ")
for action in unique_actions:
    print("ACTION: ", action)


df["triggerType"] = df["triggerDesc"].astype(str).apply(lambda x: "FREE" if "[FREE]" in x else "LOCKED")

free_triggers_before = df[df["triggerType"] == "FREE"]["triggerDesc"].unique()
locked_triggers_before = df[df["triggerType"] == "LOCKED"]["triggerDesc"].unique()

print("FREE TRIGGER BEFORE:", free_triggers_before)
print("LOCKED TRIGGER BEFORE:", locked_triggers_before)

# ===============================
# GENERATION
# ===============================

df_filtered = df[~df["triggerDesc"].astype(str).str.contains(r"\[FREE\]", regex=True)]

trigger_to_actions = df_filtered.groupby("triggerDesc")["actionDesc"].apply(lambda x: list(x.unique())).to_dict()

new_trigger_to_actions = {}

for trigger, actions in trigger_to_actions.items():
    new_actions = replace_placeholder(actions, PLACEHOLDER_MAP)
    new_trigger_to_actions[trigger] = new_actions

for trigger, actions in new_trigger_to_actions.items():
    print(f"Trigger: {trigger}")
    for action in actions:
        print(f"  - {action}")
    print()

generated_rows = []

for _, row in df.iterrows():
    trigger = row["triggerDesc"]
    ttype = row["triggerType"]

    if ttype == "LOCKED":
        actions = new_trigger_to_actions.get(trigger, [])

        for action in actions:
            generated_rows.append((trigger, action))
    else:
        actions = unique_actions

        for action in actions:
            trigger = trigger.replace(" [FREE]", "")
            generated_rows.append((trigger, action))

gen_df = pd.DataFrame(generated_rows)
new_column_names = ["triggerDesc", "actionDesc"]
gen_df.columns = new_column_names

gen_df.to_csv(OUTPUT_FILE, index=False)
print("\nDataset salvato in:", OUTPUT_FILE)