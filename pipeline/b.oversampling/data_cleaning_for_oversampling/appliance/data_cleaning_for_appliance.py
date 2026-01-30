import pandas as pd
import os
import re


# ===============================
# CONFIG
# ===============================

INPUT_FILE = "../../../../dataset/filtered_data/device_category_data/Appliance.csv"


# ===============================
# NORMALIZATION RULES
# ===============================

TRIGGER_RULES = [
    (r".*device.*turned on.*|device turned on", "IF a device is turned on"),
    (r".*electricity price.*lowest.*", "IF electricity prices are low"),
    (r".*electricity.*cheap.*", "IF electricity prices are low"),
    (r".*(smoke detector|smoke alarm|dangerous smoke).*", "IF a smoke detector detects smoke"),
    (r".*every.*day.*specific time.*", "IF every day at a specified time"),
    (r".*button.*pressed.*|.*press.*button.*|flic.*", "IF a button is pressed"),
    (r".*to do list.*edit.*|.*edit.*to do list.*", "IF a to-do list item is edited"),
    (r".*enter or exit.*area.*", "IF the user enters or exits a specified area"),
    (r".*enter.*area.*", "IF the user enters a specified area"),
    (r".*exit.*area.*", "IF the user exits a specified area"),
    (r".*(alexa trigger|ok google|google assistant).*", "IF a voice assistant is activated with a specific phrase"),
    (r".*cookit.*finished.*", "IF a smart cooking device finishes successfully"),
    (r".*cookit.*turned on.*", "IF a smart cooking device is turned on"),
    (r".*binary switch.*turned on.*", "IF a binary switch is turned on"),
    (r".*freezer.*door.*open.*", "IF a freezer door is opened"),
    (r".*sleep.*out of bed.*", "IF a sleep tracking device detects the user leaving the bed"),
    (r".*specific days.*week.*time.*", "IF on specific days of the week at a specified time"),
    (r".*(homeseer|knx).*turned on.*", "IF a connected device is turned on."),
    (r".*security panel.*armed|disarmed.*", "IF a security system is armed or disarmed"),
    (r".*co2.*intensity.*lowest.*", "IF carbon intensity is lowest within a time window"),
    (r".*exchanged power.*threshold.*", "IF power usage satisfies a threshold condition"),
    (r".*(telegram|text message|ifttt).*", "IF a message containing a key phrase is sent"),
    (r".*google calendar.*event.*", "IF shortly before a calendar event with a specific keyword"),
    (r".*sunrise.*", "IF shortly before sunrise at the user's location"),
    (r".*lawnmower.*button.*", "IF a lawn mower button is pressed")
]

ACTION_RULES = [
    (r".*deactivate.*eco mode.*", ", THEN deactivate eco mode on the appliance."),
    (r".*deactivate.*fresh mode.*", ", THEN deactivate fresh mode on the appliance."),
    (r".*deactivate.*vacation mode.*", ", THEN deactivate vacation mode on the appliance."),
    (r".*preheat.*oven.*hot air.*fast preheat.*", ", THEN preheat the oven using the hot air program with fast preheat."),
    (r".*start.*dishwasher.*selected.*", ", THEN start the selected dishwasher program."),
    (r".*start.*hood.*selected.*", ", THEN start the selected hood program."),
    (r".*turn off.*coffee machine.*", ", THEN turn off a coffee machine."),
    (r".*turn off.*dishwasher.*", ", THEN turn off the dishwasher."),
    (r".*turn off.*hood.*", ", THEN turn off the hood."),
    (r".*turn off.*oven.*", ", THEN turn off then oven."),
    (r".*turn on.*coffee machine.*", ", THEN turn on the coffee machine."),
    (r".*turn on.*dishwasher.*", ", THEN turn on the dishwasher."),
    (r".*turn on.*hood.*default program.*", ", THEN turn on the hood with the default program."),
    (r".*start.*dishwasher.*", ", THEN start the dishwasher."),
    (r".*selected coffee.*", ", THEN start the coffee with the specified program."),
    (r".*start.*hood.*", ", THEN turn on the hood."),
    (r".*turn on.*oven.*", ", THEN turn on the oven.")
]

# ===============================
# NORMALIZATION FUNCTIONS
# ===============================

def normalize_text(text, rules, unmatched_tag):

    if pd.isna(text):
        return text

    text = str(text).lower().strip()

    for pattern, normalized in rules:
        if re.search(pattern, text, re.IGNORECASE):
            return normalized

    return f"{unmatched_tag} {text}"


def normalize_trigger(text):
    return normalize_text(text, TRIGGER_RULES, "__UNMATCHED_TRIGGER__")


def normalize_action(text):
    return normalize_text(text, ACTION_RULES, "__UNMATCHED_ACTION__")


# ===============================
# PROCESS
# ===============================

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

filtered_df = df

filtered_df = filtered_df[
    ~filtered_df["triggerDesc"].str.contains(
        r".*particulate matter.*",
        case=False,
        na=False
    )
]

filtered_df = filtered_df[
    ~filtered_df["actionDesc"].str.contains(
        "NaN",
        case=False,
        na=False
    )
]

# Salva il nuovo CSV
filtered_df.to_csv(INPUT_FILE, index=False)

print(f"File salvato: {INPUT_FILE}")
print(f"Righe originali: {len(df)}")
print(f"Righe dopo filtro: {len(filtered_df)}")


# Check columns
if "triggerDesc" not in df.columns:
    raise ValueError("Missing column: triggerDesc")

if "actionDesc" not in df.columns:
    raise ValueError("Missing column: actionDesc")


# Backup original
df["originalTrigger"] = df["triggerDesc"]
df["originalAction"] = df["actionDesc"]


# Normalize
print("Normalizing triggers...")
df["triggerDesc"] = df["triggerDesc"].apply(normalize_trigger)

print("Normalizing actions...")
df["actionDesc"] = df["actionDesc"].apply(normalize_action)


# ===============================
# SAVE OUTPUT
# ===============================

folder = os.path.dirname(INPUT_FILE)
base = os.path.basename(INPUT_FILE).replace(".csv", "")

output_file = os.path.join(
    folder,
    f"{base}_normalized_for_oversampling.csv"
)

df.to_csv(output_file, index=False)

print(f"Saved: {output_file}")


# ===============================
# VALIDATION / REPORT
# ===============================

print("\n========== NORMALIZATION REPORT ==========")

# Trigger stats
unique_triggers = df["originalTrigger"].dropna().unique()
unique_unmatched_triggers = (
    df[df["triggerDesc"].str.contains("__UNMATCHED_TRIGGER__", na=False)]
    ["originalTrigger"]
    .dropna()
    .unique()
)

total_unique_triggers = len(unique_triggers)
unmatched_unique_triggers = len(unique_unmatched_triggers)

unique_trigger_coverage = 100 * (total_unique_triggers - unmatched_unique_triggers) / total_unique_triggers

# Action stats
unique_actions = df["originalAction"].dropna().unique()
unique_unmatched_actions = (
    df[df["actionDesc"].str.contains("__UNMATCHED_ACTION__", na=False)]
    ["originalAction"]
    .dropna()
    .unique()
)

total_unique_actions = len(unique_actions)
unmatched_unique_actions = len(unique_unmatched_actions)

unique_action_coverage = 100 * (total_unique_actions - unmatched_unique_actions) / total_unique_actions

print("\n=== UNIQUE COVERAGE ===")

print(f"Unique triggers normalized: "
      f"{total_unique_triggers - unmatched_unique_triggers}/{total_unique_triggers} "
      f"({unique_trigger_coverage:.2f}%)")

print(f"Unique actions normalized: "
      f"{total_unique_actions - unmatched_unique_actions}/{total_unique_actions} "
      f"({unique_action_coverage:.2f}%)")


# === PRINT ALL UNMATCHED ===

print("\n========== ALL UNMATCHED TRIGGERS ==========\n")

unmatched_triggers_df = (
    df[df["triggerDesc"].str.contains("__UNMATCHED_TRIGGER__", na=False)]
    [["originalTrigger", "triggerDesc"]]
    .drop_duplicates()
)

if unmatched_triggers_df.empty:
    print("No unmatched triggers ✅")
else:
    for _, row in unmatched_triggers_df.iterrows():
        print(f"- ORIGINAL : {row['originalTrigger']}")
        print(f"  CURRENT  : {row['triggerDesc']}\n")


print("\n========== ALL UNMATCHED ACTIONS ==========\n")

unmatched_actions_df = (
    df[df["actionDesc"].str.contains("__UNMATCHED_ACTION__", na=False)]
    [["originalAction", "actionDesc"]]
    .drop_duplicates()
)

if unmatched_actions_df.empty:
    print("No unmatched actions ✅")
else:
    for _, row in unmatched_actions_df.iterrows():
        print(f"- ORIGINAL : {row['originalAction']}")
        print(f"  CURRENT  : {row['actionDesc']}\n")


print("==========================================")

