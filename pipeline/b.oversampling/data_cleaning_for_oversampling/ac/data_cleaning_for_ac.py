import pandas as pd
import os
import re


# ===============================
# CONFIG
# ===============================

INPUT_FILE = "../../../../dataset/filtered_data/device_category_data/Air_Climate.csv"

# ===============================
# NORMALIZATION RULES
# ===============================

TRIGGER_RULES = [
    (r".*device.*turned on.*|device turned on", "IF a device is turned on [FREE]"),
    (r".*calendar.*event.*", "IF shortly before a calendar event with a specific keyword occurs [FREE]"),
    (r".*(alexa trigger|ok google|google assistant).*", "IF a voice assistant is activated with a specific phrase [FREE]"),
    (r".*before.*time-of-day.*peak rates.*start.*", "IF before time-of-day peak rates start"),
    (r".*every day at a specified time.*", "IF every day at a specified time [FREE]"),
    (r".*message.*key phrase.*", "IF a message containing a key phrase is sent [FREE]"),
    (r".*home.*set to away.*", "IF every time no one is at home"),
    (r".*home.*set to home.*", "IF every time someone is at home"),
    (r".*user enters a specified area.*", "IF the user enters a specified area"),
    (r".*user exits a specified area.*", "IF the user exits a specified area"),
    (r".*button.*pressed.*", "IF a button is pressed [FREE]"),
    (r".*enter.*area.*", "IF the user enters a specified area"),
    (r".*exit.*area.*", "IF the user exits a specified area"),
    (r".*voice assistant.*activated.*", "IF a voice assistant is activated with a specific phrase [FREE]"),
    (r".*a/c unit.*temperature.*above.*value.*", "IF an AC unit detects temperature above a specified value"),
    (r".*a/c unit.*temperature.*below.*value.*", "IF an AC unit detects temperature below a specified value"),
    (r".*android device.*connects to wifi.*", "IF an Android device connects to a specified Wi-Fi network"),
    (r".*android device.*disconnects from wifi.*", "IF an Android device disconnects from a specified Wi-Fi network"),
    (r".*fitbit.*logs.*sleep.*", "IF a device logs new sleep data"),
    (r".*every.*day.*specific time.*", "IF every day at a specified time [FREE]"),
    (r".*every time.*connects.*", "IF your Android device connects to a specified wifi network"),
    (r".*every time.*disconnects.*", "IF your Android device disconnects from a specified wifi network"),
    (r".*press the button.*", "IF a specified button is pressed [FREE]"),
    (r".*ifttt receives.*", "IF ifttt receives a specified event from an external service [FREE]"),
    (r".*myfox.*security system.*partially armed.*", "IF a security system is partially armed"),
    (r".*smartthings.*device.*switched off.*", "IF a smart device is switched off [FREE]"),
    (r".*smartthings.*device.*switched on.*", "IF a smart device is switched on [FREE]"),
    (r".*routine.*activated.*", "IF a routine is activated [FREE]"),
    (r".*air purifier.*air quality.*", "IF an air purifier detects a specific air quality level"),
    (r".*room temperature.*condition.*threshold.*", "IF the room temperature satisfies a threshold condition"),
    (r".*solar power.*drops below.*", "IF solar power drops below a specified value"),
    (r".*temperature.*specific device.*exceeds.*threshold.*", "IF a device temperature exceeds a threshold"),
    (r".*temperature.*satisfies.*threshold.*", "IF the room temperature satisfies a threshold condition"),
    (r".*device.*temperature.*above.*threshold.*", "IF a device detects temperature above a specified threshold"),
    (r".*device.*temperature.*below.*threshold.*", "IF a device detects temperature below a specified threshold"),
    (r".*indoor temperature.*dropping below.*threshold.*", "IF indoor temperature drops below a specified threshold"),
    (r".*indoor temperature.*rising above.*threshold.*", "IF indoor temperature rises above a specified threshold"),
    (r".*sunrise.*", "IF a set number of minutes before sunrise"),
    (r".*sunset.*", "IF a set number of minutes before sunset"),
    (r".*unit.*turned on.*", "IF a set number of minutes after a unit has been turned on"),
    (r".*current weather condition.*rain|snow|cloudy|clear.*", "IF the weather conditions change"),
    (r".*local humidity.*above.*value.*", "IF local humidity is above a specified value"),
    (r".*local temperature.*drops below.*value.*", "IF local temperature is below a specified value"),
    (r".*local temperature.*rises above.*value.*", "IF local temperature is above a specified value"),
    (r".*room enters manual mode.*", "IF every time a room enters manual mode"),
    (r".*particulate matter.*", "IF the air quality is below a specified value")
]


ACTION_RULES = [
    (r".*changes modes.*auto.*sleep.*", ", THEN change the AC unit mode to sleep."),
    (r".*turns off.*air purifier.*", ", THEN turns off the air purifier."),
    (r".*turns on.*air purifier.*", ", THEN turns on the air purifier."),
    (r".*turns.*display.*on or off.*", ", THEN turns the AC unit display on."),
    (r".*turns.*fan.*speed.*low.*", ", THEN sets the fan to low speed."),
    (r".*turns.*fan.*medium.*", ", THEN sets the fan to medium speed."),
    (r".*turns.*fan.*high.*", ", THEN sets the fan to high speed."),
    (r".*disable.*timer.*", ", THEN disable the indicated timer."),
    (r".*enable.*timer.*", ", THEN enable the indicated timer."),
    (r".*econo mode.*enable|disable.*|.*holiday mode.*enable|disable.*", ", THEN disable eco mode on the AC unit."),
    (r".*execute.*scene.*", ", THEN set the mode of the air conditioner to turbo."),
    (r".*set.*mode.*air conditioner.*", ", THEN set the mode of the air conditioner to eco."),
    (r".*turn off.*air conditioner.*", ", THEN turn off the air conditioner."),
    (r".*turn off.*a/c.*specified room.*", ", THEN turn off the air conditioner in a specified room."),
    (r".*turn off.*daikin.*ac unit.*|.*turn.*intesishome.*a/c.*off.*", ", THEN turn off the AC unit."),
    (r".*turn on.*air conditioner.*", ", THEN turn on the air conditioner."),
    (r".*turn on.*a/c.*specified room.*comfort mode.*", ", THEN turn on the air conditioner in a specified room in comfort mode."),
    (r".*turn on.*daikin.*ac unit.*|.*turn.*intesishome.*a/c.*on.*", ", THEN turn on the AC unit."),
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
        r"NaN",
        case=False,
        na=False
    )
]

filtered_df = filtered_df[
    ~filtered_df["actionDesc"].str.contains(
        r".*comfort feedback.*",
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
