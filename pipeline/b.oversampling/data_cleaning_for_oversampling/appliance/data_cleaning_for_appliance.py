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
    (r".*device.*turned on.*|device turned on", "IF a device is turned on [FREE]"), #FREE
    (r".*electricity.*cheap.*|.*electricity price.*lowest.*", "IF electricity prices are low "),
    (r".*(smoke detector|smoke alarm|dangerous smoke).*", "IF a smoke detector detects smoke"),
    (r".*every.*day.*specific time.*", "IF every day at a specified time"),
    (r".*button.*pressed.*|.*press.*button.*|flic.*|.*lawnmower.*button.*", "IF a button is pressed [FREE]"), #FREE
    (r".*to do list.*edit.*|.*edit.*to do list.*", "IF a to-do list item is edited [FREE]"), #FREE
    (r".*enter.*area.*|.*enter or exit.*area.*", "IF the user enters a specified area"),
    (r".*exit.*area.*", "IF the user exits a specified area"),
    (r".*(alexa trigger|ok google|google assistant).*", "IF a voice assistant is activated with a specific phrase [FREE]"), #FREE
    (r".*cookit.*finished.*", "IF a smart cooking device finishes successfully"),
    (r".*cookit.*turned on.*", "IF a smart cooking device is turned on"),
    (r".*binary switch.*turned on.*", "IF a binary switch is turned on [FREE]"), #FREE
    (r".*freezer.*door.*open.*", "IF a freezer door is opened"),
    (r".*sleep.*out of bed.*", "IF a sleep tracking device detects the user leaving the bed"),
    (r".*specific days.*week.*time.*", "IF is one of the specific days of the week at a specified time [FREE]"), #FREE
    (r".*(homeseer|knx).*turned on.*", "IF a connected device is turned on"),
    (r".*co2.*intensity.*lowest.*", "IF carbon emissions is lowest within a time window"),
    (r".*exchanged power.*threshold.*", "IF power usage satisfies a threshold condition"),
    (r".*(telegram|text message|ifttt).*", "IF a message containing a key phrase is sent [FREE]"), #FREE
    (r".*google calendar.*event.*", "IF a calendar event with a specific keyword occurs [FREE]"), #FREE
    (r".*sunrise.*", "IF it is sunrise")
]

ACTION_RULES = [
    (r".*deactivate.*eco mode.*|.*deactivate.*fresh mode.*|.*deactivate.*vacation mode.*", ", THEN deactivate [ACTION_X] on the appliance."), # deactivate eco mode è non eco
    (r".*preheat.*oven.*hot air.*fast preheat.*", ", THEN preheat the oven using the hot air program with fast preheat [ACTION_Y]."),
    (r".*start.*dishwasher.*selected.*", ", THEN start the selected dishwasher program [ACTION_Y]."),
    (r".*start.*hood.*selected.*", ", THEN start the selected hood program [ACTION_Y]."),
    (r".*turn off.*coffee machine.*", ", THEN turn off a coffee machine [ACTION_W]."),
    (r".*turn off.*dishwasher.*", ", THEN turn off the dishwasher [ACTION_W]."),
    (r".*turn off.*hood.*", ", THEN turn off the hood [ACTION_W]."),
    (r".*turn off.*oven.*", ", THEN turn off then oven [ACTION_W]."),
    (r".*turn on.*coffee machine.*", ", THEN turn on the coffee machine [ACTION_Y]."),
    (r".*turn on.*dishwasher.*", ", THEN turn on the dishwasher in [ACTION_X]."),
    (r".*turn on.*hood.*default program.*", ", THEN turn on the hood with the default program [ACTION_Y]."),
    (r".*start.*dishwasher.*", ", THEN start the dishwasher [ACTION_Y]."),
    (r".*selected coffee.*", ", THEN start the coffee with the specified program [ACTION_Y]."),
    (r".*start.*hood.*", ", THEN turn on the hood [ACTION_Y]."),
    (r".*turn on.*oven.*", ", THEN turn on the oven [ACTION_Y].")
]

# ===============================
# NORMALIZATION FUNCTION
# ===============================


def normalize(df):

    unmatched_triggers = []
    unmatched_actions = []

    def normalize_text(text, RULES, unmatched_list):
        if not isinstance(text, str):
            return text

        text_l = text.lower()

        for pattern, replacement in RULES:
            if re.search(pattern, text_l):
                return replacement

        unmatched_list.append(text)
        return text


    df["triggerDesc"] = df["triggerDesc"].apply(
        normalize_text,
        args=(TRIGGER_RULES, unmatched_triggers)
    )


    df["actionDesc"] = df["actionDesc"].apply(
        normalize_text,
        args=(ACTION_RULES, unmatched_actions)
    )


    return unmatched_triggers, unmatched_actions


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

filtered_df.to_csv(INPUT_FILE, index=False)

print(f"NUMBER OF ROWS BEFORE FILTERING: {len(df)}")
print(f"NUMBER OF ROWS AFTER FILTERING: {len(filtered_df)}")

df["originalTrigger"] = df["triggerDesc"]
df["originalAction"] = df["actionDesc"]

unmatched_triggers = []
unmatched_actions = []
unmatched_triggers, unmatched_actions = normalize(df)

# ===============================
# SAVE OUTPUT
# ===============================

folder = os.path.dirname(INPUT_FILE)
base = os.path.basename(INPUT_FILE).replace(".csv", "")

OUTPUT_FILE = os.path.join(
    folder,
    f"{base}_normalized_for_oversampling.csv"
)

df.to_csv(OUTPUT_FILE, index=False)

print(f"SAVED: {OUTPUT_FILE}")

# ===============================
# VALIDATION / REPORT
# ===============================

print("\n========== NORMALIZATION REPORT ==========\n")

print("Loading dataset...")

df = pd.read_csv(OUTPUT_FILE)

unique_triggers = df["triggerDesc"].dropna().unique()
unique_actions = df["actionDesc"].dropna().unique()

print(f"UNIQUE TRIGGERS NORMALIZED: {unique_triggers.__len__()}")
print(f"UNIQUE ACTIONS NORMALIZED: {unique_actions.__len__()}")

# Trigger stats
unique_triggers = df["originalTrigger"].dropna().unique()
unique_unmatched_triggers = list(set(unmatched_triggers))

total_unique_triggers = len(unique_triggers)
unmatched_unique_triggers = len(unique_unmatched_triggers)

unique_trigger_coverage = 100 * (total_unique_triggers - unmatched_unique_triggers) / total_unique_triggers

# Action stats
unique_actions = df["originalAction"].dropna().unique()
unique_unmatched_actions = list(set(unmatched_actions))

total_unique_actions = len(unique_actions)
unmatched_unique_actions = len(unique_unmatched_actions)

unique_action_coverage = 100 * (total_unique_actions - unmatched_unique_actions) / total_unique_actions

print("\n========== UNIQUE COVERAGE ==========\n")

print(f"UNIQUE TRIGGERS NORMALIZED: "
      f"{total_unique_triggers - unmatched_unique_triggers}/{total_unique_triggers} "
      f"({unique_trigger_coverage:.2f}%)")

print(f"UNIQUE ACTIONS NORMALIZED: "
      f"{total_unique_actions - unmatched_unique_actions}/{total_unique_actions} "
      f"({unique_action_coverage:.2f}%)")

print("\n========== ALL UNMATCHED TRIGGERS ==========\n")

if unique_unmatched_triggers.__len__() == 0:
    print("No unmatched triggers")
else:
    print(unique_unmatched_triggers)

print("\n========== ALL UNMATCHED ACTIONS ==========\n")

if unique_unmatched_actions.__len__() == 0:
    print("No unmatched actions")
else:
    print(unique_unmatched_actions)

print("==========================================")