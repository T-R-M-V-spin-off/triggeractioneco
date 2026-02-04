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
    (r".*device.*turned on.*|device turned on", "IF a device is turned on [FREE]"), #FREE
    (r".*calendar.*event.*", "IF shortly before a calendar event with a specific keyword occurs [FREE]"), #FREE
    (r".*(alexa trigger|ok google|google assistant).*|.*voice assistant.*activated.*", "IF a voice assistant is activated with a specific phrase [FREE]"), #FREE
    (r".*before.*time-of-day.*peak rates.*start.*", "IF before time-of-day peak rates start"),
    (r".*every day at a specified time.*|.*every.*day.*specific time.*", "IF every day at a specified time [FREE]"), #FREE
    (r".*message.*key phrase.*", "IF a message containing a key phrase is sent [FREE]"), #FREE
    (r".*home.*set to away.*", "IF every time no one is at home"),
    (r".*home.*set to home.*", "IF every time someone is at home"),
    (r".*user enters a specified area.*|.*enter.*area.*", "IF the user enters a specified area"),
    (r".*user exits a specified area.*|.*exit.*area.*", "IF the user exits a specified area"),
    (r".*button.*pressed.*|.*press the button.*", "IF a button is pressed [FREE]"), #FREE
    (r".*a/c unit.*temperature.*below.*value.*", "IF an AC unit detects temperature below a specified value"),
    (r".*android device.*connects to wifi.*", "IF an Android device connects to a specified Wi-Fi network"),
    (r".*android device.*disconnects from wifi.*", "IF an Android device disconnects from a specified Wi-Fi network"),
    (r".*fitbit.*logs.*sleep.*", "IF a device logs new sleep data"),
    (r".*every time.*connects.*", "IF your Android device connects to a specified wifi network"),
    (r".*every time.*disconnects.*", "IF your Android device disconnects from a specified wifi network"),
    (r".*ifttt receives.*", "IF ifttt receives a specified event from an external service [FREE]"), #FREE
    (r".*myfox.*security system.*partially armed.*", "IF a security system is partially armed"),
    (r".*smartthings.*device.*switched off.*", "IF a smart device is switched off [FREE]"), #FREE
    (r".*smartthings.*device.*switched on.*", "IF a smart device is switched on [FREE]"), #FREE
    (r".*routine.*activated.*", "IF a routine is activated [FREE]"), #FREE
    (r".*air purifier.*air quality.*", "IF an air purifier detects a specific air quality level"),
    (r".*room temperature.*condition.*threshold.*", "IF the room temperature satisfies a threshold condition"),
    (r".*solar power.*drops below.*", "IF solar power drops below a specified value"),
    (r".*temperature.*specific device.*exceeds.*threshold.*", "IF a device temperature exceeds a threshold"),
    (r".*temperature.*satisfies.*threshold.*", "IF the house temperature satisfies a threshold condition"),
    (r".*device.*temperature.*above.*threshold.*|.*above a value you specify.*", "IF a device detects temperature above a specified threshold"),
    (r".*device.*temperature.*below.*threshold.*", "IF a device detects temperature below a specified threshold"),
    (r".*indoor temperature.*dropping below.*threshold.*", "IF indoor temperature drops below a specified threshold"),
    (r".*indoor temperature.*rising above.*threshold.*", "IF indoor temperature rises above a specified threshold"),
    (r".*sunrise.*", "IF a set number of minutes have passed after the sunrise"),
    (r".*sunset.*", "IF a set number of minutes have passed after the sunset"),
    (r".*unit.*turned on.*", "IF a unit has been turned on and a set number of minutes have passed"),
    (r".*current weather condition.*rain|snow|cloudy|clear.*", "IF the weather conditions change"),
    (r".*local humidity.*above.*value.*", "IF local humidity is above a specified value"),
    (r".*local temperature.*drops below.*value.*", "IF local temperature is below a specified value"),
    (r".*local temperature.*rises above.*value.*", "IF local temperature is above a specified value"),
    (r".*room enters manual mode.*", "IF every time a room enters manual mode"),
    (r".*particulate matter.*", "IF the air quality is below a specified value")
]

ACTION_RULES = [
    (r".*changes modes.*auto.*sleep.*", ", THEN change the AC unit mode to [ACTION_X]."),
    (r".*turns off.*air purifier.*", ", THEN turns off the air purifier [ACTION_W]."),
    (r".*turns on.*air purifier.*", ", THEN turns on the air purifier [ACTION_Y]."),
    (r".*turns.*display.*on or off.*", ", THEN turns the AC unit display on [ACTION_Y]."),
    (r".*turns.*fan.*speed.*(low|medium|high).*", ", THEN sets the fan to [ACTION_Z] speed."),
    (r".*disable.*timer.*", ", THEN disable the indicated timer [ACTION_Y]."),
    (r".*enable.*timer.*", ", THEN enable the indicated timer [ACTION_Y]."),
    (r".*econo mode.*enable|disable.*|.*holiday mode.*enable|disable.*", ", THEN enable [ACTION_X] on the AC unit."),
    (r".*execute.*scene.*", ", THEN set the mode of the air conditioner to [ACTION_X]."),
    (r".*set.*mode.*air conditioner.*", ", THEN set the mode of the air conditioner to [ACTION_X]."),
    (r".*turn off.*air conditioner.*", ", THEN turn off the air conditioner [ACTION_W]."),
    (r".*turn off.*a/c.*specified room.*", ", THEN turn off the air conditioner in a specified room [ACTION_W]."),
    (r".*turn off.*daikin.*ac unit.*|.*turn.*intesishome.*a/c.*off.*", ", THEN turn off the AC unit [ACTION_W]."),
    (r".*turn on.*air conditioner.*", ", THEN turn on the air conditioner [ACTION_Y]."),
    (r".*turn on.*a/c.*specified room.*comfort mode.*", ", THEN turn on the air conditioner in a specified room in [ACTION_X]."),
    (r".*turn on.*daikin.*ac unit.*|.*turn.*intesishome.*a/c.*on.*", ", THEN turn on the AC unit [ACTION_X]."),
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
