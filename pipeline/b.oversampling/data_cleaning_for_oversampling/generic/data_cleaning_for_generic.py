import pandas as pd
import os
import re


# ===============================
# CONFIG
# ===============================

INPUT_FILE = "../../../../dataset/filtered_data/device_category_data/Generic_Other.csv"


# ===============================
# TRIGGER RULES (GENERALIZED)
# ===============================

TRIGGER_RULES = [
    (r".*every single day.*time set by you.*", ", IF a specified time occurs"),
    (r".*sunset.*", ", IF it is sunset, within a specified time"),
    (r".*sunrise.*", ", IF it is sunrise, within a specified time"),
    (r".*(alexa trigger|ok google|google assistant).*", "IF a voice assistant is activated with a specific phrase"),
    (r".*button.*pressed.*", "IF a button is pressed"),
    (r".*device.*temperature.*above.*threshold.*", "IF a device detects temperature above a specified threshold"),
    (r".*device.*temperature.*below.*threshold.*", "IF a device detects temperature below a specified threshold"),
    (r".*current weather condition.*rain|snow|cloudy|clear.*", "IF the weather conditions change"),
    (r".*ifttt (an|any) email.*", "IF an email is received with a specified hashtag"),
    (r".*enter.*area.*", "IF the user enters a specified area"),
    (r".*exit.*area.*", "IF the user exits a specified area"),
    (r".*voice assistant.*activated.*", "IF a voice assistant is activated with a specific phrase"),
    (r".*specific days.*", "IF a specific day of the week occurs"),
    (r".*(set|change|adjust).*(temperature|setpoint|target|level|degree|°|freezer|water heater).*", ", THEN set device temperature"),
    (r".*wemo switch.*", "IF a smart device is turned on"),

    (r".*(motion).*doorbell.*|.*detects motion.*", "IF motion is detected"),

    (r".*alarm.*goes off.*|.*alarm.*triggered.*", "IF an alarm is triggered"),

    (r".*tap.*gesture.*|.*custom ifttt task.*", "IF a custom gesture or automation is triggered"),

    (r".*set to away.*|.*home.*away.*", "IF home presence is set to away"),

    (r".*wemo.*motion.*|.*detects new motion.*", "IF motion is detected"),

    (r".*wemo.*switch.*turned off.*", "IF a smart device is turned off"),

    (r".*once an hour.*|.*:00.*:15.*:30.*:45.*", "IF a scheduled time interval occurs"),

    (r".*click.*flic.*|.*button.*flic.*", "IF a button is pressed"),

    (r".*enter or exit.*area.*|.*enter.*exit.*area.*", "IF the user enters or exits a specified area"),

    (r".*ok google.*|.*google assistant.*phrase.*", "IF a voice assistant is activated with a specific phrase"),

    (r".*arlo.*motion.*|.*camera.*motion.*", "IF motion is detected"),

    (r".*minutes before.*calendar.*keyword.*", "IF shortly before a calendar event with a specific keyword"),

    (r".*set to home.*|.*home.*present.*", "IF home presence is set to home"),

    (r".*temperature.*rise.*above.*threshold.*", "IF temperature rises above a specified threshold"),

    (r".*android.*connects.*wifi.*", "IF a device connects to a WiFi network"),

    (r".*ifttt.*apilio.*event.*", "IF an external automation event is received"),

    (r".*temperature.*drops.*below.*threshold.*|.*drops.*below.*value.*", "IF temperature drops below a specified threshold"),

    (r".*wemo.*long press.*|.*turned on or off.*long press.*", "IF a smart switch is long-pressed"),

    (r".*ring.*doorbell.*|.*rings.*doorbell.*", "IF a doorbell is rung"),

    (r".*android.*disconnects.*wifi.*", "IF a device disconnects from a WiFi network"),

    (r".*new notification.*android.*|.*app.*notification.*", "IF a notification is received from a mobile app"),



]


# ===============================
# ACTION RULES (GENERALIZED)
# ===============================

ACTION_RULES = [

    (r".*turn a wemo.*on.*", ", THEN turn on a specified device."),
    (r".*turn on.*|switch on.*|power on.*duration.*", ", THEN turn on a specified device for a specified duration."),
    (r".*turn on.*|switch on.*|power on.*", ", THEN turn on a specified device."),

    (r".*turn a wemo.*off.*", ", THEN turn on a specified device."),
    (r".*turn off.*|switch off.*|power off.*", ", THEN turn off a device for a specified time."),
    (r".*turn off.*|switch off.*|power off.*", ", THEN turn off a device."),

    (r".*toggle.*|on or off.*",", THEN toggle a device on"),

    (r".*(eco|quiet|vacation|holiday).*mode.*", ", THEN change the device operating mode to eco"),
    (r".*(auto|manual).*mode.*", ", THEN change the device operating mode to auto"),

    (r".*scene.*", ", THEN activate to a specified scene."),

    (r".*(start|resume).*robot.*", ", THEN start the robot."),
    (r".*(stop|pause|dock).*robot.*", ", THEN pause the robot."),

    (r".*lock.*|unlock.*",", THEN unlock a device"),

    (r".*(brightness|dimmer|level).*",", THEN set brightness to a specified level"),

    # INFRARED
    (r".*infrared|ir signal.*",", THEN send a notify to a smart device."),

    # PLUG / OUTLET
    (r".*turn on.*plug|outlet|socket.*", ", , THEN turn on the specified plug."),
    (r".*turn off.*plug|outlet|socket.*", ", , THEN turn off the specified plug."),

    (r".*pause.*thermosmart.*|.*pause.*device.*", ", THEN pause the thermostat."),

    (r".*unpause.*thermosmart.*|.*resume.*program.*", ", THEN resume the thermostat."),

    (r".*turn.*maker.*off , THEN on.*|.*off , THEN on.*", ", THEN restart the device"),

    (r".*turn.*(product|device|maker|ac|a/c).*off.*", ", THEN turn off a device"),

    (r".*turn.*(product|device|maker|ac|a/c).*on.*", ", THEN turn on a device"),

    (r".*set.*operating mode.*water heater.*|.*heatmiser.*timer boost.*", ", THEN change heating or water heater mode"),

    (r".*activate.*timer boost.*|.*boost.*timer.*", ", THEN temporarily boost heating"),

    (r".*trigger.*bot.*press.*switch.*", ", THEN trigger an automation bot"),

    (r".*open.*hubitat.*device.*|.*open.*device.*", ", THEN open a connected device"),

    (r".*close.*hubitat.*device.*|.*close.*device.*", ", THEN close a connected device"),

    (r".*change.*hubitat.*mode.*", ", THEN change home system mode"),

    (r".*clean.*", ", THEN let the robot clean a specified room."),

    (r".*activate.*away mode.*|.*activate.*home mode.*", ", THEN change home presence mode"),

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
        "NaN",
        case=False,
        na=False
    )
]

filtered_df = filtered_df[
    ~filtered_df["actionDesc"].str.contains(
        r".*curtain.*|.*lutron shade.*",
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
