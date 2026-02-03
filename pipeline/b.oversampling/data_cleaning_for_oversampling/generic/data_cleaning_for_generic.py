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
    (r".*enter.*area.*", "IF the user enters a specified area"),
    (r".*exit.*area.*", "IF the user exits a specified area"),
    (r".*home.*set to away.*|.*set to away.*|.*away mode.*", "IF home presence is set to away"),
    (r".*home.*set to home.*|.*set to home.*|.*home mode.*", "IF home presence is set to home"),
    (r".*(everyone left|family member).*", "IF everyone leaves the house"),
    (r".*at least one family member.*come home.*", "IF someone in the family comes home"),
    (r".*movement.*|.*detects motion.*|.*user.*presence.*detected.*|.*no longer detected.*|.*motion.*detected.*", "IF your device detects motion"),
    (r".*geo-circles*", "IF a vehicle exits"),
    (r".*press.*button.*|.*tap.*gesture.*|.*flic.*|.*iHome Enhance button.*|.*Swann Video Doorbell button.*|.*lawnmower.*",
     "IF a button is pressed [FREE]"),
    (r".*pusher.*watch.*", "IF a smartwatch button is pressed [FREE]"),
    (r".*(alexa trigger).*|.*ok google.*|.*google assistant.*", "IF a voice assistant is activated with a specific phrase [FREE]"),
    (r".*(motion).*sensor.*|.*camera.*detects.*motion.*|.*wemo.*motion.*|.*swann.*motion.*|.*wyze.*cam.*|.*z-wave.*", "IF motion is detected"),
    (r".*motion.*sensor.*becomes clear.*|.*no movement.*", "IF motion stops or sensor clears"),
    (r".*(device|switch|zone|product).*turn.*on.*|.*switched on.*|.*powered on.*", "IF a device is turned on [FREE]"),
    (r".*(device|switch|zone|product).*turn.*off.*|.*switched off.*|.*unplugged.*", "IF a device is turned off"),
    (r".*(on or off).*|.*toggle.*relay.*", "IF a device changes state"),
    (r".*(timer).*goes off.*", "IF a timer expires [FREE]"),
    (r".*wemo insight switch.*standby.*", "IF a device enters standby mode"),
    (r".*temperature.*above.*|.*rise above.*", "IF temperature rises above a threshold"),
    (r".*exceeds.*|.*ambient.*exceeds.*", "IF temperature exceeds a threshold"),
    (r".*temperature.*below.*|.*drops below.*|.*ambient.*below.*", "IF temperature drops below a threshold"),
    (r".*(humidity).*above.*|.*below.*|.*returns within normal.*", "IF humidity crosses a threshold"),
    (r".*(air pollution|air quality|pm2\.5|co2|voc|radon).*", "IF air pollution or gas levels exceed a threshold"),
    (r".*uv index.*above.*", "IF UV index exceeds a threshold"),
    (r".*(wind speed|gust).*above.*", "IF wind speed exceeds a threshold"),
    (r".*(leak|water).*detect.*", "IF a water leak is detected"),
    (r".*(smoke|carbon monoxide).*detect.*|.*nest protect.*detects.*", "IF dangerous gas or smoke is detected"),
    (r".*new notification.*android.*|.*app.*notification.*|.*SMS.*|.*incoming call.*|.*voicemail.*", "IF a mobile notification or call event occurs [FREE]"),
    (r".*ifttt.*(email|apilio|webhook|external).*", "IF an external event is received [FREE]"),
    (r".*telegram.*key phrase.*|.*@ifttt.*bot.*", "IF a keyword is sent via messaging bot [FREE]"),
    (r".*calendar.*start.*|.*meeting starts.*", "IF a calendar event starts [FREE]"),
    (r".*calendar.*end.*|.*meeting ends.*", "IF a calendar event ends [FREE]"),
    (r".*new task.*|.*to do list.*|.*reminder.*", "IF a new task or reminder is added [FREE]"),
    (r".*spreadsheet.*cell.*(updated|changed|modified).*", "IF a spreadsheet cell is updated [FREE]"),
    (r".*armed.*|.*disarmed.*|.*alarm.*", "IF an alarm or security event occurs"),
    (r".*security device.*enters.*home mode.*", "IF a security device enters home mode"),
    (r".*lock.*locked.*", "IF a security device changes state to locked"),
    (r".*unlocked.*", "IF a security device changes state to unlocked"),
    (r".*mode.*armed.*", "IF the mode is armed"),
    (r".*mode.*disarmed.*", "IF the mode is disarmed"),
    (r".*(electricity price|price|dynamic pricing|hourly pricing|real time market).*", "IF energy pricing changes"),
    (r".*carbon.*", "IF carbon intensity changes"),
    (r".*(power|watts).*above.*|.*consumption.*", "IF power consumption exceeds a threshold"),
    (r".*(exporting solar).*", "IF solar power is exported to the grid"),
    (r".*ohmhour.*start.*|.*end.*", "IF a demand response event starts or ends"),
    (r".*sunset.*", "IF event occurs relative to sunset"),
    (r".*sunrise.*", "IF event occurs relative to sunrise"),
    (r".*every single day.*|.*daily at.*|.*once a day.*", "IF event occurs daily at a specific time"),
    (r".*once an hour.*|.*:00|:15|:30|:45", "IF event occurs hourly at specified minutes"),
    (r".*specific days of the week.*", "IF event occurs on specific weekdays"),
    (r".*once a year.*", "IF event occurs once a year"),
    (r".*every month.*", "IF event occurs monthly"),
    (r".*weather.*condition.*|.*forecast.*", "IF weather or forecast conditions are met"),
    (r".*doorbell.*motion.*", "IF motion is detected at doorbell"),
    (r".*audio.*event.*", "IF audio event is detected"),
    (r".*contact sensor.*opens.*|.*door.*opens.*|.*door.*moves.*|.*door.*opened.*|.*door sensor.*", "IF the door is opened"),
    (r".*contact sensor.*closes.*|.*door.*closes.*|.*door.*(closed|locked).*", "IF the door is closed"),
    (r".*sensor.*activated.*", "IF a sensor is activated"),
    (r".*wemo.*light.*switches.*on.*", "IF a light switches on"),
    (r".*wemo.*light.*switches.*off.*", "IF a light switches off"),
    (r".*dimmer.*turned off.*", "IF a dimmer is turned off"),
    (r".*android device.*connects.*(wifi|bluetooth).*", "IF Android device connects from the network or Bluetooth"),
    (r".*android device.*disconnects.*(wifi|bluetooth).*","IF Android device disconnects from the network or Bluetooth"),
    (r".*google wifi.*connects.*", "IF a device connects to Google Wifi"),
    (r".*google wifi.*disconnects.*", "IF a device disconnects from Google Wifi"),
    (r".*robot.*completes.*job.*|starts.*job.*|litter robot.*", "IF a robot completes or starts a job"),
    (r".*homey.*flow.*started.*|.*routine.*|.*nexia automation.*", "IF a routine is started"),
    (r".*ecobee.*thermostat.*temperature.*", "IF the thermostat senses temperature changes"),
    (r".*heat mode.*", "IF a specific device mode is set to heat mode"),
    (r".*ecobee.*thermostat.*humidity.*", "IF the thermostat senses humidity changes"),
    (r".*wemo insight.*|.*ce trigger.*", "IF prices cost exceeds threshold"),
    (r".*low battery.*", "IF a device is low on battery"),
    (r".*tweet.*@mentioned.*|#.*|at location.*|.*tag.*", "IF a message event occurs"),
    (r".*facebook.*status.*|tagged.*photo.*", "IF a status or photo tagging occurs"),
    (r".*cell*", "IF a cell of the spreadsheet is updated"),
    (r".*every time.*|.*triggered.*", "IF general event occurs"),
]

# ===============================
# ACTION RULES (GENERALIZED)
# ===============================

ACTION_X = [
    ["eco mode", "comfort mode", "turbo mode"],
    ["low power mode", "default mode", "maximum power"],
    ["energy saving mode", "auto mode", "boost mode"]
]

ACTION_Y = [
    ["for a specific amount of time", "", "for the entire day"],
    ["for a limited time", "until stopped", "indefinitely"],
    ["temporarily", "until manually changed", "continuously"]
]

ACTION_Z = [
    ["at low power", "at medium power", "at maximum power"],
    ["at minimum power ", "at default power", "at high power"],
    ["to power saving mode", "at moderate power", "at boosted power"]
]

ACTION_W = [
    ["after a specified amount of time", "", "after a large amount of time"],
    ["after a fixed delay", "without delay", "after several hours"],
    ["after a defined period ", "immediately", "much later"]
]

ACTION_RULES = [
    (r".*turn a wemo.*on.*", ", THEN turn on a specific smart device in [ACTION_X]."),
    (r".*turn on.*|switch on.*|power on.*duration.*", ", THEN turn on a device for a specific duration in [ACTION_X]."),
    (r".*turn on.*|switch on.*|power on.*", ", THEN turn on a specified device [ACTION_Y]."),
    (r".*turn a wemo.*off.*", ", THEN turn off a specified device [ACTION_W]."),
    (r".*turn off.*|switch off.*|power off.*", ", THEN switch off a device [ACTION_W]."),
    (r".*toggle.*|on or off.*", ", THEN toggle a device on in [ACTION_X]."),
    (r".*(eco|quiet|vacation|holiday).*mode.*", ", THEN change the device operating mode to [ACTION_X]."),
    (r".*(auto|manual).*mode.*", ", THEN change the device operating mode to [ACTION_X]."),
    (r".*scene.*", ", THEN activate to a specified scene [ACTION_Z]."),
    (r".*(start|resume).*robot.*", ", THEN start the robot in [ACTION_X]."),
    (r".*(stop|pause|dock).*robot.*", ", THEN pause the robot [ACTION_Y]."),
    (r".*lock.*|unlock.*",", THEN unlock a device [ACTION_W]."),
    (r".*(brightness|dimmer|level).*", ", THEN set brightness [ACTION_Z]."),
    (r".*infrared|ir signal.*",", THEN send a notify to a smart device [ACTION_W]."),
    (r".*turn on.*plug|outlet|socket.*", ", THEN turn on the specified plug [ACTION_Y]."),
    (r".*turn off.*plug|outlet|socket.*", ", THEN turn off the specified plug [ACTION_W]."),
    (r".*pause.*thermosmart.*|.*pause.*device.*", ", THEN pause the thermostat [ACTION_W]."),
    (r".*unpause.*thermosmart.*|.*resume.*program.*", ", THEN resume the thermostat in [ACTION_X]."),
    (r".*turn.*maker.*off , THEN on.*|.*off , THEN on.*", ", THEN switch on a specific device [ACTION_X]."),
    (r".*turn.*(product|device|maker|ac|a/c).*off.*", ", THEN turn off a smart device [ACTION_W]."),
    (r".*turn.*(product|device|maker|ac|a/c).*on.*", ", THEN turn on a smart device [ACTION_Y]."),
    (r".*set.*operating mode.*water heater.*|.*heatmiser.*timer boost.*", ", THEN boost heating mode [ACTION_Y]."),
    (r".*activate.*timer boost.*|.*boost.*timer.*", ", THEN boost heating [ACTION_Y]."),
    (r".*trigger.*bot.*press.*switch.*", ", THEN trigger an automation bot in [ACTION_X]."),
    (r".*open.*hubitat.*device.*|.*open.*device.*", ", THEN open a connected device [ACTION_Y]."),
    (r".*close.*hubitat.*device.*|.*close.*device.*", ", THEN close a connected device [ACTION_W]."),
    (r".*change.*hubitat.*mode.*|.*activate.*away mode.*|.*activate.*home mode.*", ", THEN change home system mode to [ACTION_X]."),
    (r".*clean.*", ", THEN let the robot clean a specific room [ACTION_Z].")
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
        r".*subscribed.*|.*problem occurs.*|.*Foursquare.*|.*cloudBit.*|.*rf bridge.*|.*membership.*|.*aura.*|"
        r".*location check-in.*|.*group decision.*|.*prayer.*|.*youtube.*|.*fibaro.*|.*live stream.*",
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
