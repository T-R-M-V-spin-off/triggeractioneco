import pandas as pd
import os
import re


# ===============================
# CONFIG
# ===============================

INPUT_FILE = "../../../../dataset/filtered_data/device_category_data/Thermostat_Heating.csv"


# ===============================
# NORMALIZATION RULES
# ===============================

TRIGGER_RULES = [
    (r".*device.*turned on.*|device turned on", "IF a device is turned on [FREE]"), #FREE
    (r".*electricity.*cheap.*|.*electricity price.*lowest.*", "IF electricity prices are low"),
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
    (r".*sunrise.*", "IF it is sunrise"),
    (r".*device.*turned on.*|device turned on", "IF a device is turned on [FREE]"),
    (r".*calendar.*event.*", "IF shortly before a calendar event with a specific keyword occurs [FREE]"),
    (r".*(alexa trigger|ok google|google assistant).*",
     "IF a voice assistant is activated with a specific phrase [FREE]"),
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
    (r".*particulate matter.*", "IF the air quality is below a specified value"),
    (r".*temperature.*(drops below|falls below|less than).*", "IF temperature drops below a threshold"),
    (r".*temperature.*(rises above|exceeds|greater than).*", "IF temperature rises above a threshold"),
    (r".*wirelesstag.*temperature.*returns.*", "IF WirelessTag temperature returns to normal range"),
    (r".*nest protect.*carbon monoxide.*", "IF Nest Protect detects carbon monoxide"),
    (r".*nest protect.*smoke.*", "IF Nest Protect detects smoke"),
    (r".*awair.*dust.*", "IF Awair detects high dust concentration"),
    (r".*awair.*(co2|carbon dioxide).*", "IF Awair detects high CO2 levels"),
    (r".*awair.*voc.*", "IF Awair detects high VOC levels"),
    (r".*co2.*level.*rises above.*", "IF CO2 level rises above a threshold"),
    (r".*alarm.*goes off.*|.*alarm event.*", "IF an alarm triggers"),
    (r".*alarm.*armed.*", "IF an alarm is armed"),
    (r".*phyn.*alert.*", "IF Phyn detects an alert"),
    (r".*camera.*motion event.*", "IF a camera detects motion"),
    (r".*wirelesstag.*too wet.*", "IF WirelessTag detects moisture"),
    (r".*eight app.*(night|day) mode.*", "IF Night or Day mode is toggled in Eight App"),
    (r".*ewelink.*switch.*(turned on|turned off).*", "IF eWeLink switch is toggled"),
    (r".*(knx|binary switch).*turned off.*", "IF a KNX or binary switch is turned off"),
    (r".*tado.*away mode.*", "IF Tado switches to Away mode"),
    (r".*smartthings.*motion.*", "IF SmartThings detects motion"),
    (r".*aura air.*sensor.*reached.*", "IF Aura Air sensor reaches a specific value"),
    (r".*ecobee.*schedule.*overridden.*", "IF Ecobee schedule is overridden"),
    (r".*ecobee.*humidity.*greater.*", "IF Ecobee detects high humidity"),
    (r".*(jaguar|kronaby|festina|lotus).*pusher.*", "IF a smartwatch button is pressed"),
    (r".*withings.*get into bed.*", "IF Withings detects user getting into bed"),
    (r".*focus.*time.*session.*", "IF a Focus Time session starts or ends"),
    (r".*harvest.*new project.*", "IF a new project is created on Harvest"),
    (r".*send.*sms.*android.*", "IF an SMS is sent from Android"),
    (r".*receive.*sms.*android.*", "IF an SMS is received on Android"),
    (r".*tweet.*hashtag.*", "IF a new tweet with a specific hashtag is posted"),
    (r".*super chat.*live stream.*", "IF a YouTube Super Chat is received"),
    (r".*ballotpedia.*breaking news.*", "IF Ballotpedia reports breaking news"),
    (r".*npr.*new story.*", "IF a new story is published on NPR"),
    (r".*strava.*complete.*activity.*", "IF a new activity is completed on Strava"),
    (r".*electricity price.*(stays|falls|rises).*", "IF electricity price fluctuates beyond threshold"),
    (r".*is christmas.*", "IF it is Christmas"),
    (r".*plage tarifaire.*commence.*", "IF an energy tariff period starts"),
    (r".*carbon.*clean.*dirty.*", "IF carbon intensity changes"),
    (r".*peak time savings.*", "IF peak energy savings hours are announced"),
    (r".*electricity.*expensive.*", "IF electricity prices are high"),
    (r".*electricity.*cheap.*", "IF electricity prices are low"),
    (r".*time-of-day.*peak.*start.*|.*time-of-day.*peak.*end.*", "IF peak energy rates start or end"),
    (r".*price change.*|.*average price changes.*|.*dynamic pricing.*", "IF electricity prices change"),
    (r".*ohmhour.*start.*|.*ohmhour.*end.*", "IF a demand response event starts or ends"),
    (r".*carbon intensity.*changes.*", "IF carbon intensity changes"),
    (r".*jaguar.*", "IF a Jaguar watch button is pressed"),
    (r".*kronaby.*", "IF a Kronaby watch button is pressed"),
    (r".*festina.*", "IF a Festina watch button is pressed"),
    (r".*lotus.*", "IF a Lotus watch button is pressed"),
    (r".*eight app.*", "IF Night or Day mode is toggled in Eight App"),
    (r".*harvest.*", "IF a new project is created on Harvest"),
    (r".*strava.*", "IF a new activity is completed on Strava"),
    (r".*npr.*", "IF a new story is published on NPR"),
    (r".*aura air.*", "IF Aura Air sensor reaches a specific value"),
    (r".*wirelesstag.*return.*", "IF WirelessTag temperature returns to normal range"),
    (r".*wirelesstag.*wet.*", "IF WirelessTag detects moisture"),
    (r".*ecobee.*smart.*", "IF Ecobee switches to Smart Home/Away profile"),
    (r".*smartthings.*temperature.*", "IF SmartThings detects high temperature"),
    (r".*awair.*dust.*", "IF Awair detects high dust concentration"),
    (r".*awair.*carbon.*", "IF Awair detects high CO2 levels"),
    (r".*awair.*voc.*", "IF Awair detects high VOC levels"),
    (r".*carbon dioxide.*rises.*", "IF CO2 level rises above a threshold"),
    (r".*calendar.*keyword.*before.*start.*", "IF shortly before a calendar event with a keyword"),
    (r".*calendar.*ending.*keyword.*", "IF shortly after a calendar event with a keyword"),
    (r".*focus time.*started.*", "IF a focus session starts"),
    (r".*focus time.*end.*", "IF a focus session ends"),
    (r".*once an hour.*(:00|:15|:30|:45).*", "IF every hour at quarter intervals"),
    (r".*npr.*new story.*", "IF a new news story is published"),
    (r".*new item.*feed.*contains.*", "IF a feed item contains a keyword"),
    (r".*submit.*new note.*", "IF a new note is created"),
    (r".*connects.*google wifi.*", "IF a device connects to WiFi"),
    (r".*disconnects.*google wifi.*", "IF a device disconnects from WiFi"),
    (r".*tp-link.*connect.*", "IF a device connects to a router"),
    (r".*tp-link.*disconnect.*", "IF a device disconnects from a router"),
    (r".*new reminder.*added.*", "IF a new reminder is added"),
    (r".*reminder.*completed.*", "IF a reminder is completed"),
    (r".*add.*item.*to do list.*", "IF an item is added to a to-do list"),
    (r".*lock.*lock n go.*", "IF a door is locked"),
    (r".*auto unlock.*|.*unlocked.*", "IF a door is unlocked"),
    (r".*mode.*armed.*", "IF a security mode is armed"),
    (r".*mode.*disarmed.*", "IF a security mode is disarmed"),
    (r".*alarm panel.*armed.*", "IF an alarm system is armed"),
    (r".*alarm panel.*disarmed.*", "IF an alarm system is disarmed"),
    (r".*myfox.*armed.*", "IF a security system is armed"),
    (r".*adt.*arm state.*", "IF a security system state changes"),
    (r".*smanos.*armed.*", "IF a security system is armed"),
    (r".*abode.*mode.*changed.*", "IF a security mode changes"),
    (r".*door.*opened.*|.*door sensor.*open.*", "IF a door is opened"),
    (r".*door.*closed.*|.*window.*closed.*", "IF a door or window is closed"),
    (r".*window.*opens.*|.*window.*moves.*", "IF a window is opened or moved"),
    (r".*motion sensor.*detect.*", "IF motion is detected"),
    (r".*weatherflow.*temperature.*below.*|.*spotter.*below.*", "IF temperature drops below a threshold"),
    (r".*weatherflow.*temperature.*above.*|.*spotter.*above.*", "IF temperature rises above a threshold"),
    (r".*forecasted.*high.*above.*", "IF tomorrow's high temperature exceeds a threshold"),
    (r".*forecasted.*low.*below.*", "IF tomorrow's low temperature drops below a threshold"),
    (r".*current temperature.*above.*|.*current temperature.*below.*", "IF temperature crosses a threshold"),
    (r".*awair.*carbon dioxide.*", "IF CO2 level exceeds a threshold"),
    (r".*awair.*dust.*|.*pm2.5.*", "IF particulate matter exceeds a threshold"),
    (r".*awair.*voc.*", "IF VOC level exceeds a threshold"),
    (r".*air quality.*above.*|.*air quality.*below.*", "IF air quality crosses a threshold"),
    (r".*global air pollution.*", "IF air pollution exceeds a threshold"),
    (r".*fitbit.*sleep.*", "IF new sleep data is logged"),
    (r".*daily calorie.*goal.*", "IF a daily calorie goal is achieved"),
    (r".*start sleep.*zeeq.*", "IF sleep tracking is started"),
    (r".*every family member.*left.*", "IF everyone leaves home"),
    (r".*presence.*detected.*", "IF user presence is detected"),
    (r".*presence.*no longer.*", "IF user presence ends"),
    (r".*scene.*launched.*|.*routine.*activated.*", "IF an automation routine is activated"),
    (r".*homey.*flow.*started.*", "IF a home automation flow starts"),
    (r".*aura air.*sensor.*value.*", "IF an air sensor reaches a threshold"),
    (r".*festina.*pusher.*|.*jaguar.*pusher.*|.*kronaby.*pusher.*|.*lotus.*pusher.*",
     "IF a smartwatch button is pressed"),
    (r".*air monitor.*button.*", "IF a monitoring device button is pressed"),
    (r".*foursquare.*check in.*|.*swarm.*check in.*", "IF the user checks in at a location"),
    (r".*strava.*activity.*", "IF a fitness activity is completed"),
    (r".*harvest.*project.*", "IF a new project is created"),
    (r".*switchbot.*motion.*", "IF motion is detected"),
    (r".*switchbot.*temperature.*|.*switchbot.*humidity.*", "IF environmental conditions change"),
    (r".*wirelesstag.*temperature.*above.*", "IF tag temperature exceeds a threshold"),
    (r".*wirelesstag.*temperature.*below.*", "IF tag temperature drops below a threshold"),
    (r".*wirelesstag.*humidity.*too wet.*", "IF moisture level is too high"),
    (r".*particle.publish.*|.*interesting event.*", "IF a custom IoT event is received"),
    (r".*rescuetime.*alert.*", "IF a productivity alert is triggered"),
    (r".*caavo.*search.*", "IF a media search is performed")
]

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
    ["at low value", "at medium value", "at maximum value"],
    ["at minimum value ", "at default value", "at high value"],
    ["to power saving mode", "at moderate value", "at boosted value"]
]

ACTION_W = [
    ["after a specified amount of time", "", "after a large amount of time"],
    ["after a fixed delay", "without delay", "after several hours"],
    ["after a defined period ", "immediately", "much later"]
]

ACTION_RULES = [
    (r".*set.*temperature.*heatmiser.*", ", THEN sets the temperature [ACTION_Z] on thermostat."),
    (r".*set.*target temperature.*window ac.*", ", THEN sets the target temperature [ACTION_Z] on the window air conditioner."),
    (r".*set.*refrigerator temperature.*", ", THEN sets the refrigerator temperature [ACTION_Z]."),
    (r".*set.*freezer temperature.*", ", THEN sets the freezer temperature [ACTION_Z]."),
    (r".*set.*temperature.*water heater.*", ", THEN sets the water heater temperature [ACTION_Z]."),
    (r".*set.*thermostat.*temperature.*", ", THEN sets the thermostat temperature [ACTION_Y]."),
    (r".*set.*target temperature.*thermostat.*", ", THEN sets the thermostat target temperature [ACTION_Y]."),
    (r".*set.*room temperature.*", ", THEN sets the room temperature [ACTION_Z]."),
    (r".*set.*setpoint.*", ", THEN sets the device temperature setpoint [ACTION_Z]."),
    (r".*set.*specific temperature.*current mode.*", ", THEN sets the thermostat temperature in [ACTION_X]]."),
    (r".*set.*manual.*specific temperature.*|.*set.*manual mode.*", ", THEN sets the thermostat to [ACTION_X]."),
    (r".*set.*temperature.*trv.*", ", THEN sets the temperature [ACTION_Z]."),
    (r".*activate.*(eco|fresh) mode.*", ", THEN activates [ACTION_X] on the appliance."),
    (r".*activate.*vacation mode.*|.*holiday mode.*", ", THEN activates vacation mode [ACTION_Z]."),
    (r".*activate.*standby mode.*", ", THEN activates [ACTION_X]."),
    (r".*cancel.*standby mode.*", ", THEN cancels [ACTION_X]."), # cancels eco mode è non eco
    (r".*set.*heating mode.*off.*", ", THEN turns heating mode off [ACTION_W]."),
    (r".*set.*heating mode.*schedule.*", ", THEN sets heating mode to [ACTION_X]."),
    (r".*set.*thermostat mode.*", ", THEN sets the thermostat mode to [ACTION_X]."),
    (r".*set.*cool mode.*", ", THEN sets cooling mode [ACTION_Z]."),
    (r".*set.*(heat|heating) mode.*", ", THEN sets heating mode [ACTION_Z]."),
    (r".*set.*frost.*guard.*|.*turn off.*frost protect.*", ", THEN enables [ACTION_X]."),
    (r".*set.*away mode.*", ", THEN sets home to [ACTION_X]."),
    (r".*boost.*heating.*", ", THEN boosts heating [ACTION_Y]."),
    (r".*boost.*hot water.*", ", THEN boosts hot water [ACTION_Y]."),
    (r".*start.*heating.*preferred temperature.*", ", THEN starts heating at a preferred temperature [ACTION_Y]."),
    (r".*start.*heating.*", ", THEN starts heating [ACTION_Z]."),
    (r".*enable.*temperature hold.*", ", THEN enables a temperature hold [ACTION_Y]."),
    (r".*set.*thermostat.*hold.*comfort profile.*", ", THEN sets a timed [ACTION_X] to hold."),
    (r".*set.*hold.*specified temperature.*hours.*", ", THEN sets a temperature hold [ACTION_Y]."),
    (r".*resume.*programmed schedule.*", ", THEN resumes the programmed schedule [ACTION_Y]."),
    (r".*return.*default schedule.*", ", THEN restores the default schedule [ACTION_Z]."),
    (r".*change.*schedule.*", ", THEN updates the thermostat schedule [ACTION_W]."),
    (r".*create.*vacation event.*", ", THEN creates an [ACTION_X] temperature schedule."),
    (r".*increase.*room temperature.*", ", THEN increases room temperature [ACTION_Z]."),
    (r".*decrease.*room temperature.*", ", THEN decreases room temperature [ACTION_W]."),
    (r".*set.*fan.*on.*|.*fan.*auto.*", ", THEN configures thermostat fan mode [ACTION_Z]."),
    (r".*turn on.*fan.*nest.*", ", THEN activates the HVAC fan [ACTION_Y]."),
    (r".*turn on.*air conditioner.*", ", THEN turns on the air conditioner and configures settings to [ACTION_X]."),
    (r".*turn your a/c on.*", ", THEN turns on the air conditioner [ACTION_Z]."),
    (r".*turn your aros.*a/c.*", ", THEN turns on the air conditioner in [ACTION_X]."),
    (r".*set.*thermostat.*heat-cool mode.*", ", THEN sets thermostat to [ACTION_X]."),
    (r".*temporarily cool.*freezer.*", ", THEN temporarily boosts freezer cooling [ACTION_Y]."),
    (r".*temporarily cool.*refrigerator.*", ", THEN temporarily boosts refrigerator cooling [ACTION_Y]."),
    (r".*turn your heating off.*", ", THEN turns off heating [ACTION_W]."),
    (r".*turn off.*thermostat.*", ", THEN turns off the thermostat [ACTION_W]."),
    (r".*turn thermostat back on.*", ", THEN restores the previous thermostat mode [ACTION_Z]."),
    (r".*switch.*electric switch off.*", ", THEN switches off the electric switch [ACTION_W]."),
    (r".*return control.*tado.*", ", THEN change temperature control in [ACTION_X]."),
    (r".*order.*back.*default schedule.*", ", THEN restores default heating schedule [ACTION_Z]."),
    (r".*remove.*running event.*thermostat.*", ", THEN cancels the current thermostat event [ACTION_W]."),
    (r".*set.*thermostat group mode.*", ", THEN sets thermostat group mode to [ACTION_X]."),
    (r".*set.*home.*away mode.*time.*", ", THEN sets home to temporary [ACTION_X]."),
    (r".*set the temperature.*|.*setpoint.*|.*temperature you specify.*", ", THEN the thermostat to the temperature you specify [ACTION_W]."),
    (r".*turn.*off.*", ", THEN turn off the thermostat [ACTION_W].")
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
        r"dust concentration|Volatile Organic|tag senses that|or moisture monitoring",
        case=False,
        na=False
    )
]

filtered_df = filtered_df[
    ~filtered_df["actionDesc"].str.contains(
        r".*back on.*|.*position.*|.*trv.*|.*indefinite.*|.*permanent.*",
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
