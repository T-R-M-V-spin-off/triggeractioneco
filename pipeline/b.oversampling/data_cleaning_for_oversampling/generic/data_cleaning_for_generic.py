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

    # ================= AIR / ENVIRONMENT / POLLUTION =================

    (r".*(air pollution|air quality).*above.*|.*pm2\.5.*|.*particulate matter.*", 
     "IF air pollution exceeds a threshold"),

    (r".*(carbon dioxide|co2).*above.*", 
     "IF carbon dioxide level exceeds a threshold"),

    (r".*(voc|volatile organic).*above.*|.*below.*", 
     "IF volatile organic compounds cross a threshold"),

    (r".*uv index.*above.*", 
     "IF UV index exceeds a threshold"),

    (r".*(radon).*above.*|.*below.*", 
     "IF radon level crosses a threshold"),

    (r".*(wind speed|gust speed).*above.*", 
     "IF wind speed exceeds a threshold"),

    (r".*(humidity).*above.*|.*below.*|.*too dry.*|.*normal humidity.*", 
     "IF humidity crosses a threshold"),

    # ================= TEMPERATURE =================

    (r".*(temperature).*above.*|.*rises above.*|.*greater than.*", 
     "IF temperature rises above a threshold"),

    (r".*(temperature).*below.*|.*drops below.*|.*less than.*", 
     "IF temperature drops below a threshold"),

    (r".*(ambient temperature).*above.*|.*below.*", 
     "IF ambient temperature crosses a threshold"),

    # ================= ENERGY / PRICE =================

    (r".*(electricity price|price).*above.*|.*below.*|.*average price.*", 
     "IF electricity price changes beyond a threshold"),

    (r".*(dynamic pricing|hourly pricing|real time market).*", 
     "IF dynamic energy pricing changes"),

    (r".*(carbon intensity|carbon clean|carbon dirty).*", 
     "IF carbon intensity changes"),

    (r".*(ohmhour).*start.*|.*ohmhour.*end.*", 
     "IF a demand response event starts or ends"),

    # ================= POWER / CONSUMPTION =================

    (r".*(power|watts).*above.*|.*consumption.*", 
     "IF power consumption exceeds a threshold"),

    (r".*(exporting solar).*", 
     "IF solar power is exported to the grid"),

    # ================= BATTERY / CHARGING =================

    (r".*(battery).*below.*|.*drops below.*", 
     "IF device battery drops below a threshold"),

    (r".*(plugged in|unplugged).*", 
     "IF device is plugged in or unplugged"),

    # ================= CONNECTIVITY =================

    (r".*(connects|disconnects).*wifi.*|.*google wifi.*", 
     "IF device connects or disconnects from WiFi"),

    (r".*(disconnects).*bluetooth.*", 
     "IF device disconnects from Bluetooth"),

    # ================= CALLS / SMS / VOICEMAIL =================

    (r".*(receive|send).*sms.*", 
     "IF an SMS is sent or received"),

    (r".*(incoming call|answer call|miss call|phone call).*", 
     "IF a phone call event occurs"),

    (r".*(voicemail).*", 
     "IF a voicemail is received"),

    # ================= CALENDAR / MEETINGS =================

    (r".*(calendar).*start.*|.*meeting starts.*", 
     "IF a calendar event starts"),

    (r".*(calendar).*end.*|.*meeting ends.*", 
     "IF a calendar event ends"),

    (r".*(minutes before).*calendar.*|.*before.*event.*", 
     "IF shortly before a calendar event"),

    (r".*(within 15 minutes).*calendar.*", 
     "IF shortly after a calendar event"),

    (r".*(new event).*calendar.*", 
     "IF a new calendar event is added"),

    # ================= PRESENCE / FAMILY =================

    (r".*(everyone left|family member left).*", 
     "IF everyone leaves home"),

    (r".*(family member has come home|someone comes home).*", 
     "IF someone arrives home"),

    (r".*(presence).*detected.*|.*no longer detected.*", 
     "IF user presence changes"),

    # ================= DOORS / LOCKS / CONTACT =================

    (r".*(door|contact sensor).*open.*|.*opened.*", 
     "IF a door or contact sensor opens"),

    (r".*(door|contact sensor).*close.*|.*closed.*", 
     "IF a door or contact sensor closes"),

    (r".*(lock).*locked.*|.*unlocked.*", 
     "IF a lock is locked or unlocked"),

    # ================= MOTION =================

    (r".*(motion).*detect.*|.*movement.*|.*no movement.*", 
     "IF motion is detected or stops"),

    # ================= DEVICES ON / OFF =================

    (r".*(device|switch|zone|product).*turn.*on.*|.*switched on.*", 
     "IF a device is turned on"),

    (r".*(device|switch|zone|product).*turn.*off.*|.*switched off.*", 
     "IF a device is turned off"),

    (r".*(on or off).*", 
     "IF a device changes state"),

    # ================= BUTTONS / WEARABLE =================

    (r".*(press|double press|long press).*button.*|.*logi.*|.*flic.*|.*wink.*", 
     "IF a button is pressed"),

    (r".*(watch|kronaby|festina|jaguar|lotus).*pusher.*", 
     "IF a smartwatch button is pressed"),

    # ================= ROBOTS / AUTOMATION =================

    (r".*(robot).*start.*job.*|.*completes job.*", 
     "IF a robot starts or completes a task"),

    (r".*(routine|flow|automation).*activated.*|.*runs.*", 
     "IF an automation routine runs"),

    # ================= SOCIAL / MEDIA =================

    (r".*(tweet|twitter).*hashtag.*|.*post.*tweet.*", 
     "IF a tweet is posted"),

    (r".*@mentioned.*tweet.*", 
     "IF mentioned on Twitter"),

    (r".*(youtube|video).*new.*", 
     "IF a new video is published"),

    (r".*(super chat|membership).*", 
     "IF a channel interaction occurs"),

    (r".*(facebook).*status.*|.*photo.*tagged.*", 
     "IF a Facebook post or tag occurs"),

    # ================= REMINDERS / TODO =================

    (r".*(to do|todo|reminder).*add.*|.*complete.*|.*delete.*", 
     "IF a reminder or task changes"),

    # ================= SECURITY / ALARMS =================

    (r".*(alarm).*triggered.*|.*alarm event.*|.*armed.*|.*disarmed.*", 
     "IF a security alarm changes state"),

    (r".*(smoke|carbon monoxide).*detect.*", 
     "IF dangerous gas or smoke is detected"),

    (r".*(leak|water).*detect.*", 
     "IF a water leak is detected"),

    # ================= WEATHER FORECAST =================

    (r".*(forecast|tomorrow).*weather.*", 
     "IF weather forecast changes"),

    # ================= SENSORS / GENERIC =================

    (r".*(sensor).*activated.*|.*input signal.*", 
     "IF a sensor is activated"),

    (r".*(specific value).*sensor.*", 
     "IF a sensor reaches a target value"),

    # ================= SCHEDULED =================

    (r".*(once a year|every month).*", 
     "IF a scheduled periodic event occurs"),

    (r".*(timer goes off).*", 
     "IF a timer expires"),

# ================= AIR POLLUTION / CARBON =================

    (r".*(global air pollution|air pollution).*higher than.*|.*carbon.*clean.*|.*carbon.*dirty.*",
     "IF air pollution or carbon intensity crosses a threshold"),

    # ================= SPREADSHEET =================

    (r".*(spreadsheet).*cell.*(updated|changed|modified).*",
     "IF a spreadsheet cell is updated"),

    # ================= WEBHOOK / EXTERNAL EVENT =================

    (r".*(ifttt receives|receives).*event.*(apilio|webhook|external).*",
     "IF an external webhook event is received"),

    # ================= MOTION / SECURITY CAM =================

    (r".*(senses motion|detects motion|motion sensor).*",
     "IF motion is detected"),

    # ================= ENERGY PRICE ADVANCED =================

    (r".*(electricity price).*(falls|rises|threshold|time).*",
     "IF electricity price changes within a time window"),

    # ================= RELAY / TOGGLE =================

    (r".*(relay).*turned on.*toggle.*|.*toggle mode.*relay.*",
     "IF a relay is toggled on"),

    # ================= PRESENCE FULL HOUSE =================

    (r".*(every family member has left|everyone left).*",
     "IF everyone leaves home"),

    # ================= VOICE / KEYWORD =================

    (r".*(says a particular word|keyword|phrase).*group.*|.*voice command.*",
     "IF a specific keyword is spoken"),

    # ================= SLEEP TRACKING =================

    (r".*(sleep).*get (out of|into) bed.*|.*logs new sleep.*",
     "IF sleep activity is detected"),

    # ================= HOME AUTOMATION FLOWS =================

    (r".*(flow).*started.*homey.*|.*automation.*started.*",
     "IF an automation flow is started"),

    # ================= TARIFF / PEAK / PRICING ZONE =================

    (r".*(plage tarifaire|peak rates|tariff|time-of-day).*start.*|.*end.*",
     "IF a tariff or peak rate period starts or ends"),

    # ================= SMOKE / GAS =================

    (r".*(smoke|carbon monoxide|co).*detects.*(dangerous|warning).*",
     "IF dangerous smoke or gas is detected"),

    # ================= LIGHT / SWITCH =================

    (r".*(light|dimmer|switch).*switch(es)? (on|off)|.*turned (on|off).*",
     "IF a light or switch changes state"),

    # ================= AUDIO DETECTION =================

    (r".*audio event.*detected.*|.*sound.*detected.*",
     "IF an audio event is detected"),

    # ================= SMARTWATCH BUTTON =================

    (r".*(pusher).*watch.*|.*watch.*button.*",
     "IF a smartwatch button is pressed"),

    # ================= MODE SWITCH =================

    (r".*(switches to|enters|changes).*mode.*(home|away|heat|security).*",
     "IF a device changes operating mode"),

    # ================= PRAYER / RELIGIOUS TIME =================

    (r".*(time for prayer).*|.*prayer time.*",
     "IF prayer time occurs"),

    # ================= TEMPERATURE SENSOR TAG =================

    (r".*(ambient temperature).*exceeds.*threshold.*|.*returns between.*thresholds.*",
     "IF temperature crosses configured thresholds"),

    # ================= TELEGRAM / MESSAGING BOT =================

    (r".*(telegram).*key phrase.*|.*@ifttt.*bot.*",
     "IF a keyword is sent via messaging bot"),

    # ================= FITNESS / GOALS =================

    (r".*(daily step goal).*achieve.*|.*do not achieve.*goal.*",
     "IF a fitness goal is achieved or missed"),

    # ================= VEHICLE / GEO =================

    (r".*(vehicle).*exit.*geo.*|.*geofence.*exit.*",
     "IF a vehicle exits a geofence"),

    # ================= SECURITY PANEL =================

    (r".*(security panel).*receives.*alarm.*|.*alarm activated.*",
     "IF a security panel alarm occurs"),

    # ================= GROUP DECISION =================

    (r".*(group decision|direct action).*taken.*",
     "IF a group decision is made"),

    # ================= CHECK-IN / LOCATION =================

    (r".*(check in).*foursquare.*|.*swarm.*|.*venue.*category.*|.*hashtag.*",
     "IF a location check-in occurs"),

    # ================= RSS / FEED =================

    (r".*(new item).*feed.*|.*rss.*new.*",
     "IF a new feed item is published"),

    # ================= SMART DEVICE POWER =================

    (r".*(powered on|power on).*device.*|.*enters standby.*",
     "IF a device powers on or enters standby"),

    # ================= DEVICE ISSUES =================

    (r".*(low battery|device offline|issue occurs).*",
     "IF a device problem occurs"),

    # ================= YOUTUBE / VIDEO =================

    (r".*(new video).*public.*|.*subscribed.*user.*video.*",
     "IF a subscribed user publishes a video"),

    # ================= TODO / NOTES =================

    (r".*(add).*to do list.*|.*new task.*",
     "IF a new task is added"),

    # ================= SOCIAL STATUS =================

    (r".*(facebook).*status.*message.*|.*plain text.*",
     "IF a social status is posted"),

    # ================= DOOR / LOCK =================

    (r".*(door).*open.*|.*close.*|.*locked.*",
     "IF a door or lock changes state"),

    # ================= SCREENSHOT =================

    (r".*(screenshot).*taken.*android.*",
     "IF a screenshot is taken"),

    # ================= ALEXA =================

    (r".*(alexa).*shopping list.*ask.*",
     "IF Alexa shopping list is queried"),

    # ================= ALARM STATE =================

    (r".*(during alarms).*|.*alarm event.*",
     "IF an alarm is active"),

    # ================= SPREADSHEET (SPECIFIC CELL) =================

    (r".*(particular|specific).*cell.*(updated|changed|modified).*spreadsheet.*",
     "IF a specific spreadsheet cell is updated"),

    (r".*spreadsheet.*cell.*(updated|changed).*",
     "IF a spreadsheet cell is updated"),

    # ================= VOICE KEYWORD / ORION =================

    (r".*(orion).*group.*(says|say).*word.*phrase.*",
     "IF a specific keyword is spoken in a voice group"),

    (r".*(say|says).*particular.*(word|phrase).*",
     "IF a specific spoken keyword is detected"),

    # ================= FRENCH TARIFF =================

    (r".*(plage tarifaire).*commence.*logement.*",
     "IF a tariff period starts (French)"),

    (r".*(plage tarifaire).*commence.*",
     "IF a tariff period starts"),

    # ================= NEST PROTECT (SMOKE / CO) =================

    (r".*nest protect.*detects.*(dangerous|warning).*smoke.*",
     "IF Nest Protect detects smoke"),

    (r".*nest protect.*detects.*(dangerous|warning).*carbon monoxide.*",
     "IF Nest Protect detects carbon monoxide"),

    # ================= TADO / HOME MODE =================

    (r".*(tado).*switches to.*away mode.*",
     "IF thermostat switches to away mode"),

    (r".*(tado).*switches to.*home mode.*",
     "IF thermostat switches to home mode"),

    (r".*(security device).*enters.*home mode.*",
     "IF security system enters home mode"),

    # ================= FITNESS GOAL =================

    (r".*achieve.*daily step goal.*",
     "IF daily step goal is achieved"),

    # ================= MIHOME POWER =================

    (r".*(mi\|home|mihome).*powered on.*",
     "IF a MiHome device is powered on"),

    # ================= MYFOX ALARM / RISK =================

    (r".*(myfox).*receives.*alarm.*",
     "IF Myfox security alarm occurs"),

    (r".*(myfox).*detects.*risk.*",
     "IF Myfox detects a household risk"),

    # ================= HEAT MODE =================

    (r".*device.*set to.*heat mode.*",
     "IF a device switches to heat mode"),

    # ================= RF BRIDGE ALARM =================

    (r".*(rf bridge).*alarm.*activated.*",
     "IF RF bridge alarm is activated"),

    # ================= SWANN MOTION =================

    (r".*(swann).*detects.*motion.*",
     "IF Swann camera detects motion"),

    # ================= SITE MODE =================

    (r".*site.*changes.*mode.*",
     "IF a site changes operating mode"),

    # ================= ALEXA SHOPPING =================

    (r".*(alexa).*what'?s on.*shopping list.*",
     "IF Alexa shopping list is queried"),

    (r".*(alexa).*shopping list.*ask.*",
     "IF Alexa shopping list is queried"),

    # ================= DOOR SENSOR OPEN =================

    (r".*(opens the door|door.*open).*activated door sensor.*|.*someone opens the door.*",
     "IF a door is opened"),

    (r".*alarm.*bridge.*","IF an alarm of your network bridge is activated")
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
        r"NaN",
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
