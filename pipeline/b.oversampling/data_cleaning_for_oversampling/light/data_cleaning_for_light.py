import pandas as pd
import os
import re


# ===============================
# CONFIG
# ===============================

INPUT_FILE = "../../../../dataset/filtered_data/device_category_data/Light.csv"


# ===============================
# NORMALIZATION RULES
# ===============================
TRIGGER_RULES = [
    (r".*pir detects person.*", "IF PIR detects person"),
    (r".*watering event.*executed.*", "IF watering executed"),
    (r".*receive your phone call.*woop.la.*", "IF phone call received"),
    (r".*given parameter.*exceed.*threshold.*", "IF parameter exceeds threshold"),
    (r".*new record.*ifttt event.*sobject.*", "IF Salesforce record created"),
    (r".*killed in rust.*not playing.*", "IF killed in Rust offline"),
    (r".*slickdeals.*new deal.*", "IF Slickdeals deal posted"),
    (r".*breaking news alert.*", "IF breaking news"),
    (r".*bart.*delayed.*station.*", "IF BART delay"),
    (r".*new incident.*assigned.*current user.*", "IF incident assigned"),
    (r".*water heater.*exceeds.*energy.*threshold.*", "IF water heater energy high"),
    (r".*caavo control center.*search query.*", "IF Caavo search performed"),
    (r".*single press.*misfit.*", "IF misfit single press"),
    (r".*double press.*misfit.*", "IF misfit double press"),
    (r".*triple press.*misfit.*", "IF misfit triple press"),
    (r".*long press.*logi.*", "IF logi long press"),
    (r".*short press.*logi.*", "IF logi short press"),
    (r".*press.*wink relay.*", "IF wink button pressed"),
    (r".*press.*lametric.*", "IF lametric button pressed"),
    (r".*wemo.*detects.*motion.*", "IF wemo motion detected"),
    (r".*withings.*detects.*motion.*", "IF withings motion detected"),
    (r".*netatmo.*detects.*person.*", "IF netatmo person detected"),
    (r".*netatmo.*detects.*vehicle.*", "IF netatmo vehicle detected"),
    (r".*simcam.*detects.*person.*", "IF simcam person detected"),
    (r".*simcam.*face id.*", "IF simcam face recognized"),
    (r".*aura.*detects.*motion.*", "IF aura motion detected"),
    (r".*wyze.*detects.*sound.*", "IF wyze sound detected"),
    (r".*swann.*detects.*sound.*", "IF swann sound detected"),
    (r".*noise rises.*decibel.*", "IF noise threshold exceeded"),
    (r".*door.*starts to open.*", "IF garage door opening"),
    (r".*door.*left open.*", "IF door left open"),
    (r".*door.*locked.*", "IF door locked"),
    (r".*contact sensor.*opens.*", "IF contact opens"),
    (r".*contact sensor.*closes.*", "IF contact closes"),
    (r".*enters away mode.*", "IF away mode entered"),
    (r".*exits away mode.*", "IF away mode exited"),
    (r".*switches to away mode.*", "IF away mode switched"),
    (r".*night mode.*on.*", "IF night mode on"),
    (r".*night mode.*off.*", "IF night mode off"),
    (r".*new track.*playlist.*", "IF new playlist track"),
    (r".*save.*track.*spotify.*", "IF spotify save track"),
    (r".*amazon prime music.*played.*", "IF amazon song played"),
    (r".*stream is going live.*", "IF stream live"),
    (r".*super chat.*", "IF superchat received"),
    (r".*super sticker.*", "IF super sticker received"),
    (r".*washer.*cycle is complete.*", "IF washer done"),
    (r".*dryer.*finishes.*", "IF dryer done"),
    (r".*dishwasher.*finishes.*", "IF dishwasher done"),
    (r".*oven.*finished.*", "IF oven finished"),
    (r".*ikettle.*finished boiling.*", "IF kettle boiled"),
    (r".*monitors changes.*weather condition.*(rain|snow|cloudy|clear).*", "IF weather condition change"),
    (r".*within 15 minutes of the sunset.*", "IF sunset approaching"),
    (r".*within 15 minutes of the sunrise.*", "IF sunrise approaching"),
    (r".*tomorrow.?s forecasted weather.*", "IF tomorrow weather change"),
    (r".*retrieves today.?s current weather.*", "IF today weather report"),
    (r".*enter an area you specify.*", "IF area entered"),
    (r".*exit an area you specify.*", "IF area exited"),
    (r".*vehicle exits.*geo.*circle.*", "IF vehicle geofence exit"),
    (r".*at least one family member has come home.*", "IF family arrived"),
    (r".*every family member has left.*", "IF family left"),
    (r".*motion event is detected.*", "IF motion detected"),
    (r".*motion is detected.*doorbell.*", "IF doorbell motion"),
    (r".*camera detects.*motion.*", "IF camera motion"),
    (r".*detects.*sound.*", "IF sound detected"),
    (r".*motion is no longer detected.*", "IF motion cleared"),
    (r".*lock is unlocked by.*person.*", "IF lock unlocked by user"),
    (r".*lock is locked by.*person.*", "IF lock locked by user"),
    (r".*lock is unlocked.*", "IF lock unlocked"),
    (r".*lock is locked.*", "IF lock locked"),
    (r".*door is opened.*", "IF door opened"),
    (r".*door is closed.*", "IF door closed"),
    (r".*alarm goes off.*", "IF alarm triggered"),
    (r".*alexa trigger.*", "IF alexa custom phrase"),
    (r".*ok google.*phrase.*", "IF google assistant phrase"),
    (r".*ok google.*post.*tweet.*", "IF google post tweet"),
    (r".*ok google.*set nest.*", "IF google set thermostat"),
    (r".*carbon dioxide.*rises above.*", "IF co2 high"),
    (r".*carbon dioxide.*drops below.*", "IF co2 low"),
    (r".*air quality.*unhealthy.*", "IF air quality unhealthy"),
    (r".*electricity price changes.*", "IF electricity price change"),
    (r".*renewable generation.*changes.*", "IF renewable energy change"),
    (r".*connects to.*wifi.*", "IF wifi connected"),
    (r".*disconnects from.*wifi.*", "IF wifi disconnected"),
    (r".*miss a phone call.*android.*", "IF missed call"),
    (r".*receive.*new sms.*", "IF sms received"),
    (r".*dryer cycle is complete.*", "IF dryer finished"),
    (r".*oven turns off.*timer.*", "IF oven finished"),
    (r".*robot finishes cleaning.*", "IF robot cleaning done"),
    (r".*coffee.*finished brewing.*", "IF coffee ready"),
    (r".*post.*new tweet.*hashtag.*", "IF tweet with hashtag"),
    (r".*new follower.*", "IF new follower"),
    (r".*upload.*new.*youtube.*video.*", "IF youtube upload"),
    (r".*share.*new photo.*instagram.*", "IF instagram post"),
    (r".*specific days of the week.*time.*", "IF weekly schedule"),
    (r".*once a year.*", "IF yearly schedule"),
    (r".*every single day.*specific time.*", "IF daily schedule"),
    (r".*once an hour.*:00.*:15.*:30.*:45.*", "IF quarter hour"),
    (r".*send ifttt any email.*trigger@applet.ifttt.com.*", "IF ifttt email trigger"),
    (r".*email.*hashtag.*#ifttt.*", "IF ifttt hashtag email"),
    (r".*receives.*event from apilio.*", "IF apilio event"),
    (r".*daily step goal.*", "IF step goal reached"),
    (r".*daily distance goal.*", "IF distance goal reached"),
    (r".*log.*weight.*fitbit.*", "IF weight logged"),
    (r".*sleep detects.*out of bed.*", "IF out of bed"),
    (r".*home.*set to away.*", "IF home away mode"),
    (r".*home.*set to home.*", "IF home home mode"),
    (r".*away mode is on.*", "IF away mode on"),
    (r".*away mode is canceled.*", "IF away mode canceled"),
    (r".*energy saving mode scene.*", "IF energy saving mode"),
    (r".*press the button.*", "IF generic button press"),
    (r".*click.*flic.*", "IF flic clicked"),
    (r".*ihome enhance button.*", "IF ihome button"),
    (r".*skybell.*button pressed.*", "IF skybell pressed"),
    (r".*audio event is detected.*", "IF an audio event is detected"),
    (r".*sound is detected.*microphone.*", "IF sound is detected"),
    (r".*sound is detected.*", "IF sound is detected"),
    (r".*says a particular word.*phrase.*", "IF a voice keyword is detected"),
    (r".*detects temperatures above.*", "IF temperature exceeds a threshold"),
    (r".*temperature drops below.*", "IF temperature falls below a threshold"),
    (r".*temperature rises above.*", "IF temperature rises above a threshold"),
    (r".*local temperature.*drops below.*", "IF local temperature drops"),
    (r".*local temperature.*rises above.*", "IF local temperature rises"),
    (r".*outdoor temperature.*rises above.*", "IF outdoor temperature rises"),
    (r".*room temperature.*above.*", "IF room temperature is high"),
    (r".*room temperature.*below.*", "IF room temperature is low"),
    (r".*unlocks the door.*auto unlock.*", "IF door is unlocked"),
    (r".*door.*is open.*garage mode.*", "IF garage door is open"),
    (r".*door.*panel.*opened.*", "IF door panel is opened"),
    (r".*wemo light switch.*turned on.*", "IF WeMo light is turned on"),
    (r".*wemo light switch.*turned off.*", "IF WeMo light is turned off"),
    (r".*wemo insight switch.*turned on.*", "IF WeMo Insight is turned on"),
    (r".*wemo insight switch.*turned off.*", "IF WeMo Insight is turned off"),
    (r".*wemo insight switch.*standby.*", "IF WeMo enters standby"),
    (r".*eweling.*switch.*turned on.*off.*", "IF eWeLink switch changes"),
    (r".*motion is detected.*", "IF motion is detected"),
    (r".*motion sensor detects motion.*", "IF motion sensor is triggered"),
    (r".*arlo device detects motion.*", "IF Arlo detects motion"),
    (r".*wyze cam detects motion.*", "IF Wyze detects motion"),
    (r".*person is detected.*camera.*", "IF a person is detected"),
    (r".*human is detected.*", "IF human is detected"),
    (r".*new video public.*", "IF new public video is posted"),
    (r".*upload.*public video.*youtube.*", "IF YouTube video is uploaded"),
    (r".*like a video on youtube.*", "IF a YouTube video is liked"),
    (r".*post a new tweet.*", "IF a tweet is posted"),
    (r".*are @mentioned.*tweet.*", "IF mentioned on Twitter"),
    (r".*new photo on facebook.*", "IF Facebook photo is posted"),
    (r".*plain text status.*facebook.*", "IF Facebook status is posted"),
    (r".*incoming call.*contact list.*", "IF a known caller calls"),
    (r".*incoming call.*not in.*contact.*", "IF an unknown caller calls"),
    (r".*receive an incoming call.*", "IF a call is received"),
    (r".*send an sms.*android.*", "IF an SMS is sent"),
    (r".*receive an sms.*android.*", "IF an SMS is received"),
    (r".*new notification.*android.*", "IF Android notification is received"),
    (r".*minutes before.*google calendar.*", "IF calendar event is upcoming"),
    (r".*within 15 minutes.*ending time.*calendar.*", "IF calendar event is ending"),
    (r".*scheduled meeting starts.*", "IF meeting starts"),
    (r".*scheduled meeting ends.*", "IF meeting ends"),
    (r".*alarm is armed.*", "IF alarm is armed"),
    (r".*alarm is disarmed.*", "IF alarm is disarmed"),
    (r".*alarm is triggered.*", "IF alarm is triggered"),
    (r".*alarm sounds.*", "IF alarm sounds"),
    (r".*receives an alarm.*security.*", "IF security alarm occurs"),
    (r".*air quality.*below.*", "IF air quality is low"),
    (r".*air quality.*alert.*", "IF air quality alert occurs"),
    (r".*co2.*level.*above.*", "IF CO2 is high"),
    (r".*radon level.*above.*", "IF radon is high"),
    (r".*pollen count.*above.*", "IF pollen is high"),
    (r".*battery drops below.*", "IF battery is low"),
    (r".*low battery.*detected.*", "IF low battery is detected"),
    (r".*power rises above.*watts.*", "IF power usage is high"),
    (r".*order is out for delivery.*", "IF order is out for delivery"),
    (r".*order is being prepped.*", "IF order is being prepared"),
    (r".*order is ready for pickup.*", "IF order is ready"),
    (r".*order is in the oven.*", "IF order is cooking"),
    (r".*time for prayer.*", "IF prayer time occurs"),
    (r".*time for fast.*", "IF fasting time occurs"),
    (r".*robot starts a job.*", "IF robot starts working"),
    (r".*robot starts cleaning.*", "IF robot starts cleaning"),
    (r".*laundry cycle ends.*", "IF laundry is finished"),
    (r".*dryer cycle.*end.*", "IF dryer finishes"),
    (r".*markets close.*price rises.*", "IF stock price rises"),
    (r".*markets close.*price drops.*", "IF stock price drops"),
    (r".*client device connects.*router.*", "IF device connects to router"),
    (r".*tp-link router.*connects.*", "IF device connects to TP-Link"),
    (r".*leak is detected.*", "IF water leak is detected"),
    (r".*smoke is detected.*", "IF smoke is detected"),
    (r".*oven door.*opened.*", "IF oven door is opened"),
    (r".*refrigerator.*door.*open.*", "IF fridge door is open"),
    (r".*carbon intensity changes.*", "IF carbon intensity change"),
    (r".*co2.*above.*", "IF co2 high"),
    (r".*co2.*below.*", "IF co2 low"),
    (r".*pollution.*higher.*", "IF pollution high"),
    (r".*ohmhour.*starts.*", "IF ohmhour starts"),
    (r".*device.*connects.*asus.*", "IF asus device connect"),
    (r".*device.*disconnects.*tp-link.*", "IF tplink disconnect"),
    (r".*d-link.*connects.*", "IF dlink connect"),
    (r".*child.*money added.*", "IF child money added"),
    (r".*child.*money removed.*", "IF child money removed"),
    (r".*issue is closed.*", "IF github issue closed"),
    (r".*github.*notification.*", "IF github notification"),
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
    (r".*change.*brightness.*wemo.*group.*|.*change.*brightness.*wemo.*light.*",", THEN sets the brightness level of smart lights [ACTION_Z]."),
    (r".*sleep fader.*wemo.*group.*|.*sleep fader.*wemo.*light.*", ", THEN start smart lights [ACTION_Z]."),
    (r".*turns off.*light switch.*|.*turn off.*light switch.*", ", THEN turns off a light switch [ACTION_Y]."),
    (r".*turns on.*light switch.*|.*turn on.*light switch.*", ", THEN turns on a light switch in [ACTION_X]."),
    (r".*set.*scene.*hue.*|.*activate.*scene.*|.*nanoleaf.*scene.*|.*wiz.*scene.*|.*lutron.*scene.*", ", THEN activates a lighting scene [ACTION_Y]."),
    (r".*briefly.*hue.*off.*on.*", ", THEN briefly toggles the lights off and on [ACTION_Z]."),
    (r".*change.*brightness.*bulb.*|.*set.*brightness.*light.*|.*dim.*brighten.*", ", THEN adjusts the brightness of lights [ACTION_W]."),
    (r".*change.*color temperature.*|.*specified color.*|.*bulb.*", ", THEN sets the color temperature of lights [ACTION_Y]."),
    (r".*change.*color.*nanoleaf.*|.*change.*color.*hue.*|.*change.*color.*lights.*|.*set.*color.*", ", THEN changes the color of lights [ACTION_Y]."),
    (r".*random.*color.*|.*match.*dominant colors.*", ", THEN sets lights to a random or image-based color [ACTION_Y]."),
    (r".*dynamic mode.*|.*flashing.*|.*jumping.*|.*strobe.*", ", THEN enables [ACTION_X] as lighting mode."),
    (r".*pulse.*gently.*|.*blink.*|.*breathe.*", ", THEN triggers a light animation effect [ACTION_Z]."),
    (r".*pre-defined scene.*", ", THEN sets lights to a predefined scene [ACTION_Z]."),
    (r".*switch off.*light.*", ", THEN switches off the lights [ACTION_W]."),
    (r".*switch on.*light.*", ", THEN switches on the lights [ACTION_Z]."),
    (r".*temporarily.*nanoleaf.*solid.*blinking.*", ", THEN changes light color [ACTION_Y]."),
    (r".*toggle.*wemo.*|.*toggle.*hive.*|.*toggle.*hue.*|.*toggle.*lights.*", ", THEN toggles the lights on or off [ACTION_Y]."),
    (r".*turn.*wemo.*off.*on.*|.*turn.*wemo.*on.*off.*", ", THEN power-cycles a smart switch in [ACTION_X]."),
    (r".*turn off.*functional light.*hood.*|.*turn off.*ambient light.*hood.*", ", THEN turns off the hood lighting [ACTION_W]."),
    (r".*turn on.*ambient light.*hood.*|.*turn on.*functional light.*hood.*", ", THEN turns on the hood lighting [ACTION_Z]."),
    (r".*turn off.*nanoleaf.*|.*turn off.*hue.*|.*turn off.*sengled.*|.*turn off.*selected light.*", ", THEN turns off smart lights [ACTION_W]."),
    (r".*turn on.*nanoleaf.*|.*turn on.*hue.*|.*turn on.*sengled.*|.*turn on.*selected light.*", ", THEN turns on smart lights in [ACTION_X]."),
    (r".*color loop.*hue.*", ", THEN starts a color loop lighting effect [ACTION_Y]."),
    (r".*on/off.*led.*projector.*", ", THEN toggles the projector LED light [ACTION_Z]."),
    (r".*night light.*off.*|.*night light.*on.*|.*hive.*off.*duration.*", ", THEN turns off the light [ACTION_W]."),
    (r".*hive.*on.*duration.*", ", THEN turns on the light [ACTION_Y]."),
    (r".*turn your lights off if.*on.*|.*turn your lights on if.*off.*", ", THEN toggles lights based on current state in [ACTION_X]."),
    (r".*turn your lights off.*|.*turns off.*wemo.*group.*|.*turns off.*wemo.*light.*", ", THEN turns off smart lights [ACTION_W]."),
    (r".*turn your lights on.*|.*turns on.*wemo.*group.*|.*turns on.*wemo.*light.*", ", THEN turns on smart lights [ACTION_X]."),
    (r".*adjust.*multiple.*lights.*shades.*activating.*scene.*lutron.*|.*lutron.*app.*scene.*", ", THEN activates a lighting scene in [ACTION_X]."),
    (r".*turn.*wemo.*light switch.*on.*remain on.*|.*turn.*wemo.*switch.*on.*", ", THEN turns on a light switch [ACTION_Y]."),
    (r".*turn.*wemo.*light switch.*off.*remain off.*|.*turn.*wemo.*switch.*off.*", ", THEN turns off a light switch [ACTION_W]."),
    (r".*set.*(your )?lights?.*specified brightness.*|.*set.*dim level.*light.*|"
     r".*set.*light.*specified brightness.*|.*change.*light.*brightness.*specified level.*|.*change.*lights?.*brightness.*level.*",
     ", THEN adjusts the brightness of lights [ACTION_Z]."),
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
        r".*space.*|.*astronaut.*|.*applet every day.*|.*ce trigger.*|.*health organization.*|"
        r".*tecan instrument.*|.*frontpage at slickdeals.*|.*swann.*",
        case=False,
        na=False
    )
]

filtered_df = filtered_df[
    ~filtered_df["actionDesc"].str.contains(
        r".*wiz lights.*",
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
    f"{base}_normalized_final.csv"
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


# Remove useless columns
COLUMNS_TO_DROP = [
    "id",
    "triggerChannelTitle",
    "triggerTitle",
    "actionChannelTitle",
    "actionTitle",
    "deviceCategory",
    "originalTrigger",
    "originalAction"
]

df = df.drop(
    columns=COLUMNS_TO_DROP,
    errors="ignore"
)

df["freeOrLocked"] = "LOCKED"

df.to_csv(output_file, index=False)

print(f"Saved: {output_file}")

print("==========================================")
