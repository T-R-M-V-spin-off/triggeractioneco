import pandas as pd
import re

# ========= CONFIGURAZIONE =========

INPUT_CSV = "dataset/raw_dataset.csv"
OUTPUT_FILTRATO = "keywords_filtered_dataset.csv"
OUTPUT_ACTION_TRIGGER_X = "action_channel_filtered_dataset.csv"
COLONNA_ID = "id"


PAROLE_DA_CERCARE = [
    "light",
    "lights",
    "temperature",
    "turn-on",
    "turn-off",
    "pollution",
    "turn on",
    "turn off",
    "smart",
    "heat",
    "heater",
    "open",
    "close",
]

COLONNE_DA_CONTROLLARE = [
    "title",
    "desc",
    "triggerDesc",
    "actionDesc"
]

# ========= ACTION TRIGGER CON X =========

ACTION_TRIGGER_X = [
    "WeMo Smart Plug",
    "LIFX",
    "Philips Hue",
    "Lutron Caséta and RA2 Select",
    "Aura",
    "iRobot",
    "SmartThings",
    "WiZ",
    "Nest Thermostat",
    "Gogogate",
    "Heatmiser",
    "Smart Life",
    "Homey",
    "Ambi Climate",
    "SwitchBot",
    "WeMo Maker",
    "SmartLiving Home Automation",
    "TP-Link Kasa",
    "Wink: Shortcuts",
    "WeMo Light Switch",
    "GE Appliances Window AC",
    "WeMo Insight Switch",
    "Philips HearLink",
    "Yeelight",
    "ecobee",
    "BeoLiving Intelligence",
    "Nature Remo",
    "LightwaveRF Lighting",
    "Hive Active Heating™ - UK and Europe",
    "Moodo & Moodo AIR",
    "Apilio",
    "Hive Active Light™",
    "AduroSmart",
    "Heatzy",
    "LightwaveRF Power",
    "My Leviton",
    "Wemo Dimmer",
    "Nanoleaf Smarter Series",
    "Optoma",
    "AC Cloud Control",
    "WeMo Lighting",
    "Smappee",
    "Energenie Mi|Home",
    "Home Connect Dishwasher",
    "MagicLight WiFi",
    "Sensibo",
    "Connexoon Asia & Oceania",
    "ThermoSmart",
    "BDR Thermea Group",
    "Honeywell Single-zone Thermostat",
    "Home Connect Oven",
    "nVent Nuheat Signature Thermostat",
    "Netatmo Thermostat",
    "AL-KO Smart Garden",
    "Daikin Online Controller",
    "CloudRain Smart Garden Irrigation",
    "Honeywell Total Connect Comfort",
    "Home Connect Washer",
    "Brilliant Smart",
    "MagicHue",
    "Home Connect Hood",
    "RainMachine",
    "Home Connect Coffee Machine",
    "openHAB",
    "LIVISI Home",
    "Wink: Pivot Power Genius",
    "iDevices",
    "GE Appliances GeoSpring™",
    "YoLink",
    "Honeywell evohome",
    "Aquanta",
    "Awair",
    "Fanimation",
    "BG Home",
    "tadoº Heating",
    "Hager IoT",
    "Wink: Aros",
    "AQUAREA Smart Cloud",
    "LightwaveRF Heating",
    "Home Connect Fridge Freezer",
    "Hubitat",
    "Wattio SmartHome",
    "Levoit Air Purifier",
    "iHome Control",
    "GE Appliances Refrigerator",
    "Blink (Europe)",
    "Mysa Thermostat",
    "Warmup Smart Thermostat",
    "VeSync Dimmer",
    "Samsung Washer",
    "Blue by ADT",
    "Uplink Remote",
    "Sengled",
    "Nefit Easy",
    "Anyware Services",
    "Futurehome",
    "MSmartHome Dehumidifier",
    "iotty Smart Home",
    "Epion",
    "Aqara Home for US",
    "GE Appliances Dryer",
    "WeMo Humidifier",
    "WeMo Slow Cooker",
    "Wemo Air Purifier"
]

# ========= LETTURA CSV =========

df = pd.read_csv(INPUT_CSV)

# ========= FILTRAGGIO GENERICO =========

pattern_generico = re.compile(
    "|".join(map(re.escape, PAROLE_DA_CERCARE)),
    re.IGNORECASE
)

def riga_match_generico(row):
    for col in COLONNE_DA_CONTROLLARE:
        if col in row and pd.notna(row[col]):
            if pattern_generico.search(str(row[col])):
                return True
    return False

df_filtrato = df[df.apply(riga_match_generico, axis=1)]
df_filtrato.to_csv(OUTPUT_FILTRATO, index=False)

# ========= FILTRAGGIO actionTrigger (SOLO X) =========

pattern_action = re.compile(
    "|".join(map(re.escape, ACTION_TRIGGER_X)),
    re.IGNORECASE
)

df_action_x = df[
    df["actionChannelTitle"].notna() &
    df["actionChannelTitle"].astype(str).str.contains(pattern_action)
]

df_action_x.to_csv(OUTPUT_ACTION_TRIGGER_X, index=False)

# ========= OUTPUT =========

print(f"CSV filtrato generico: {len(df_filtrato)} righe")
print(f"CSV actionTrigger X: {len(df_action_x)} righe")

# ========= INTERSEZIONE TRA I DUE DATASET =========

ids_filtrato = set(df_filtrato[COLONNA_ID])
ids_action_x = set(df_action_x[COLONNA_ID])

ids_comuni = ids_filtrato.intersection(ids_action_x)

df_comune = df[df[COLONNA_ID].isin(ids_comuni)]

df_comune = df_comune.drop(columns=["isDoRecipe",
                      "title",
                      "desc",
                      "triggerChannelId",
                      "triggerChannelUrl",
                      "triggerId",
                      "actionChannelId",
                      "actionChannelUrl",
                      "actionId",
                      "favoritesCount",
                      "addCount",
                      "creatorName",
                      "creatorUrl",
                      "url",
                      "created"])

df_comune.to_csv("output_comune.csv", index=False)

print(f"Entry in comune tra i due dataset: {len(ids_comuni)}")

# Funzione per creare la regola formattata
def format_rule(row):
    # Trasformiamo tutto in minuscolo prima di qualsiasi operazione
    trigger = str(row["triggerTitle"]).lower().strip()
    action = str(row["actionDesc"]).lower().replace("this action will", "then").strip()
    action_short = action.split(".", 1)[0].strip()
    rule_text = f"if {trigger}, {action_short}"
    return rule_text

# Applichiamo la funzione
df_comune["rule"] = df_comune.apply(format_rule, axis=1)

# Mantieni la colonna ID e la colonna rule
df_rules = df_comune[[COLONNA_ID, "rule"]]

# Salva in CSV
df_rules.to_csv("output_rules.csv", index=False)
