import pandas as pd
import re

# Configuration
RAW_DATASET = "dataset/raw_dataset.csv"
ACTION_CHANNEL_FILTERED_DATASET = "dataset/action_channel_filtered_dataset.csv"

# This array contains action channel values to check for
ACTION_CHANNELS_TO_CHECK = [
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

df = pd.read_csv(RAW_DATASET)

action_pattern = re.compile(
    "|".join(map(re.escape, ACTION_CHANNELS_TO_CHECK)),
    re.IGNORECASE
)

df_filtered = df[
    df["actionChannelTitle"].notna() &
    df["actionChannelTitle"].astype(str).str.contains(action_pattern)
]

df_filtered.to_csv(ACTION_CHANNEL_FILTERED_DATASET, index=False)

print(f"ACTION_CHANNEL_FILTERED_DATASET - {len(df_filtered)} rows")