import pandas as pd
import re
from collections import Counter

# ================= CONFIGURATION =================

INPUT_CSV = "../dataset/raw_dataset.csv"
OUTPUT_ACTION_FILTERED = "../dataset/action_channel_filtered_dataset.csv"
OUTPUT_REPORT = "../dataset/action_report.txt"

ACTION_CHANNEL_TITLE_COLUMN = "actionChannelTitle"
ACTION_DESC_COLUMN = "actionDesc"

# Whitelisted action channels
ACTION_CHANNEL_WHITELIST = [
    "WeMo Smart Plug",
    "LIFX",
    "Philips Hue",
    "Lutron Caséta and RA2 Select",
    "iRobot",
    "SmartThings",
    "WiZ",
    "Nest Thermostat",
    "Gogogate",
    "Heatmiser",
    "Smart Life",
    "Ambi Climate",
    "SwitchBot",
    "WeMo Maker",
    "TP-Link Kasa",
    "WeMo Light Switch",
    "GE Appliances Window AC",
    "WeMo Insight Switch",
    "Yeelight",
    "ecobee",
    "Nature Remo",
    "LightwaveRF Lighting",
    "Hive Active Heating™ - UK and Europe",
    "Hive Active Light™",
    "Heatzy",
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
    "ThermoSmart",
    "BDR Thermea Group",
    "Home Connect Oven",
    "nVent Nuheat Signature Thermostat",
    "Netatmo Thermostat",
    "Daikin Online Controller",
    "Honeywell Total Connect Comfort",
    "Brilliant Smart",
    "MagicHue",
    "Home Connect Hood",
    "Home Connect Coffee Machine",
    "Wink: Pivot Power Genius",
    "iDevices",
    "GE Appliances GeoSpring™",
    "tadoº Heating",
    "Wink: Aros",
    "AQUAREA Smart Cloud",
    "LightwaveRF Heating",
    "Home Connect Fridge Freezer",
    "Hubitat",
    "Levoit Air Purifier",
    "Mysa Thermostat",
    "Warmup Smart Thermostat",
    "VeSync Dimmer",
    "Blue by ADT",
    "Sengled",
    "Nefit Easy",
    "Aura"
]

# Device classification keywords
DEVICE_CATEGORIES = {
    "Light": [
        "light", "lamp", "hue", "lifx", "bulb", "lighting", "nanoleaf", "aura", "lights"
    ],
    "Thermostat/Heating": [
        "thermostat", "heat", "heating", "temperature", "warm", "cool"
    ],
    "Appliance": [
        "fridge", "dishwasher", "oven", "washer", "dryer", "coffee", "hood"
    ],
    "Air/Climate": [
        "air", "ac", "climate", "purifier", "humidifier", "dehumidifier"
    ]
}

# ================= FUNCTIONS =================


def classify_device(text: str) -> str:

    text = str(text).lower()

    for category, keywords in DEVICE_CATEGORIES.items():
        for kw in keywords:

            pattern = r"\b" + re.escape(kw) + r"\b"

            if re.search(pattern, text):
                return category

    return "Generic/Other"


# ================= LOAD DATA =================

df = pd.read_csv(INPUT_CSV)

total_entries = len(df)

# ================= UNIQUE CHANNELS =================

all_unique_channels = (
    df[ACTION_CHANNEL_TITLE_COLUMN]
    .dropna()
    .astype(str)
    .unique()
)

total_unique_channels = len(all_unique_channels)

# ================= FILTER =================

action_pattern = re.compile(
    "|".join(map(re.escape, ACTION_CHANNEL_WHITELIST)),
    re.IGNORECASE
)

filtered_df = df[
    df[ACTION_CHANNEL_TITLE_COLUMN].notna() &
    df[ACTION_CHANNEL_TITLE_COLUMN].astype(str).str.contains(action_pattern)
    ]

filtered_count = len(filtered_df)

# Remove useless columns
COLUMNS_TO_DROP = [
    "isDoRecipe",
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
    "created"
]

filtered_df = filtered_df.drop(
    columns=COLUMNS_TO_DROP,
    errors="ignore"
)

# ================= DEVICE CLASSIFICATION =================

filtered_df["deviceCategory"] = filtered_df.apply(
    lambda row: classify_device(
        str(row.get(ACTION_CHANNEL_TITLE_COLUMN, "")) + " " +
        str(row.get(ACTION_DESC_COLUMN, ""))
    ),
    axis=1
)

# ================= SAVE DATASET =================

filtered_df.to_csv(OUTPUT_ACTION_FILTERED, index=False)

# ================= CHANNEL STATISTICS =================

channel_counts = (
    filtered_df[ACTION_CHANNEL_TITLE_COLUMN]
    .value_counts()
    .to_dict()
)

unique_channels_found = len(channel_counts)
total_whitelisted_channels = len(ACTION_CHANNEL_WHITELIST)

matched_channels = set()

for channel in all_unique_channels:
    if action_pattern.search(channel):
        matched_channels.add(channel)

covered_unique_channels = len(matched_channels)

channel_coverage = (
    covered_unique_channels / total_unique_channels
) * 100

# ================= DEVICE STATISTICS =================

device_counter = Counter(filtered_df["deviceCategory"])

# ================= REPORT =================

dataset_coverage = (filtered_count / total_entries) * 100

with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:

    f.write("SMART HOME DATASET - ACTION CHANNEL REPORT\n")
    f.write("=" * 45 + "\n\n")

    # General stats
    f.write("GENERAL STATISTICS\n")
    f.write("-" * 25 + "\n")

    f.write(f"Total dataset entries: {total_entries}\n")
    f.write(f"Filtered entries: {filtered_count}\n")
    f.write(f"Dataset coverage: {dataset_coverage:.2f}%\n\n")

    # Channel coverage
    f.write("CHANNEL COVERAGE\n")
    f.write("-" * 25 + "\n")

    f.write(f"Total unique channels in dataset: {total_unique_channels}\n")
    f.write(f"Unique channels covered by whitelist: {covered_unique_channels}\n")
    f.write(f"Channel coverage: {channel_coverage:.2f}%\n\n")

    f.write(f"Whitelist size: {total_whitelisted_channels}\n")
    f.write(f"Whitelist channels found: {unique_channels_found}\n\n")

    # Channel stats
    f.write("ENTRIES PER CHANNEL\n")
    f.write("-" * 25 + "\n")

    for channel, count in sorted(
        channel_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        f.write(f"  {channel}: {count}\n")

    f.write("\n")

    # Device stats
    f.write("DEVICE TYPE DISTRIBUTION\n")
    f.write("-" * 30 + "\n")

    for device, count in device_counter.most_common():

        percentage = (count / filtered_count) * 100

        f.write(
            f"  {device}: {count} "
            f"({percentage:.2f}%)\n"
        )

    f.write("\nEND OF REPORT\n")


# ================= CONSOLE OUTPUT =================

print("=== DATASET PROCESSING COMPLETED ===\n")

print(f"Total entries: {total_entries}")
print(f"Filtered entries: {filtered_count}")
print(f"Dataset coverage: {dataset_coverage:.2f}%")

print(
    f"Channel coverage: "
    f"{covered_unique_channels}/{total_unique_channels} "
    f"({channel_coverage:.2f}%)"
)

print(f"\nFiltered dataset saved to: {OUTPUT_ACTION_FILTERED}")
print(f"Report saved to: {OUTPUT_REPORT}")
