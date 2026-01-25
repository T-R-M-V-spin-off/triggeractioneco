import pandas as pd
from collections import Counter

# ================= CONFIGURATION =================

KEYWORD_FILE = "../dataset/keywords_filtered_dataset.csv"
ACTION_FILE = "../dataset/action_channel_filtered_dataset.csv"
RAW_FILE = "../dataset/raw_dataset.csv"

OUTPUT_COMMON = "../dataset/common_filtered_dataset.csv"
OUTPUT_REPORT = "../dataset/common_report.txt"

ID_COLUMN = "id"
ACTION_COLUMN = "actionChannelTitle"
TRIGGER_COLUMN = "triggerChannelTitle"
DESC_COLUMN = "actionDesc"

# Columns to remove in final dataset
COLUMNS_TO_DROP = [
    "isDoRecipe",
    "favoritesCount",
    "addCount",
    "creatorName",
    "creatorUrl",
    "url",
    "created"
]

# Device categories (same logic as previous scripts)
DEVICE_CATEGORIES = {
    "Light": [
        "light", "lamp", "hue", "lifx", "bulb", "lighting", "nanoleaf"
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

# ================= LOAD DATA =================

df_keywords = pd.read_csv(KEYWORD_FILE)
df_action = pd.read_csv(ACTION_FILE)
df_raw = pd.read_csv(RAW_FILE)

total_raw = len(df_raw)
total_keywords = len(df_keywords)
total_action = len(df_action)

# ================= COMPUTE INTERSECTION =================

keyword_ids = set(df_keywords[ID_COLUMN])
action_ids = set(df_action[ID_COLUMN])

common_ids = keyword_ids.intersection(action_ids)

common_df = df_raw[df_raw[ID_COLUMN].isin(common_ids)]

common_count = len(common_df)

# ================= CLEAN COLUMNS =================

common_df = common_df.drop(
    columns=COLUMNS_TO_DROP,
    errors="ignore"
)

# Save final dataset
common_df.to_csv(OUTPUT_COMMON, index=False)

# ================= DEVICE CLASSIFICATION =================

def classify_device(text):

    text = str(text).lower()

    for category, keywords in DEVICE_CATEGORIES.items():
        for kw in keywords:
            if kw in text:
                return category

    return "Generic/Other"


device_types = []

for _, row in common_df.iterrows():

    combined_text = (
        str(row.get(ACTION_COLUMN, "")) + " " +
        str(row.get(DESC_COLUMN, ""))
    )

    device_types.append(classify_device(combined_text))


device_counter = Counter(device_types)

# ================= CHANNEL STATISTICS =================

action_channels = (
    common_df[ACTION_COLUMN]
    .dropna()
    .astype(str)
)

channel_counter = Counter(action_channels)

unique_action_channels = len(channel_counter)

trigger_channels = (
    common_df[TRIGGER_COLUMN]
    .dropna()
    .astype(str)
)

trigger_counter = Counter(trigger_channels)

unique_trigger_channels = len(trigger_counter)

# ================= COVERAGE METRICS =================

keyword_coverage = (common_count / total_keywords) * 100
action_coverage = (common_count / total_action) * 100
dataset_coverage = (common_count / total_raw) * 100

# ================= REPORT GENERATION =================

with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:

    f.write("SMART HOME DATASET - INTERSECTION REPORT\n")
    f.write("=" * 50 + "\n\n")

    # General
    f.write("GENERAL STATISTICS\n")
    f.write("-" * 25 + "\n")

    f.write(f"Raw dataset entries: {total_raw}\n")
    f.write(f"Keyword filtered entries: {total_keywords}\n")
    f.write(f"Action filtered entries: {total_action}\n")
    f.write(f"Intersection entries: {common_count}\n\n")

    # Coverage
    f.write("FILTER COVERAGE\n")
    f.write("-" * 25 + "\n")

    f.write(f"Coverage of keyword filter: {keyword_coverage:.2f}%\n")
    f.write(f"Coverage of action filter: {action_coverage:.2f}%\n")
    f.write(f"Coverage of full pipeline: {dataset_coverage:.2f}%\n\n")

    # Channel stats
    f.write("CHANNEL STATISTICS\n")
    f.write("-" * 25 + "\n")

    f.write(f"Unique action channels: {unique_action_channels}\n")
    f.write(f"Unique trigger channels: {unique_trigger_channels}\n\n")

    f.write("Top action channels:\n")

    for channel, count in channel_counter.most_common(15):
        f.write(f"  {channel}: {count}\n")

    f.write("\n")

    f.write("Top trigger channels:\n")

    for channel, count in trigger_counter.most_common(15):
        f.write(f"  {channel}: {count}\n")

    f.write("\n")

    # Device stats
    f.write("DEVICE TYPE DISTRIBUTION\n")
    f.write("-" * 30 + "\n")

    for device, count in device_counter.most_common():

        percentage = (count / common_count) * 100

        f.write(
            f"  {device}: {count} "
            f"({percentage:.2f}%)\n"
        )

    f.write("\n")

    # Loss analysis
    f.write("DATA LOSS ANALYSIS\n")
    f.write("-" * 30 + "\n")

    lost_after_keywords = total_keywords - common_count
    lost_after_action = total_action - common_count

    f.write(f"Entries lost after intersection (from keyword set): {lost_after_keywords}\n")
    f.write(f"Entries lost after intersection (from action set): {lost_after_action}\n\n")

    f.write("END OF REPORT\n")


# ================= CONSOLE OUTPUT =================

print("Intersection completed")
print(f"Raw entries: {total_raw}")
print(f"Keyword entries: {total_keywords}")
print(f"Action entries: {total_action}")
print(f"Common entries: {common_count}")
print(f"Final dataset coverage: {dataset_coverage:.2f}%")
print(f"Report saved to: {OUTPUT_REPORT}")
