import pandas as pd
from collections import Counter

# ================= CONFIGURATION =================

KEYWORD_FILE = "../../dataset/filtered_data/keywords_filtered_dataset.csv"
ACTION_FILE = "../../dataset/filtered_data/action_channel_filtered_dataset.csv"
RAW_FILE = "../../dataset/raw_dataset.csv"

OUTPUT_COMMON = "../../dataset/filtered_data/common_filtered_dataset.csv"
OUTPUT_REPORT = "filter_in_common_report.txt"

ID_COLUMN = "id"
ACTION_COLUMN = "actionChannelTitle"
TRIGGER_COLUMN = "triggerChannelTitle"
DESC_COLUMN = "actionDesc"

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
    "created",
    "deviceCategory_kw",
    "deviceCategory_action"
]

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

# Intersezione dal raw dataset
common_df = df_raw[df_raw[ID_COLUMN].isin(common_ids)]
common_count = len(common_df)

# ================= MERGE DEVICE CATEGORY =================

# Mantieni tutte le colonne dei due dataset filtrati
df_keywords_subset = df_keywords[[ID_COLUMN, "deviceCategory"]].rename(columns={"deviceCategory": "deviceCategory_kw"})
df_action_subset = df_action[[ID_COLUMN, "deviceCategory"]].rename(columns={"deviceCategory": "deviceCategory_action"})

# Merge per avere entrambe le colonne deviceCategory
common_df = common_df.merge(df_keywords_subset, on=ID_COLUMN, how="left")
common_df = common_df.merge(df_action_subset, on=ID_COLUMN, how="left")

# Unifica deviceCategory
if "deviceCategory_kw" in common_df.columns and "deviceCategory_action" in common_df.columns:
    common_df["deviceCategory"] = common_df["deviceCategory_action"]
elif "deviceCategory_action" in common_df.columns:
    common_df["deviceCategory"] = common_df["deviceCategory_action"]
elif "deviceCategory_kw" in common_df.columns:
    common_df["deviceCategory"] = common_df["deviceCategory_kw"]

# ================= CLEAN COLUMNS =================

common_df = common_df.drop(columns=COLUMNS_TO_DROP, errors="ignore")

# ================= SAVE FINAL DATASET =================

common_df.to_csv(OUTPUT_COMMON, index=False)

# ================= CHANNEL STATISTICS =================

action_channels = common_df[ACTION_COLUMN].dropna().astype(str)
channel_counter = Counter(action_channels)
unique_action_channels = len(channel_counter)

trigger_channels = common_df[TRIGGER_COLUMN].dropna().astype(str)
trigger_counter = Counter(trigger_channels)
unique_trigger_channels = len(trigger_counter)

# ================= DEVICE STATISTICS =================

device_counter = Counter(common_df["deviceCategory"])

# ================= COVERAGE METRICS =================

keyword_coverage = (common_count / total_keywords) * 100
action_coverage = (common_count / total_action) * 100
dataset_coverage = (common_count / total_raw) * 100

# ================= REPORT GENERATION =================

with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:

    f.write("SMART HOME DATASET - INTERSECTION REPORT\n")
    f.write("=" * 50 + "\n\n")

    # General stats
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
    f.write("\nTop trigger channels:\n")
    for channel, count in trigger_counter.most_common(15):
        f.write(f"  {channel}: {count}\n")

    # Device stats
    f.write("\nDEVICE TYPE DISTRIBUTION\n")
    f.write("-" * 30 + "\n")
    for device, count in device_counter.most_common():
        percentage = (count / common_count) * 100
        f.write(f"  {device}: {count} ({percentage:.2f}%)\n")

    # Loss analysis
    f.write("\nDATA LOSS ANALYSIS\n")
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
print(f"Filtered dataset saved to: {OUTPUT_COMMON}")
