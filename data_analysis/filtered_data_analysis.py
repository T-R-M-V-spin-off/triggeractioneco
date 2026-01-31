import pandas as pd
import re
from pathlib import Path

# ================= CONFIG =================

INPUT_CSV = "../dataset/filtered_data/action_channel_filtered_dataset.csv"
OUTPUT_REPORT = "trigger_action_analysis_report.txt"

TRIGGER_COL = "triggerDesc"
ACTION_COL = "actionDesc"

OUTPUT_DIR = "../dataset/filtered_data_analysis"

Path(OUTPUT_DIR).mkdir(exist_ok=True)

# ================= LOAD =================

df = pd.read_csv(INPUT_CSV)
total_rows = len(df)

df[TRIGGER_COL] = df[TRIGGER_COL].fillna("").astype(str)
df[ACTION_COL] = df[ACTION_COL].fillna("").astype(str)

# lower per regex
df["_trigger_l"] = df[TRIGGER_COL].str.lower()
df["_action_l"] = df[ACTION_COL].str.lower()

# ================= REGEX =================

ACTION_START_REGEX = re.compile(r"^this action (will|well)")
TRIGGER_START_REGEX = re.compile(r"^this trigger fires")
NUMBER_REGEX = re.compile(r"\d")

TIME_WORDS = [
    "always", "every time", "every", "at ",
    "minute", "minutes", "hour", "hours", "day", "days", " current "
]

def contains_time_words(text):
    return any(word in text for word in TIME_WORDS)

# ================= TRIGGER COLUMNS =================

df["trigger_starts_with_ok"] = df["_trigger_l"].str.match(TRIGGER_START_REGEX)
df["trigger_contains_numbers"] = df["_trigger_l"].str.contains(NUMBER_REGEX)
df["trigger_contains_timewords"] = df["_trigger_l"].apply(contains_time_words)

# ================= ACTION COLUMNS =================

df["action_starts_with_ok"] = df["_action_l"].str.match(ACTION_START_REGEX)
df["action_contains_numbers"] = df["_action_l"].str.contains(NUMBER_REGEX)
df["action_contains_timewords"] = df["_action_l"].apply(contains_time_words)

# ================= SAVE CSV =================

# Trigger CSV
trigger_cols = [TRIGGER_COL, "trigger_starts_with_ok",
                "trigger_contains_numbers", "trigger_contains_timewords"]
df[trigger_cols].to_csv(f"{OUTPUT_DIR}/trigger_checked.csv", index=False)

# Action CSV
action_cols = [ACTION_COL, "action_starts_with_ok",
               "action_contains_numbers", "action_contains_timewords"]
df[action_cols].to_csv(f"{OUTPUT_DIR}/action_checked.csv", index=False)

# ================= REPORT =================

def pct(x):
    return (x / total_rows) * 100

print("\n========== TRIGGER ==========")
print(f"Starts with 'This Trigger fires': {df['trigger_starts_with_ok'].sum()} ({pct(df['trigger_starts_with_ok'].sum()):.2f}%)")
print(f"Contains numbers: {df['trigger_contains_numbers'].sum()} ({pct(df['trigger_contains_numbers'].sum()):.2f}%)")
print(f"Contains time words: {df['trigger_contains_timewords'].sum()} ({pct(df['trigger_contains_timewords'].sum()):.2f}%)")

print("\n========== ACTION ==========")
print(f"Starts with 'This Action will/well': {df['action_starts_with_ok'].sum()} ({pct(df['action_starts_with_ok'].sum()):.2f}%)")
print(f"Contains numbers: {df['action_contains_numbers'].sum()} ({pct(df['action_contains_numbers'].sum()):.2f}%)")
print(f"Contains time words: {df['action_contains_timewords'].sum()} ({pct(df['action_contains_timewords'].sum()):.2f}%)")

print("\nCSV files saved in:", OUTPUT_DIR)

# ================= FILTER TRIGGERS THAT FAIL =================

FAILED_TRIGGER_OUTPUT = f"{OUTPUT_DIR}/trigger_starts_with_false.csv"

failed_triggers_df = df[df["trigger_starts_with_ok"] == False][
    ["id", TRIGGER_COL, "deviceCategory"]
]

failed_triggers_df.to_csv(FAILED_TRIGGER_OUTPUT, index=False)

print(f"\nFailed trigger CSV saved in: {FAILED_TRIGGER_OUTPUT}")
print(f"Total failed triggers: {len(failed_triggers_df)}")