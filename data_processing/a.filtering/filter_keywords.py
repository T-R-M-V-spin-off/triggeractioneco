import pandas as pd
import re
from collections import Counter

# ================= CONFIGURATION =================

INPUT_CSV = "../../dataset/raw_dataset.csv"
OUTPUT_FILTERED = "../../dataset/filtered_data/keywords_filtered_dataset.csv"
OUTPUT_REPORT = "filter_keywords_report.txt"

SEARCH_KEYWORDS = [
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
    "ac",
    "a/c",
    "air conditioner",
    "cool",
    "oven",
    "coffee"
]

COLUMNS_TO_CHECK = [
    "title",
    "desc",
    "triggerDesc",
    "actionDesc"
]

# ================= DEVICE CATEGORIES =================

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


# ================= BUILD REGEX =================

keyword_pattern = re.compile(
    r"\b(?:" + "|".join(map(re.escape, SEARCH_KEYWORDS)) + r")\b",
    re.IGNORECASE
)


# ================= FILTER FUNCTION =================

def row_matches_keywords(row):
    """
    Check if any target column contains
    at least one keyword.
    """

    for col in COLUMNS_TO_CHECK:
        if col in row and pd.notna(row[col]):
            if keyword_pattern.search(str(row[col])):
                return True

    return False


# ================= APPLY FILTER =================

filtered_df = df[df.apply(row_matches_keywords, axis=1)].copy()

# ================= DEVICE CLASSIFICATION =================

filtered_df["deviceCategory"] = filtered_df.apply(
    lambda row: classify_device(
        str(row.get("actionChannelTitle", "")) + " " +
        str(row.get("actionDesc", ""))
    ),
    axis=1
)


filtered_count = len(filtered_df)

# ================= KEYWORD STATISTICS =================

keyword_counter = Counter()

def count_keywords(text):

    text = str(text)

    matches = set(
        m.lower()
        for m in keyword_pattern.findall(text)
    )

    for m in matches:
        keyword_counter[m] += 1


for _, row in filtered_df.iterrows():

    combined_text = ""

    for col in COLUMNS_TO_CHECK:
        if col in row and pd.notna(row[col]):
            combined_text += " " + str(row[col])

    count_keywords(combined_text)


# ================= COLUMN COVERAGE =================

column_hits = Counter()

for col in COLUMNS_TO_CHECK:

    matches = filtered_df[col].dropna().astype(str).str.contains(
        keyword_pattern,
        regex=True
    )

    column_hits[col] = matches.sum()


# ================= DEVICE STATISTICS =================

device_counter = Counter(filtered_df["deviceCategory"])


# ================= REPORT GENERATION =================

coverage = (filtered_count / total_entries) * 100

with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:

    f.write("SMART HOME DATASET - KEYWORD FILTER REPORT\n")
    f.write("=" * 45 + "\n\n")

    # General stats
    f.write("GENERAL STATISTICS\n")
    f.write("-" * 25 + "\n")
    f.write(f"Total dataset entries: {total_entries}\n")
    f.write(f"Filtered entries: {filtered_count}\n")
    f.write(f"Coverage: {coverage:.2f}%\n\n")

    # Keyword stats
    f.write("KEYWORD STATISTICS\n")
    f.write("-" * 25 + "\n")

    for kw, count in keyword_counter.most_common():
        percentage = (count / filtered_count) * 100

        f.write(
            f"  {kw}: {count} "
            f"({percentage:.2f}% of rows)\n"
        )

    f.write("\n")

    # Column stats
    f.write("COLUMN MATCH DISTRIBUTION\n")
    f.write("-" * 30 + "\n")

    for col, count in column_hits.items():
        percentage = (count / filtered_count) * 100

        f.write(
            f"  {col}: {count} "
            f"({percentage:.2f}%)\n"
        )

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

    f.write("\n")


    f.write("END OF REPORT\n")

filtered_df = filtered_df.drop(columns=["isDoRecipe",
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

# Save filtered dataset
filtered_df.to_csv(OUTPUT_FILTERED, index=False)

# ================= CONSOLE OUTPUT =================

print(f"Total entries: {total_entries}")
print(f"Filtered entries: {filtered_count}")
print(f"Coverage: {coverage:.2f}%")
print(f"Report saved to: {OUTPUT_REPORT}")
