import pandas as pd
import re

# Configuration
RAW_DATASET = "dataset/raw_dataset.csv"
KEYWORD_FILTERED_DATASET = "dataset/keywords_filtered_dataset.csv"

KEYWORDS = [
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

# This array contains all columns in which search a keyword
COLUMNS_TO_CHECK = [
    "title",
    "desc",
    "triggerDesc",
    "actionDesc"
]

df = pd.read_csv(RAW_DATASET)

generic_pattern = re.compile(
    "|".join(map(re.escape, KEYWORDS)),
    re.IGNORECASE
)

def match(row):
    for col in COLUMNS_TO_CHECK:
        if col in row and pd.notna(row[col]):
            if generic_pattern.search(str(row[col])):
                return True
    return False

df_filtered = df[df.apply(match, axis=1)]
df_filtered.to_csv(KEYWORD_FILTERED_DATASET, index=False)

print(f"KEYWORD_FILTERED_DATASET - {len(df_filtered)} rows")