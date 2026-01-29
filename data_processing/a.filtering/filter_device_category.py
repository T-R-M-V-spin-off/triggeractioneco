import pandas as pd
from collections import Counter
import os

# ================= CONFIG =================

INPUT_CSV = "../../dataset/filtered_data/action_channel_filtered_dataset.csv"
OUTPUT_FOLDER = "../../dataset/filtered_data/device_category_data"
OUTPUT_REPORT = "filter_device_category_reports"

DEVICE_CATEGORY_COL = "deviceCategory"
TRIGGER_COL = "triggerChannelTitle"
ACTION_COL = "actionChannelTitle"

TOP_N = 10  # Top N valori più frequenti da mostrare nel report

# ================= CREATE FOLDERS =================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_REPORT, exist_ok=True)

# ================= LOAD DATA =================

df = pd.read_csv(INPUT_CSV)

device_categories = df[DEVICE_CATEGORY_COL].dropna().unique()

# ================= PROCESS EACH CATEGORY =================

for category in device_categories:
    subset = df[df[DEVICE_CATEGORY_COL] == category].copy()

    # ================= SAVE CSV =================
    category_csv = os.path.join(OUTPUT_FOLDER, f"{category.replace('/', '_')}.csv")
    subset.to_csv(category_csv, index=False)

    # ================= GENERATE REPORT =================
    report_lines = [
        f"=== REPORT for Device Category: {category} ===",
        f"Total rows: {len(subset)}",
        ""
    ]

    # Trigger distribution
    trigger_counts = Counter(subset[TRIGGER_COL].dropna().astype(str))
    report_lines.append(f"Unique trigger channels: {len(trigger_counts)}")
    report_lines.append("Top trigger channels:")
    for trigger, count in trigger_counts.most_common(TOP_N):
        percent = count / len(subset) * 100
        report_lines.append(f"  {trigger}: {count} ({percent:.2f}%)")
    report_lines.append("")

    # Action distribution
    action_counts = Counter(subset[ACTION_COL].dropna().astype(str))
    report_lines.append(f"Unique action channels: {len(action_counts)}")
    report_lines.append("Top action channels:")
    for action, count in action_counts.most_common(TOP_N):
        percent = count / len(subset) * 100
        report_lines.append(f"  {action}: {count} ({percent:.2f}%)")

    # ================= SAVE REPORT =================
    report_file = os.path.join(OUTPUT_REPORT, f"{category.replace('/', '_')}_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Processed category '{category}': {len(subset)} rows saved to {category_csv}, report saved to {report_file}")
