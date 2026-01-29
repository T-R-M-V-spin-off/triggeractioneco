import pandas as pd
from pathlib import Path
import math

# ================= CONFIG =================

INPUT_CSV = "../../dataset/rules/balanced_seed_dataset.csv"
OUTPUT_CSV = "../../dataset/rules/oversampled_seed_dataset.csv"
REPORT_FILE = "oversampling_report.txt"

DEVICE_COL = "deviceCategory"
TARGET_PER_DEVICE = 150
RANDOM_SEED = 42

# ================= LOAD =================

df = pd.read_csv(INPUT_CSV)
initial_rows = len(df)

# ================= DEVICE DISTRIBUTION BEFORE =================

device_before = (
    df[DEVICE_COL]
    .value_counts()
    .rename_axis(DEVICE_COL)
    .reset_index(name="count")
)

# ================= OVERSAMPLING =================

oversampled_parts = [df]

oversampling_stats = []

for _, row in device_before.iterrows():
    device = row[DEVICE_COL]
    count = row["count"]

    if count >= TARGET_PER_DEVICE:
        oversampling_stats.append(
            (device, count, count, 0)
        )
        continue

    df_device = df[df[DEVICE_COL] == device]

    needed = TARGET_PER_DEVICE - count
    repeats = math.ceil(needed / count)

    replicated = pd.concat(
        [df_device] * repeats,
        ignore_index=True
    ).sample(
        n=needed,
        random_state=RANDOM_SEED
    )

    replicated["_oversampled"] = True
    df_device["_oversampled"] = False

    oversampled_parts.append(replicated)

    oversampling_stats.append(
        (device, count, count + needed, needed)
    )

# Merge everything
df_final = pd.concat(oversampled_parts, ignore_index=True)
df_final["_oversampled"] = df_final["_oversampled"].fillna(False)

final_rows = len(df_final)

# ================= DEVICE DISTRIBUTION AFTER =================

device_after = (
    df_final[DEVICE_COL]
    .value_counts()
    .rename_axis(DEVICE_COL)
    .reset_index(name="count")
)

# ================= SAVE DATASET =================

Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
df_final.to_csv(OUTPUT_CSV, index=False)

# ================= REPORT =================

lines = []

lines.append("=== DEVICE OVERSAMPLING REPORT ===")
lines.append(f"Initial rows: {initial_rows}")
lines.append(f"Final rows: {final_rows}")
lines.append(f"Rows added: {final_rows - initial_rows}")
lines.append(f"Target per deviceCategory: {TARGET_PER_DEVICE}")

lines.append("\n=== DEVICE CATEGORY OVERSAMPLING DETAILS ===")
for device, before, after, added in oversampling_stats:
    lines.append(
        f"{device}: {before} → {after} (+{added})"
    )

lines.append("\n=== DEVICE CATEGORY DISTRIBUTION (AFTER) ===")
for _, r in device_after.iterrows():
    lines.append(f"{r[DEVICE_COL]}: {r['count']}")

# ================= SAVE REPORT =================

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# ================= PRINT =================

print("\n".join(lines))
print(f"\nOversampled dataset saved to: {OUTPUT_CSV}")
print(f"Report saved to: {REPORT_FILE}")
