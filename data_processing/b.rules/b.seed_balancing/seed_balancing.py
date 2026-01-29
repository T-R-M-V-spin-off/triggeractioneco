import pandas as pd
from pathlib import Path

# ================= CONFIG =================

INPUT_CSV = "../../../dataset/rules/types_augmented_dataset.csv"
REPORT_FILE = "seed_balancing_report.txt"
OUTPUT_CSV = "../../../dataset/rules/balanced_seed_dataset.csv"

TRIGGER_TYPE_COL = "triggerType"
ACTION_TYPE_COL = "actionType"
DEVICE_COL = "deviceCategory"

CAP = 300
RANDOM_SEED = 42

# ================= UTILS =================

def dist(df, col):
    return (
        df[col]
        .value_counts()
        .rename_axis(col)
        .reset_index(name="count")
    )

def cap_group(df, col, cap):
    return (
        df
        .groupby(col, group_keys=False)
        .apply(lambda x: x.sample(
            n=min(len(x), cap),
            random_state=RANDOM_SEED
        ))
        .reset_index(drop=True)
    )

# ================= LOAD =================

df = pd.read_csv(INPUT_CSV)
initial_rows = len(df)

# ================= DISTRIBUTIONS BEFORE =================

trigger_before = dist(df, TRIGGER_TYPE_COL)
action_before = dist(df, ACTION_TYPE_COL)
device_before = dist(df, DEVICE_COL)

# ================= CAP BY TRIGGER TYPE =================

df_capped_trigger = cap_group(df, TRIGGER_TYPE_COL, CAP)

after_trigger_rows = len(df_capped_trigger)

# ================= CAP BY ACTION TYPE =================

df_balanced = cap_group(df_capped_trigger, ACTION_TYPE_COL, CAP)

final_rows = len(df_balanced)

# ================= DISTRIBUTIONS AFTER =================

trigger_after = dist(df_balanced, TRIGGER_TYPE_COL)
action_after = dist(df_balanced, ACTION_TYPE_COL)
device_after = dist(df_balanced, DEVICE_COL)

# ================= SAVE DATASET =================

Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
df_balanced.to_csv(OUTPUT_CSV, index=False)

# ================= REPORT =================

lines = []

lines.append("=== SEED BALANCING REPORT (CAP = 300) ===")
lines.append(f"Initial rows: {initial_rows}")
lines.append(f"Rows after trigger cap: {after_trigger_rows}")
lines.append(f"Final rows: {final_rows}")
lines.append(f"Total rows removed: {initial_rows - final_rows}")

# -------- TRIGGER --------

lines.append("\n=== TRIGGER TYPE DISTRIBUTION (BEFORE / AFTER) ===")

trigger_dist = (
    trigger_before
    .rename(columns={"count": "count_before"})
    .merge(
        trigger_after.rename(columns={"count": "count_after"}),
        on=TRIGGER_TYPE_COL,
        how="left"
    )
    .fillna(0)
)

for _, r in trigger_dist.iterrows():
    lines.append(
        f"{r[TRIGGER_TYPE_COL]}: "
        f"{int(r['count_before'])} → {int(r['count_after'])}"
    )

# -------- ACTION --------

lines.append("\n=== ACTION TYPE DISTRIBUTION (BEFORE / AFTER) ===")

action_dist = (
    action_before
    .rename(columns={"count": "count_before"})
    .merge(
        action_after.rename(columns={"count": "count_after"}),
        on=ACTION_TYPE_COL,
        how="left"
    )
    .fillna(0)
)

for _, r in action_dist.iterrows():
    lines.append(
        f"{r[ACTION_TYPE_COL]}: "
        f"{int(r['count_before'])} → {int(r['count_after'])}"
    )

# -------- DEVICE --------

lines.append("\n=== DEVICE CATEGORY DISTRIBUTION (BEFORE / AFTER) ===")

device_dist = (
    device_before
    .rename(columns={"count": "count_before"})
    .merge(
        device_after.rename(columns={"count": "count_after"}),
        on=DEVICE_COL,
        how="left"
    )
    .fillna(0)
)

for _, r in device_dist.iterrows():
    lines.append(
        f"{r[DEVICE_COL]}: "
        f"{int(r['count_before'])} → {int(r['count_after'])}"
    )

# ================= SAVE REPORT =================

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# ================= PRINT =================

print("\n".join(lines))
print(f"\nBalanced dataset saved to: {OUTPUT_CSV}")
print(f"Report saved to: {REPORT_FILE}")
