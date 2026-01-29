import pandas as pd
import itertools
import os

# ================= CONFIG =================

INPUT_CSV = "../../dataset/rules/oversampled_seed_dataset.csv"  # dataset bilanciato
OUTPUT_CSV = "../../dataset/rules/expanded_dataset.csv"
REPORT_FILE = "variants_report.txt"

TRIGGER_DESC_COL = "triggerDesc"
ACTION_DESC_COL = "actionDesc"
TRIGGER_TYPE_COL = "triggerType"
ACTION_TYPE_COL = "actionType"
DEVICE_COL = "deviceCategory"

# Numero di varianti per seed
VARIANTS_PER_SEED = 3

# ================= VARIANT DICTIONARIES =================

TRIGGER_REPHRASES = {
    "motion": [
        "when motion is detected",
        "upon detecting movement",
        "if movement occurs"
    ],
    "schedule": [
        "at the scheduled time",
        "according to the defined schedule",
        "when the specified time occurs"
    ],
    "device_state": [
        "when the device state changes",
        "upon changing device state",
        "if the device is turned on/off"
    ],
    "voice": [
        "when a voice command is received",
        "if a spoken command is detected",
        "upon voice instruction"
    ],
    "external_service": [
        "when an external service triggers",
        "upon receiving signal from another service",
        "if an external event occurs"
    ],
    "weather": [
        "when the weather condition changes",
        "if weather meets the condition",
        "upon change in weather"
    ],
    "audio": [
        "when an audio event is detected",
        "upon sensing a sound",
        "if audio is detected"
    ],
    "other": [
        "when the event occurs",
        "upon event trigger",
        "if the condition is met"
    ]
}

ACTION_REPHRASES = {
    "turn_on_off": [
        "turn the device on/off",
        "switch the device",
        "power on/off the device"
    ],
    "set_value": [
        "adjust the value",
        "set the specified value",
        "change to the desired value"
    ],
    "set_mode": [
        "set the mode",
        "activate the scene",
        "apply the mode setting"
    ],
    "toggle": [
        "toggle the state",
        "switch between states",
        "change state"
    ],
    "effect": [
        "apply the effect",
        "trigger visual effect",
        "start effect sequence"
    ],
    "start_stop": [
        "start or stop the device",
        "initiate or halt the device",
        "begin or end the action"
    ],
    "notify": [
        "send notification",
        "alert the user",
        "trigger message"
    ],
    "other": [
        "perform the action",
        "execute the task",
        "carry out the operation"
    ]
}

# ================= LOAD DATASET =================

df = pd.read_csv(INPUT_CSV)
df = df.fillna("")

# ================= GENERATE VARIANTS =================

expanded_rows = []

for idx, row in df.iterrows():
    trigger_type = row[TRIGGER_TYPE_COL]
    action_type = row[ACTION_TYPE_COL]

    # Seleziona le frasi di variazione
    trigger_variants = TRIGGER_REPHRASES.get(trigger_type, [row[TRIGGER_DESC_COL]])
    action_variants = ACTION_REPHRASES.get(action_type, [row[ACTION_DESC_COL]])

    # Limita a VARIANTS_PER_SEED
    trigger_variants = trigger_variants[:VARIANTS_PER_SEED]
    action_variants = action_variants[:VARIANTS_PER_SEED]

    # Genera combinazioni
    for t_var, a_var in itertools.product(trigger_variants, action_variants):
        new_row = row.copy()
        new_row[TRIGGER_DESC_COL] = t_var
        new_row[ACTION_DESC_COL] = a_var
        new_row["_generated"] = True
        new_row["_variant_id"] = f"{idx}_{t_var[:3]}_{a_var[:3]}"
        expanded_rows.append(new_row)

# Crea DataFrame finale
df_expanded = pd.DataFrame(expanded_rows)

# ================= SAVE DATASET =================

df_expanded.to_csv(OUTPUT_CSV, index=False)

# ================= REPORT =================

total_original = len(df)
total_generated = len(df_expanded)

trigger_counts_before = df[TRIGGER_TYPE_COL].value_counts()
trigger_counts_after = df_expanded[TRIGGER_TYPE_COL].value_counts()

action_counts_before = df[ACTION_TYPE_COL].value_counts()
action_counts_after = df_expanded[ACTION_TYPE_COL].value_counts()

report_lines = [
    f"=== DATA GENERATION REPORT ===",
    f"Original rows: {total_original}",
    f"Generated rows: {total_generated}",
    f"Total new rows: {total_generated - total_original}",
    ""
]

report_lines.append("=== TRIGGER TYPE DISTRIBUTION (BEFORE / AFTER) ===")
for t in trigger_counts_before.index:
    before = trigger_counts_before[t]
    after = trigger_counts_after.get(t, 0)
    report_lines.append(f"{t}: {before} → {after}")

report_lines.append("")
report_lines.append("=== ACTION TYPE DISTRIBUTION (BEFORE / AFTER) ===")
for a in action_counts_before.index:
    before = action_counts_before[a]
    after = action_counts_after.get(a, 0)
    report_lines.append(f"{a}: {before} → {after}")

report_lines.append("")
report_lines.append("=== DEVICE CATEGORY DISTRIBUTION (BEFORE / AFTER) ===")
device_counts_before = df[DEVICE_COL].value_counts()
device_counts_after = df_expanded[DEVICE_COL].value_counts()
for d in device_counts_before.index:
    before = device_counts_before[d]
    after = device_counts_after.get(d, 0)
    report_lines.append(f"{d}: {before} → {after}")

# Save report
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

# Print report
print("\n".join(report_lines))
print(f"\nExpanded dataset saved in: {OUTPUT_CSV}")
print(f"Report saved in: {REPORT_FILE}")
