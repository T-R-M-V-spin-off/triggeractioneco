import pandas as pd
import re

# ================= CONFIG =================
INPUT_CSV = "../../dataset/rules/expanded_dataset.csv"
OUTPUT_CSV = "../../dataset/rules/rules_structured_dataset.csv"
REPORT_FILE = "tmp_rules_report.txt"

# Colonne dal dataset
TRIGGER_DESC_COL = "triggerDesc"
ACTION_DESC_COL = "actionDesc"
TRIGGER_TYPE_COL = "triggerType"
ACTION_TYPE_COL = "actionType"
DEVICE_CATEGORY_COL = "deviceCategory"

# ================= LOAD DATASET =================
df = pd.read_csv(INPUT_CSV)

df[TRIGGER_DESC_COL] = df[TRIGGER_DESC_COL].fillna("").str.lower()
df[ACTION_DESC_COL] = df[ACTION_DESC_COL].fillna("").str.lower()

# ================= FUNZIONI =================

# Funzione per trasformare triggerType in frase leggibile
def trigger_to_text(row):
    t_type = row[TRIGGER_TYPE_COL]
    device = row[DEVICE_CATEGORY_COL].lower()
    trigger_desc = row[TRIGGER_DESC_COL]

    if t_type == "motion":
        return f"motion is detected in the {device}"
    elif t_type == "schedule":
        # cerca eventuale riferimento a tempo
        m = re.search(r"sunset|sunrise|specific time|within \d+ minutes|hour|minute|day", trigger_desc)
        if m:
            return f"it is {m.group()}"
        else:
            return "scheduled time is reached"
    elif t_type == "voice":
        return "a voice command is issued"
    elif t_type == "device_state":
        return trigger_desc if trigger_desc else "device state changed"
    elif t_type == "weather":
        return trigger_desc if trigger_desc else "weather condition occurs"
    elif t_type == "audio":
        return trigger_desc if trigger_desc else "audio event detected"
    elif t_type == "external_service":
        return f"external service triggers ({trigger_desc})"
    else:
        return trigger_desc if trigger_desc else "trigger occurs"

# Funzione per trasformare actionType in frase leggibile
def action_to_text(row):
    a_type = row[ACTION_TYPE_COL]
    device = row[DEVICE_CATEGORY_COL].lower()
    action_desc = row[ACTION_DESC_COL]

    if a_type == "turn_on_off":
        return f"turn on the {device}"
    elif a_type == "set_value":
        return f"set value of the {device}"
    elif a_type == "set_mode":
        return f"set mode of the {device}"
    elif a_type == "toggle":
        return f"toggle the {device}"
    elif a_type == "effect":
        return f"apply effect to the {device}"
    elif a_type == "start_stop":
        return f"start/stop the {device}"
    elif a_type == "notify":
        return f"send notification"
    else:
        return action_desc if action_desc else f"perform action on {device}"

# Funzione per estrarre parametri come brightness, temperature, duration
def extract_parameters(action_desc):
    params = []
    duration = None

    # brightness
    m = re.search(r"brightness\s*=\s*\d+%?", action_desc)
    if m:
        params.append(m.group())

    # temperature
    m = re.search(r"temperature\s*=\s*\d+", action_desc)
    if m:
        params.append(m.group())

    # duration
    m = re.search(r"for\s*(\d+\s*(seconds|minutes|hours))", action_desc)
    if m:
        duration = m.group(1)

    return ", ".join(params) if params else None, duration

# ================= COSTRUZIONE REGOLE =================
rule_texts = []
rule_params = []
rule_durations = []

for _, row in df.iterrows():
    trigger_text = trigger_to_text(row)
    action_text = action_to_text(row)
    params, duration = extract_parameters(row[ACTION_DESC_COL])

    # Costruzione della regola
    rule = f"WHEN {trigger_text} THEN {action_text}"
    if params:
        rule += f" WITH {params}"
    if duration:
        rule += f" FOR {duration}"

    rule_texts.append(rule)
    rule_params.append(params)
    rule_durations.append(duration)

df["ruleText"] = rule_texts
df["parameters"] = rule_params
df["duration"] = rule_durations

# ================= SALVATAGGIO =================
df.to_csv(OUTPUT_CSV, index=False)

# ================= REPORT =================
total_rules = len(df)
trigger_counts = df[TRIGGER_TYPE_COL].value_counts()
action_counts = df[ACTION_TYPE_COL].value_counts()
device_counts = df[DEVICE_CATEGORY_COL].value_counts()

report_lines = [
    f"=== RULES STRUCTURED REPORT ===",
    f"Total rules generated: {total_rules}",
    "",
    "=== TRIGGER TYPE DISTRIBUTION ==="
]
for t, c in trigger_counts.items():
    report_lines.append(f"{t}: {c} ({c / total_rules * 100:.2f}%)")

report_lines.append("")
report_lines.append("=== ACTION TYPE DISTRIBUTION ===")
for a, c in action_counts.items():
    report_lines.append(f"{a}: {c} ({c / total_rules * 100:.2f}%)")

report_lines.append("")
report_lines.append("=== DEVICE CATEGORY DISTRIBUTION ===")
for d, c in device_counts.items():
    report_lines.append(f"{d}: {c} ({c / total_rules * 100:.2f}%)")

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

# ================= PRINT =================
print("\n".join(report_lines))
print(f"\nStructured rules dataset saved to: {OUTPUT_CSV}")
print(f"Report saved to: {REPORT_FILE}")
