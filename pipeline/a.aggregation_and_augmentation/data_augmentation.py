import pandas as pd
import re

# ================= CONFIG =================

INPUT_CSV = "../../dataset/filtered_data/aggregated_dataset.csv"
OUTPUT_CSV = "../../dataset/filtered_data/augmented_dataset.csv"
REPORT_FILE = "data_augmentation_report.txt"

TRIGGER_DESC_COL = "triggerDesc"
ACTION_DESC_COL = "actionDesc"
DEVICE_CATEGORY_COL = "deviceCategory"

# ================= LOAD DATA =================

df = pd.read_csv(INPUT_CSV)

df[TRIGGER_DESC_COL] = df[TRIGGER_DESC_COL].fillna("").str.lower()
df[ACTION_DESC_COL] = df[ACTION_DESC_COL].fillna("").str.lower()

# ================= TRIGGER TYPE =================

TRIGGER_KEYWORDS = {
    "audio": ["alexa", "ok google", "say", "audio", "audio event", "sound"],
    "weather": ["temperature", "weather", "temperatures", "rain", "solar", "humidity", "co2", "carbon", "air", "pollution", "volatile", "compounds"],
    "manual_interaction": ["press", "click", "add", "pressed", "clicked", "added", "prayer"],
    "external_service": ["facebook", "android", "calendar", "apilio",
                         "spotify", "youtube", "ifttt", "security", "fitbits",
                         "logs", "log", "watch", "lotus", "post", "tweet", "hashtag",
                         "alarm", "wifi", "wi-fi", "breaking news", "ballotpedia", "routine", "spreadsheet", "raised", "runs"],
    "pricing": ["price", "pricing"],
    "motion": ["enter an area", "exit an area", "member arrives", "member leaves", "motion", "exit", "enter", "area", "left", "movement", "movements", "face"],
    "device_state": ["scene is changed", "turned on", "turned off", "unlocked", "opened",
                     "closed", "locked", "detected", "armed", "exchanged", "off", "on", "starts", "stops", "opens", "closes", "ends", "begins", "mode"],
    "schedule": [
        "sunset", "sunrise", "specific time", "within \\d+ minutes", "hours", "hour", "minutes", "minute", "seconds", "day",
        "days", "every single day", "once a year", "every time", "only on specific days"
    ]
}

def classify_trigger(text):
    for category, keywords in TRIGGER_KEYWORDS.items():
        # Creo regex sicura per le parole chiave
        pattern = r"\b(" + "|".join(keywords) + r")\b"
        if re.search(pattern, text, re.IGNORECASE):
            return category
    return "other"

df["triggerType"] = df[TRIGGER_DESC_COL].apply(classify_trigger)

# ================= ACTION TYPE =================

ACTION_KEYWORDS = {
    "set_value": ["set .* temperature", "set .* brightness", "dim", "brighten", "brightness", "low", "\\d+\\s?%",
                  "you specify", "specified", "target", "specify", "hours", "hour", "minutes",
                  "minute", "seconds", "day", "days", "fast", "hot", "cold", "slow", "maximum", "configurable"],
    "set_mode": ["set .* scene", "activate .* scene", "guardian scene",
                 "home scene", "away scene", "night scene", "mode", "color", "scene", "auto", "manual", "profile",
                 "settings", "modes", "default", "position", "change"],
    "turn_on_off": ["turn on", "turn off", "switch on", "switch off", "turn", "turns", "toggle", "start", "stop", "cancel",
                    "lock", "unlock", "removes", "activate", "deactivate", "dock", "switch", "enable", "disable",
                    "enabled", "disabled", "pause" , "resume", "resumed", "unpause", "open", "close", "sleep"],
    "effect": ["blink", "breathe", "color loop", "change the color", "pulse"],
    "notify": ["post", "send", "notify", "message", "tweet"]
}

def classify_action(text):
    for category, keywords in ACTION_KEYWORDS.items():
        pattern = r"\b(" + "|".join(keywords) + r")\b"
        if re.search(pattern, text, re.IGNORECASE):
            return category
    return "other"

df["actionType"] = df[ACTION_DESC_COL].apply(classify_action)

# ================= SAVE DATASET =================

df.to_csv(OUTPUT_CSV, index=False)

# ================= BASIC DISTRIBUTION =================

total_rows = len(df)
trigger_counts = df["triggerType"].value_counts()
action_counts = df["actionType"].value_counts()

trigger_other = trigger_counts.get("other", 0)
action_other = action_counts.get("other", 0)

# ================= DEVICE DISTRIBUTION =================

trigger_by_device = df.groupby([DEVICE_CATEGORY_COL, "triggerType"]).size().unstack(fill_value=0)
action_by_device = df.groupby([DEVICE_CATEGORY_COL, "actionType"]).size().unstack(fill_value=0)

# ================= TERNE DISTRIBUZIONE =================

ternary_counts = df.groupby([DEVICE_CATEGORY_COL, "triggerType", "actionType"]).size()
ternary_total = ternary_counts.sum()

max_count = ternary_counts.max()
min_count = ternary_counts.min()
mean_count = ternary_counts.mean()
median_count = ternary_counts.median()
std_count = ternary_counts.std()

# Outlier: terne rare <1% e dominanti >5% del totale
rare_ternes = ternary_counts[ternary_counts / ternary_total < 0.01]
dominant_ternes = ternary_counts[ternary_counts / ternary_total > 0.05]

# ================= REPORT =================

report_lines = [
    "=== DATASET REPORT ===",
    f"Total rows processed: {total_rows}",
    "",
    "=== GLOBAL TRIGGER TYPE DISTRIBUTION ==="
]

# Trigger globale
for t, c in trigger_counts.items():
    report_lines.append(f"{t}: {c} ({c / total_rows * 100:.2f}%)")
report_lines.append(f"Not classified (other): {trigger_other} ({trigger_other / total_rows * 100:.2f}%)")

report_lines.append("")
report_lines.append("=== GLOBAL ACTION TYPE DISTRIBUTION ===")
for a, c in action_counts.items():
    report_lines.append(f"{a}: {c} ({c / total_rows * 100:.2f}%)")
report_lines.append(f"Not classified (other): {action_other} ({action_other / total_rows * 100:.2f}%)")

# ================= TRIGGER x DEVICE =================
report_lines.append("")
report_lines.append("=== TRIGGER TYPE BY DEVICE CATEGORY ===")
for device in trigger_by_device.index:
    report_lines.append(f"\n[{device}]")
    row = trigger_by_device.loc[device]
    total = row.sum()
    for t, c in row.items():
        percent = (c / total) * 100 if total > 0 else 0
        report_lines.append(f"{t}: {c} ({percent:.2f}%)")

# ================= ACTION x DEVICE =================
report_lines.append("")
report_lines.append("=== ACTION TYPE BY DEVICE CATEGORY ===")
for device in action_by_device.index:
    report_lines.append(f"\n[{device}]")
    row = action_by_device.loc[device]
    total = row.sum()
    for a, c in row.items():
        percent = (c / total) * 100 if total > 0 else 0
        report_lines.append(f"{a}: {c} ({percent:.2f}%)")

# ================= TERNE QUANTITATIVE =================
report_lines.append("")
report_lines.append("=== DISTRIBUZIONE TERNE: deviceCategory x triggerType x actionType ===")
for idx, c in ternary_counts.items():
    perc = c / ternary_total * 100
    report_lines.append(f"{idx}: {c} ({perc:.2f}%)")

report_lines.append("")
report_lines.append("=== STATISTICHE QUANTITATIVE TERNE ===")
report_lines.append(f"Totale terne: {len(ternary_counts)}")
report_lines.append(f"Totale combinazioni registrate: {ternary_total}")
report_lines.append(f"Minimo esempi per terna: {min_count} ({min_count/ternary_total*100:.2f}%)")
report_lines.append(f"Massimo esempi per terna: {max_count} ({max_count/ternary_total*100:.2f}%)")
report_lines.append(f"Media esempi per terna: {mean_count:.2f} ({mean_count/ternary_total*100:.2f}%)")
report_lines.append(f"Mediana esempi per terna: {median_count} ({median_count/ternary_total*100:.2f}%)")
report_lines.append(f"Deviazione standard: {std_count:.2f} ({std_count/ternary_total*100:.2f}%)")
report_lines.append(f"Numero terne rare (<1% del totale): {len(rare_ternes)}")
report_lines.append(f"Numero terne dominanti (>5% del totale): {len(dominant_ternes)}")

# ================= SAVE REPORT =================
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

# ================= PRINT =================
print("\n".join(report_lines))
print(f"\nDataset salvato in: {OUTPUT_CSV}")
print(f"Report salvato in: {REPORT_FILE}")