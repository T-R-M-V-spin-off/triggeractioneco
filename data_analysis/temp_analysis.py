import pandas as pd
from collections import Counter

# ================= CONFIG =================

INPUT_CSV = "../dataset/filtered_data/action_channel_filtered_dataset.csv"
REPORT_FILE = "temp_analysis_report.txt"

TRIGGER_DESC_COL = "triggerDesc"
ACTION_DESC_COL = "actionDesc"
DEVICE_CATEGORY_COL = "deviceCategory"

UNIQUE_CHARS = 64

# ================= LOAD DATA =================

df = pd.read_csv(INPUT_CSV)

# riempi eventuali valori mancanti
df[TRIGGER_DESC_COL] = df[TRIGGER_DESC_COL].fillna("")
df[ACTION_DESC_COL] = df[ACTION_DESC_COL].fillna("")

# ================= CREA CHIAVI UNICHE =================

df["trigger_key"] = df[TRIGGER_DESC_COL].str[:UNIQUE_CHARS]
df["action_key"] = df[ACTION_DESC_COL].str[:UNIQUE_CHARS]

# ================= FUNZIONE DI REPORT =================

def generate_distribution(df, col_key, col_name):
    """
    Calcola la distribuzione dei valori unici raggruppati per deviceCategory.
    """
    report = []
    device_categories = df[DEVICE_CATEGORY_COL].unique()

    for device in device_categories:
        subset = df[df[DEVICE_CATEGORY_COL] == device]
        counts = Counter(subset[col_key])
        total = sum(counts.values())
        report.append(f"--- Device Category: {device} ---")
        report.append(f"Totale righe: {len(subset)}")
        report.append(f"Valori unici ({col_name} primi {UNIQUE_CHARS} caratteri): {len(counts)}")
        report.append("Top 10 valori unici:")
        for val, c in counts.most_common(10):
            val_str = val.replace("\n", " ")  # evita a capo
            report.append(f"  {val_str}: {c} ({c/total*100:.2f}%)")
        report.append("")  # riga vuota tra categorie

    return report

# ================= GENERA REPORT =================

report_lines = ["=== TRIGGER DESCRIPTION DISTRIBUTION ==="]
report_lines += generate_distribution(df, "trigger_key", "triggerDesc")
report_lines.append("\n=== ACTION DESCRIPTION DISTRIBUTION ===")
report_lines += generate_distribution(df, "action_key", "actionDesc")

# ================= SALVA REPORT =================

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

# ================= PRINT =================

print("\n".join(report_lines))
print(f"\nReport salvato in: {REPORT_FILE}")
