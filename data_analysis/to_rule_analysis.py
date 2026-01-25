import pandas as pd

# ================= LOAD CSV =================
TRANSFORMED_CSV = "../data_processing/transformed_dataset.csv"  # dataset trasformato
ORIGINAL_CSV = "../dataset/action_channel_filtered_dataset.csv"  # dataset con deviceCategory

df_transformed = pd.read_csv(TRANSFORMED_CSV)
df_original = pd.read_csv(ORIGINAL_CSV)[['id', 'deviceCategory']]

# Fai merge per aggiungere deviceCategory al dataset trasformato
df = df_transformed.merge(df_original, on='id', how='left')

# Mantieni solo le colonne rilevanti
df = df[['id', 'triggerDesc', 'actionDesc', 'deviceCategory']].copy()


# ================= ANALISI DISTRIBUZIONE =================
def analyze_distribution(column, df_full, top_n=50):
    """
    Analizza la distribuzione di una colonna, restituisce counts ordinati per frequenza
    e un report testuale. Calcola la distribuzione deviceCategory usando df_full (con id->category).
    """
    counts = column.value_counts(dropna=False)
    total = counts.sum()
    report_lines = [
        f"Analisi della colonna '{column.name}':",
        f"Totale valori: {total}",
        f"Valori unici: {counts.shape[0]}",
        "",
        f"Top {top_n} valori più frequenti:"
    ]

    for i, (val, freq) in enumerate(counts.head(top_n).items(), 1):
        val_display = val if pd.notna(val) else "<NA>"
        percent = (freq / total) * 100
        report_lines.append(f"{i}. {val_display} - {freq} occorrenze ({percent:.2f}%)")

    # ================= ISTANZE DOPO TOP_N =================
    if counts.shape[0] > top_n:
        after_top_n_count = counts.iloc[top_n:].sum()
        remaining_rows_if_removed = total - after_top_n_count
        report_lines.append("")
        report_lines.append(f"Numero di istanze totali dopo il {top_n}° valore unico: {after_top_n_count}")
        report_lines.append(f"Righe rimanenti se si eliminano tutte le entry oltre il {top_n}° valore unico: {remaining_rows_if_removed}")
    else:
        after_top_n_count = 0
        remaining_rows_if_removed = total
        report_lines.append("")
        report_lines.append(f"Totale valori unici <= {top_n}, nessuna entry oltre il {top_n}° valore unico.")

    # ================= DISTRIBUZIONE DEVICE CATEGORY =================
    report_lines.append("\nDistribuzione deviceCategory (senza eliminazioni):")
    device_counts_before = df_full['deviceCategory'].value_counts(dropna=False)
    for cat, freq in device_counts_before.items():
        report_lines.append(f"{cat}: {freq}")

    # Filtra top_n
    top_n_values = set(counts.head(top_n).index)
    df_top_n = df_full[df_full[column.name].isin(top_n_values)]
    report_lines.append("\nDistribuzione deviceCategory (dopo eliminazioni post top_n):")
    device_counts_after = df_top_n['deviceCategory'].value_counts(dropna=False)
    for cat, freq in device_counts_after.items():
        report_lines.append(f"{cat}: {freq}")

    report_text = "\n".join(report_lines)
    return counts, report_text, remaining_rows_if_removed, after_top_n_count, top_n_values

# ================= ANALISI =================
# Trigger
trigger_counts, trigger_report, trigger_remaining, trigger_after_50, top_50_trigger = analyze_distribution(
    df['triggerDesc'], df, top_n=50)

# Action
action_counts, action_report, action_remaining, action_after_50, top_50_action = analyze_distribution(
    df['actionDesc'], df, top_n=50)

# ================= RIGHE RIMANENTI SE FILTRIAMO TOP 50 TRIGGER + ACTION =================
mask_top50_both = df['triggerDesc'].isin(top_50_trigger) & df['actionDesc'].isin(top_50_action)
remaining_rows_both = mask_top50_both.sum()

# ================= GENERA REPORT =================
REPORT_FILE = "trigger_action_distribution_with_device_report.txt"
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("=== DISTRIBUZIONE TRIGGER DESCRIPTIONS ===\n")
    f.write(trigger_report + "\n\n")
    f.write("=== DISTRIBUZIONE ACTION DESCRIPTIONS ===\n")
    f.write(action_report + "\n\n")
    f.write(f"Righe rimanenti se si eliminano tutte le entry oltre il 50° valore unico sia di trigger che di action: {remaining_rows_both}\n")

print(f"Report generato in {REPORT_FILE}")
print("\n--- Anteprima ---\n")
print(trigger_report.splitlines()[0])
print(action_report.splitlines()[0])
print(f"Righe rimanenti se filtriamo top 50 trigger + action: {remaining_rows_both}")
