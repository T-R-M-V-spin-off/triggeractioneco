import os
import pandas as pd
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================

INPUT_CSV = "../dataset/raw_dataset.csv"
OUTPUT_DIR = "plots"
REPORT_FILE = os.path.join(OUTPUT_DIR, "channel_distribution_report.txt")

ACTION_COLUMN = "actionChannelTitle"
TRIGGER_COLUMN = "triggerChannelTitle"

MIN_THRESHOLD = 10

# ================= SETUP =================

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

# ================= COUNT VALUES =================

action_counts = df[ACTION_COLUMN].value_counts(dropna=True).to_dict()
trigger_counts = df[TRIGGER_COLUMN].value_counts(dropna=True).to_dict()

# ================= FILTER + SORT =================

def filter_and_sort(counter, threshold):
    """
    Keep only channels with count >= threshold
    and sort them in descending order.
    """
    filtered = {k: v for k, v in counter.items() if v >= threshold}
    sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    return sorted_items

action_sorted = filter_and_sort(action_counts, MIN_THRESHOLD)
trigger_sorted = filter_and_sort(trigger_counts, MIN_THRESHOLD)

# ================= PLOTTING FUNCTION =================

def plot_distribution(data, title, filename, xlabel, ylabel):
    names = [x[0] for x in data]
    values = [x[1] for x in data]

    plt.figure(figsize=(16, 7))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(names)), names, rotation=90, fontsize=8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, format="svg")
    plt.close()
    return output_path

# ================= GENERATE PLOTS =================

action_plot_path = plot_distribution(
    action_sorted,
    "Action Channels (>=10 entries) - Sorted",
    "action_channels_distribution.svg",
    "Channel Name",
    "Entry Count"
)

trigger_plot_path = plot_distribution(
    trigger_sorted,
    "Trigger Channels (>=10 entries) - Sorted",
    "trigger_channels_distribution.svg",
    "Channel Name",
    "Entry Count"
)

# ================= GENERATE REPORT =================

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("SMART HOME DATASET - CHANNEL DISTRIBUTION REPORT\n")
    f.write("="*60 + "\n\n")

    # -------- ACTION CHANNELS --------
    f.write("ACTION CHANNELS STATISTICS\n")
    f.write("-"*40 + "\n")
    f.write(f"Total unique action channels: {len(action_counts)}\n")
    f.write(f"Action channels >= {MIN_THRESHOLD}: {len(action_sorted)}\n")
    f.write(f"Action channels < {MIN_THRESHOLD}: {len(action_counts) - len(action_sorted)}\n")
    f.write("\nTop 10 Action Channels:\n")
    for name, count in action_sorted[:10]:
        f.write(f"{name}: {count}\n")
    f.write("\nFull distribution (>= threshold) sorted:\n")
    for name, count in action_sorted:
        f.write(f"{name}: {count}\n")
    f.write("\n")

    # -------- TRIGGER CHANNELS --------
    f.write("TRIGGER CHANNELS STATISTICS\n")
    f.write("-"*40 + "\n")
    f.write(f"Total unique trigger channels: {len(trigger_counts)}\n")
    f.write(f"Trigger channels >= {MIN_THRESHOLD}: {len(trigger_sorted)}\n")
    f.write(f"Trigger channels < {MIN_THRESHOLD}: {len(trigger_counts) - len(trigger_sorted)}\n")
    f.write("\nTop 10 Trigger Channels:\n")
    for name, count in trigger_sorted[:10]:
        f.write(f"{name}: {count}\n")
    f.write("\nFull distribution (>= threshold) sorted:\n")
    for name, count in trigger_sorted:
        f.write(f"{name}: {count}\n")
    f.write("\n")

    # -------- PLOT PATHS --------
    f.write("PLOT FILES:\n")
    f.write(f"Action channels plot: {action_plot_path}\n")
    f.write(f"Trigger channels plot: {trigger_plot_path}\n")

# ================= SUMMARY =================

print("Plots and report generated successfully!\n")
print(f"Report saved at: {REPORT_FILE}")
print(f"Action plot: {action_plot_path}")
print(f"Trigger plot: {trigger_plot_path}")
