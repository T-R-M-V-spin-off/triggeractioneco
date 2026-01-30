from statistics import mean

import pandas as pd

# ================= CONFIG =================

INPUT_FILE = "final_aggregated_dataset.csv"
OUTPUT_FILE = "final_with_eco_score.csv"


# ================= KEYWORDS =================

OCCUPANCY_KEYWORDS = [
    "enter", "exit", "away", "home", "presence", "location", "arrive", "leave", "left", "motion", "arrives", "leaves", "person"
]

ENERGY_SAVE_KEYWORDS = [
    "sleep", "eco", "low", "disable", "stop", "reduce", "specified", "threshold", "specific", "temporarily"
]

AUTOMATION_KEYWORDS = [
    "every day", "schedule", "sunrise", "sunset", "time", "calendar",
    "weather", "above", "below", "specific day", "scheduled",
    "specific day"
]


# ================= UTILS =================

def normalize(text):
    """Lowercase + safe string"""
    if pd.isna(text):
        return ""
    return str(text).lower()


# ================= METRICS =================

def compute_occupancy_awareness(row):
    text = " ".join([
        normalize(row.get("triggerTitle")),
        normalize(row.get("triggerDesc"))
    ])

    for k in OCCUPANCY_KEYWORDS:
        if k in text:
            return 1.0

    return 0.0


def compute_energy_saving(row):
    text = " ".join([
        normalize(row.get("actionTitle")),
        normalize(row.get("actionDesc"))
    ])

    score = 0

    for k in ENERGY_SAVE_KEYWORDS:
        if k in text:
            score += 1

    return max(min(score, 2), -2) / 2   # normalizza tra -1 e +1


def compute_automation(row):
    text = " ".join([
        normalize(row.get("triggerTitle")),
        normalize(row.get("triggerDesc"))
    ])

    for k in AUTOMATION_KEYWORDS:
        if k in text:
            return 1.0

    return 0.0


# ================= MAIN SCORE =================

def compute_eco_numeric(row):

    occ = compute_occupancy_awareness(row)
    energy = compute_energy_saving(row)
    auto = compute_automation(row)

    # Pesi (puoi modificarli per esperimenti)
    score = mean([occ, energy, auto])

    # Normalizza in 0-100
    score_100 = (score + 1) * 50

    return round(score_100, 2)


def map_to_label(score):

    if score >= 65:
        return "Eco"

    elif score >= 40:
        return "Neutra"

    else:
        return 0.5

    df["eco_score"] = df["eco_score_value"].apply(map_to_label)

    # Statistiche
    print("\nDistribuzione Eco Score:")
    print(df["eco_score"].value_counts(normalize=True) * 100)
