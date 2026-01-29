import pandas as pd
import re

# ================= CONFIG =================

INPUT_FILE = "final_aggregated_dataset.csv"
OUTPUT_FILE = "final_with_eco_score.csv"


# ================= KEYWORDS =================

OCCUPANCY_KEYWORDS = [
    "enter", "exit", "away", "home", "presence", "location", "arrive", "leave",
]

ENERGY_SAVE_KEYWORDS = [
    "turn off", "off", "sleep", "eco", "low", "disable", "stop", "reduce"
]

ENERGY_WASTE_KEYWORDS = [
    "high", "max", "boost", "full", "start"
]

AUTOMATION_KEYWORDS = [
    "every day", "schedule", "sunrise", "sunset", "time", "calendar",
    "weather", "temperature", "humidity"
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

    for k in ENERGY_WASTE_KEYWORDS:
        if k in text:
            score -= 1

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


def compute_waste_risk(row):
    text = " ".join([
        normalize(row.get("actionTitle")),
        normalize(row.get("actionDesc"))
    ])

    if any(k in text for k in ["on", "start", "max", "boost"]):
        return 1.0

    return 0.0


# ================= MAIN SCORE =================

def compute_eco_numeric(row):

    occ = compute_occupancy_awareness(row)
    energy = compute_energy_saving(row)
    auto = compute_automation(row)
    waste = compute_waste_risk(row)

    # Pesi (puoi modificarli per esperimenti)
    score = (
        0.35 * occ +
        0.35 * energy +
        0.20 * auto -
        0.30 * waste
    )

    # Normalizza in 0-100
    score_100 = (score + 1) * 50

    return round(score_100, 2)


def map_to_label(score):

    if score >= 65:
        return "Eco"

    elif score >= 40:
        return "Neutra"

    else:
        return "Non-Eco"


# ================= RUN =================

def main():

    print("Caricamento dataset...")
    df = pd.read_csv(INPUT_FILE)

    print("Calcolo eco-metriche...")

    df["eco_score_value"] = df.apply(compute_eco_numeric, axis=1)

    df["eco_score"] = df["eco_score_value"].apply(map_to_label)

    # Statistiche
    print("\nDistribuzione Eco Score:")
    print(df["eco_score"].value_counts(normalize=True) * 100)

    print("\nMedia score:", df["eco_score_value"].mean())

    # Salvataggio
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nFile salvato in: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()