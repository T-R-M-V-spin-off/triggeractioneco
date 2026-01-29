import pandas as pd
import random

# ================= CONFIG =================

INPUT_FILE = "final_with_eco_score.csv"
OUTPUT_FILE = "final_with_variants.csv"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ================= TEMPLATES =================

TEMPLATES = {

    "Air/Climate": {

        "eco": [
            "{rule} FOR 20 minutes WHEN no presence",
            "{rule} ONLY if temperature is critical",
            "{rule} WITH eco ventilation mode",
            "{rule} WHEN windows are closed",
            "{rule} IF air quality is poor FOR 15 minutes"
        ],

        "neutral": [
            "{rule}",
            "{rule} WITH default settings",
            "{rule} WHEN activated",
            "{rule} USING standard mode"
        ],

        "noneco": [
            "{rule} FOR all day",
            "{rule} ALWAYS ON",
            "{rule} WITH max airflow",
            "{rule} WITHOUT auto shutdown",
            "{rule} RUN continuously"
        ]
    },


    "Appliance": {

        "eco": [
            "{rule} ONLY during off-peak hours",
            "{rule} WITH energy saving mode",
            "{rule} WHEN load is full",
            "{rule} USING low power cycle",
            "{rule} IF battery level is high"
        ],

        "neutral": [
            "{rule}",
            "{rule} USING default program",
            "{rule} WHEN started",
            "{rule} WITH normal mode"
        ],

        "noneco": [
            "{rule} IMMEDIATELY",
            "{rule} REPEAT every hour",
            "{rule} WITH maximum power",
            "{rule} ALWAYS ACTIVE",
            "{rule} WITHOUT power limits"
        ]
    },


    "Thermostat/Heating": {

        "eco": [
            "{rule} SET to 19C at night",
            "{rule} WHEN nobody is home",
            "{rule} WITH adaptive heating",
            "{rule} ONLY during cold peaks",
            "{rule} USING smart scheduling"
        ],

        "neutral": [
            "{rule}",
            "{rule} USING manual mode",
            "{rule} WITH default schedule",
            "{rule} WHEN requested"
        ],

        "noneco": [
            "{rule} SET to 25C all day",
            "{rule} RUN continuously",
            "{rule} WITHOUT temperature limits",
            "{rule} OVERRIDE all schedules",
            "{rule} MAX heating mode"
        ]
    },


    "Light": {

        "eco": [
            "{rule} TURN OFF after 5 minutes",
            "{rule} WHEN daylight is detected",
            "{rule} USING low brightness",
            "{rule} IF no motion detected",
            "{rule} WITH ambient mode"
        ],

        "neutral": [
            "{rule}",
            "{rule} WITH default brightness",
            "{rule} USING normal lighting",
            "{rule} WHEN switched"
        ],

        "noneco": [
            "{rule} FULL BRIGHTNESS all day",
            "{rule} ALWAYS ON",
            "{rule} WITHOUT dimming",
            "{rule} MAX LUMENS",
            "{rule} NEVER auto-off"
        ]
    },


    "Generic/Other": {

        "eco": [
            "{rule} WITH power optimization",
            "{rule} WHEN idle",
            "{rule} IN low energy mode",
            "{rule} USING smart control"
        ],

        "neutral": [
            "{rule}",
            "{rule} WITH default configuration",
            "{rule} WHEN triggered"
        ],

        "noneco": [
            "{rule} WITHOUT limits",
            "{rule} ALWAYS RUNNING",
            "{rule} MAX PERFORMANCE",
            "{rule} NO shutdown"
        ]
    }
}


# ================= HELPERS =================

def get_category_templates(category):

    if category in TEMPLATES:
        return TEMPLATES[category]

    # fallback
    return TEMPLATES["Generic/Other"]


def generate_variant(rule, category, mode):

    templates = get_category_templates(category)

    pool = templates.get(mode, templates["neutral"])

    template = random.choice(pool)

    return template.format(rule=rule)


# ================= MAIN LOGIC =================

def build_variants(row):

    base = row["ifttt_rule"]
    label = row["eco_score"]
    category = row["deviceCategory"]

    eco = generate_variant(base, category, "eco")
    neutral = generate_variant(base, category, "neutral")
    noneco = generate_variant(base, category, "noneco")

    # Rispetta la label originale
    if label == "Eco":
        eco = base

    elif label == "Neutra":
        neutral = base

    elif label == "Non-Eco":
        noneco = base

    return pd.Series([neutral, eco, noneco])


# ================= RUN =================

def main():

    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print("Generating variants...")

    df[["neutral_rule", "eco_rule", "noneco_rule"]] = df.apply(
        build_variants,
        axis=1
    )

    # Check
    print("\nSample:")

    print(df[[
        "deviceCategory",
        "eco_score",
        "ifttt_rule",
        "neutral_rule",
        "eco_rule",
        "noneco_rule"
    ]].head(5))

    # Save
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved in: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()