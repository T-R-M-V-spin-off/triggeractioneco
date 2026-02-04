import json
import random

import pandas as pd


# ================= CONFIG =================

INPUT_FILE = "../../dataset/filtered_data/ultra_mega_final.csv"
OUTPUT_FILE = "../../dataset/filtered_data/dpo_dataset.jsonl"


# ================= LOAD =================

df = pd.read_csv(INPUT_FILE)

print(f"Loaded: {len(df)} rows")


# ================= BUILD DPO =================

dpo_samples = []
eco_rules = []
neutral_rules = []
noneco_rules = []


for index, row in enumerate(df.itertuples(index=False)):

    text = str(row.triggerDesc) + " " + str(row.actionDesc)

    if row.isRuleEco == "ECO":
        eco_rules.append(text)

    elif row.isRuleEco == "NEUTRAL":
        neutral_rules.append(text)

    elif row.isRuleEco == "NON_ECO":
        noneco_rules.append(text)


    # Ogni 9 righe
    if (index + 1) % 9 == 0:

        # Sicurezza
        if len(eco_rules) == 3 and len(neutral_rules) == 3 and len(noneco_rules) == 3:

            for _ in range(3):

                prompt = random.choice(neutral_rules)
                neutral_rules.remove(prompt)

                chosen = random.choice(eco_rules)
                eco_rules.remove(chosen)

                rejected = random.choice(noneco_rules)
                noneco_rules.remove(rejected)

                dpo_samples.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected
                })

        else:
            print(
                "[WARNING] Bad block:",
                len(eco_rules),
                len(neutral_rules),
                len(noneco_rules)
            )


        # Reset
        eco_rules.clear()
        neutral_rules.clear()
        noneco_rules.clear()


print(f"DPO samples: {len(dpo_samples)}")


# ================= SAVE =================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for s in dpo_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")


print(f"Saved: {OUTPUT_FILE}")
