import pandas as pd
import random
import json


# ===========================
# CONFIG
# ===========================

ECO_FILE = "ultra_mega_final_eco.csv"
NEUTRAL_FILE = "ultra_mega_final_neutral.csv"
NONECO_FILE = "ultra_mega_final_non_eco.csv"

OUTPUT_FILE = "dpo_dataset.jsonl"

NUM_SAMPLES = 20000

SEED = 42


# ===========================
# SET SEED
# ===========================

random.seed(SEED)


# ===========================
# LOAD DATA
# ===========================

print("Loading datasets...")

eco_df = pd.read_csv(ECO_FILE)
neutral_df = pd.read_csv(NEUTRAL_FILE)
noneco_df = pd.read_csv(NONECO_FILE)


# ===========================
# CLEAN
# ===========================

def clean_df(df):

    df = df.dropna(subset=["triggerDesc", "actionDesc"])

    df["triggerDesc"] = df["triggerDesc"].astype(str)
    df["actionDesc"] = df["actionDesc"].astype(str)

    return df


eco_df = clean_df(eco_df)
neutral_df = clean_df(neutral_df)
noneco_df = clean_df(noneco_df)


print("Samples loaded:")
print("ECO:", len(eco_df))
print("NEUTRAL:", len(neutral_df))
print("NON-ECO:", len(noneco_df))


# ===========================
# BUILD RULE
# ===========================

def build_rule(row):

    trigger = row["triggerDesc"].strip()
    action = row["actionDesc"].strip()

    return f"{trigger}{action}"


eco_rules = eco_df.apply(build_rule, axis=1).tolist()
neutral_rules = neutral_df.apply(build_rule, axis=1).tolist()
noneco_rules = noneco_df.apply(build_rule, axis=1).tolist()


# ===========================
# GENERATE DPO
# ===========================

print("Generating DPO pairs...")

dpo_data = []

for i in range(NUM_SAMPLES):

    prompt = random.choice(neutral_rules)
    chosen = random.choice(eco_rules)
    rejected = random.choice(noneco_rules)

    sample = {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }

    dpo_data.append(sample)

    if i % 1000 == 0 and i > 0:
        print(f"{i} samples generated...")


# ===========================
# SAVE
# ===========================

print("Saving file...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for item in dpo_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


print("Done!")
print(f"Saved {len(dpo_data)} samples to {OUTPUT_FILE}")
