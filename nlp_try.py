from datasets import Dataset
from transformers import pipeline
import pandas as pd

INPUT_CSV = "dataset/raw_dataset.csv"
OUTPUT_CSV = "dataset/green_candidate_subset.csv"

TEXT_COLUMNS = ["title", "desc", "triggerDesc", "actionDesc"]
LABELS = [
    "rule that can be optimized for energy sustainability",
    "rule that is only informational or notification-based"
]
THRESHOLD_KEEP = 0.65

# pipeline
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0,
    batch_size=64
)

# CSV -> HuggingFace Dataset
df = pd.read_csv(INPUT_CSV)
df["full_text"] = df[TEXT_COLUMNS].fillna("").agg(" ".join, axis=1)
ds = Dataset.from_pandas(df)

# Funzione corretta per batch
def classify_batch(batch):
    # batch["full_text"] è sempre lista di stringhe
    texts = [str(t) for t in batch["full_text"]]

    # inizializza keep
    keep_flags = [False] * len(texts)

    # filtra solo testi non vuoti
    non_empty_idx = [i for i, t in enumerate(texts) if t.strip() != ""]
    non_empty_texts = [texts[i] for i in non_empty_idx]

    if len(non_empty_texts) > 0:
        results = classifier(non_empty_texts, batch_size=len(non_empty_texts))
        for idx, res in zip(non_empty_idx, results):
            label = res["labels"][0]
            score = res["scores"][0]
            keep_flags[idx] = label == LABELS[0] and score >= THRESHOLD_KEEP

    batch["keep"] = keep_flags
    return batch

# map in batch
ds = ds.map(classify_batch, batched=True, batch_size=64)

# filter dataset
df_filtered = ds.filter(lambda x: x["keep"]).to_pandas()
df_filtered.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Original dataset: {len(df)}")
print(f"🌱 Filtered eco-friendly dataset: {len(df_filtered)}")
