import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
from tqdm import tqdm
import random

# ================= CONFIG =================
INPUT_FILE = "final_with_variants.csv"  # il tuo CSV con colonne: neutral_rule, eco_rule
MODEL_SAVE = "flan_t5_dpo.pt"
PRETRAINED_MODEL = "google/flan-t5-base"
BATCH_SIZE = 8
EPOCHS = 3
LR = 3e-5
MAX_LEN_INPUT = 128
MAX_LEN_OUTPUT = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= LOAD DATA =================
df = pd.read_csv(INPUT_FILE)
df = df.dropna(subset=["neutral_rule", "eco_rule"])


# ================= DATASET =================
class RuleDataset(Dataset):
    def __init__(self, df, tokenizer, max_len_input, max_len_output):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len_input = max_len_input
        self.max_len_output = max_len_output

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        source = self.df.iloc[idx]["neutral_rule"]
        target = self.df.iloc[idx]["eco_rule"]

        source_enc = self.tokenizer(
            source,
            max_length=self.max_len_input,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        target_enc = self.tokenizer(
            target,
            max_length=self.max_len_output,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = target_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100  # ignoriamo padding per la loss

        return {
            "input_ids": source_enc["input_ids"].squeeze(),
            "attention_mask": source_enc["attention_mask"].squeeze(),
            "labels": labels
        }


# ================= TOKENIZER E DATASET =================
tokenizer = T5Tokenizer.from_pretrained(PRETRAINED_MODEL)
dataset = RuleDataset(df, tokenizer, MAX_LEN_INPUT, MAX_LEN_OUTPUT)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ================= MODEL =================
model = T5ForConditionalGeneration.from_pretrained(PRETRAINED_MODEL).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)

# ================= TRAINING =================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=total_loss / (loop.n + 1))

    print(f"Epoch {epoch + 1} finished, avg loss: {total_loss / len(dataloader):.4f}")

# ================= SAVE MODEL =================
torch.save(model.state_dict(), MODEL_SAVE)
print(f"Model saved to {MODEL_SAVE}")

# ================= GENERATION EXAMPLES =================
model.eval()
example_neutral_rules = [
    "IF temperature drops below 20 THEN turn on heater",
    "IF motion is detected THEN turn on lights",
    "IF air quality is poor THEN turn on air purifier",
    "IF humidity rises above 70 THEN turn on dehumidifier",
    "IF door opens THEN send notification",
    "IF window opens THEN close AC",
    "IF presence is detected THEN turn on fan",
    "IF smoke is detected THEN turn off heater",
    "IF water leak is detected THEN shut off valve",
    "IF night time THEN dim lights"
]

print("\n=== Generazione regole eco ===")
for rule in example_neutral_rules:
    input_enc = tokenizer(rule, return_tensors="pt").to(DEVICE)
    generated_ids = model.generate(
        **input_enc,
        max_length=MAX_LEN_OUTPUT,
        num_beams=4,
        early_stopping=True
    )
    generated_rule = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Neutral: {rule}")
    print(f"Generated eco: {generated_rule}\n")
