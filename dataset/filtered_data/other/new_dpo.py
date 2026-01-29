import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import random

# ================= CONFIG =================
INPUT_FILE = "final_with_variants.csv"
MODEL_SAVE = "dpo_gen_model.pt"
PRETRAINED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 hidden size
HIDDEN_DIM = 256
VOCAB_SIZE = 30522  # tokenizer vocab size

# ================= LOAD DATA =================
df = pd.read_csv(INPUT_FILE)

# ================= TOKENIZER =================
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)


# ================= GENERATE PAIRS =================
def generate_pairs(df):
    pairs = []
    for _, row in df.iterrows():
        neutral = row["neutral_rule"]
        eco = row["eco_rule"]
        pairs.append((neutral, eco))
    return pairs


pairs = generate_pairs(df)
random.shuffle(pairs)


# ================= DATASET =================
class GenerationDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        neutral, eco = self.pairs[idx]
        neutral_enc = tokenizer(neutral, truncation=True, padding="max_length", max_length=self.max_len,
                                return_tensors="pt")
        eco_enc = tokenizer(eco, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return {
            "neutral_input_ids": neutral_enc["input_ids"].squeeze(0),
            "neutral_attention_mask": neutral_enc["attention_mask"].squeeze(0),
            "eco_input_ids": eco_enc["input_ids"].squeeze(0)
        }


dataset = GenerationDataset(pairs, tokenizer, MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# ================= MODEL =================
class EncoderDecoder(nn.Module):
    def __init__(self, encoder_model, embedding_dim, hidden_dim, vocab_size):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model)
        self.decoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.output_linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask, target_ids=None):
        # Encoder
        with torch.no_grad():  # congeliamo BERT
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            encoder_emb = encoder_outputs.last_hidden_state[:, 0, :]  # CLS token
        # Replichiamo embedding per seq length
        seq_len = target_ids.shape[1] if target_ids is not None else MAX_LEN
        decoder_input = encoder_emb.unsqueeze(1).repeat(1, seq_len, 1)
        # Decoder
        decoder_outputs, _ = self.decoder(decoder_input)
        logits = self.output_linear(decoder_outputs)
        return logits


model = EncoderDecoder(PRETRAINED_MODEL, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# ================= TRAINING =================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["neutral_input_ids"].to(DEVICE)
        attention_mask = batch["neutral_attention_mask"].to(DEVICE)
        target_ids = batch["eco_input_ids"].to(DEVICE)

        logits = model(input_ids, attention_mask, target_ids)
        loss = loss_fn(logits.view(-1, VOCAB_SIZE), target_ids.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), MODEL_SAVE)
print(f"Generative DPO model saved to {MODEL_SAVE}")


# ================= GENERATION =================
def generate_eco(neutral_rule, top_k=5):
    model.eval()
    with torch.no_grad():
        enc = tokenizer(neutral_rule, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        logits = model(input_ids, attention_mask)  # [1, seq_len, vocab_size]
        # argmax token per token
        token_ids = torch.argmax(logits, dim=-1)[0]
        generated = tokenizer.decode(token_ids, skip_special_tokens=True)
    return generated


# ================= ESEMPI =================
neutral_examples = [
    "IF temperature drops below 20 THEN turn on heater",
    "IF motion is detected THEN turn on lights",
    "IF air quality is poor THEN turn on air purifier",
    "IF humidity rises above 70 THEN turn on dehumidifier",
    "IF door opens THEN send notification"
]

for nr in neutral_examples:
    eco_gen = generate_eco(nr)
    print(f"\nNeutral: {nr}")
    print(f"Generated eco: {eco_gen}")
