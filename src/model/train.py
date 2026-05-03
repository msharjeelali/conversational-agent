# src/model/train.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from torch.optim import AdamW
import os

MODEL_NAME  = "microsoft/DialoGPT-small"
SAVE_PATH   = "model/checkpoints/dialo_finetuned"
TRAIN_CSV   = "data/train.csv"
VAL_CSV     = "data/validation.csv"
MAX_LEN     = 128
BATCH_SIZE  = 8
EPOCHS      = 3
LR          = 1e-5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Training on: {DEVICE}")

class DialogDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len):
        df = pd.read_csv(csv_path).dropna().head(5000)
        self.pairs     = list(zip(df["input"], df["response"]))
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inp, resp = self.pairs[idx]

        inp_ids  = self.tokenizer.encode(inp  + self.tokenizer.eos_token)
        resp_ids = self.tokenizer.encode(resp + self.tokenizer.eos_token)

        max_inp_len = self.max_len // 2
        inp_ids = inp_ids[:max_inp_len]

        combined = inp_ids + resp_ids
        combined = combined[:self.max_len]

        pad_len    = self.max_len - len(combined)
        input_ids  = combined + [self.tokenizer.eos_token_id] * pad_len

        labels = [-100] * len(inp_ids) + combined[len(inp_ids):] + [-100] * pad_len

        assert any(l != -100 for l in labels), "All labels masked — check tokenizer"

        return {
            "input_ids": torch.tensor(input_ids),
            "labels":    torch.tensor(labels)
        }

def train():
    os.makedirs(SAVE_PATH, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(DEVICE)
    scaler = torch.amp.GradScaler('cuda') 

    train_dataset = DialogDataset(TRAIN_CSV, tokenizer, MAX_LEN)
    val_dataset   = DialogDataset(VAL_CSV,   tokenizer, MAX_LEN)

    sample = train_dataset[0]
    labels = sample["labels"].tolist()
    non_masked = [l for l in labels if l != -100]
    print(f"Sanity check — non-masked label tokens in sample 0: {len(non_masked)}")
    assert len(non_masked) > 0, "Labels are all -100, something is wrong!"

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LR, eps=1e-8, weight_decay=0.01)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * EPOCHS
    )

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            labels    = batch["labels"].to(DEVICE)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, labels=labels)
                loss    = outputs.loss

            if torch.isnan(loss):
                print(f"NaN loss at step {step}, skipping batch")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            if step % 100 == 0:
                print(f"Epoch {epoch+1} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                labels    = batch["labels"].to(DEVICE)
                outputs   = model(input_ids=input_ids, labels=labels)
                if not torch.isnan(outputs.loss):
                    val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"\nEpoch {epoch+1} complete | Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f}\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
            print(f"✓ Saved best checkpoint (val loss: {best_val_loss:.4f})")

    print("\nTraining complete.")
    print(f"Best model saved to: {SAVE_PATH}")

if __name__ == "__main__":
    train()