import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV  = os.path.join(BASE_DIR, "data/processed/train.csv")
VAL_CSV    = os.path.join(BASE_DIR, "data/processed/validation.csv")
SAVE_PATH  = os.path.join(BASE_DIR, "model/checkpoints/intent_classifier")
MODEL_NAME = "distilbert-base-uncased"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS     = 3
BATCH_SIZE = 16
LR         = 2e-5

INTENT_LABELS = {"inform": 0, "question": 1, "directive": 2, "commissive": 3, "dummy": 0}
ID_TO_INTENT  = {0: "inform", 1: "question", 2: "directive", 3: "commissive"}

print(f"Training intent classifier on: {DEVICE}")

class IntentDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        df = pd.read_csv(csv_path).dropna()
        df["label"] = df["intent"].map(INTENT_LABELS).fillna(0).astype(int)
        self.texts  = df["input"].tolist()
        self.labels = df["label"].tolist()
        self.tok    = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx])
        }

def train():
    os.makedirs(SAVE_PATH, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=4
    ).to(DEVICE)

    train_loader = DataLoader(IntentDataset(TRAIN_CSV, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(IntentDataset(VAL_CSV,   tokenizer), batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_scheduler("linear", optimizer=optimizer,
                               num_warmup_steps=50,
                               num_training_steps=len(train_loader) * EPOCHS)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            labels    = batch["label"].to(DEVICE)

            loss = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 200 == 0:
                print(f"Epoch {epoch+1} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attn_mask = batch["attention_mask"].to(DEVICE)
                labels    = batch["label"].to(DEVICE)
                logits    = model(input_ids=input_ids, attention_mask=attn_mask).logits
                preds     = logits.argmax(dim=-1)
                correct  += (preds == labels).sum().item()
                total    += labels.size(0)

        acc = correct / total
        print(f"\nEpoch {epoch+1} complete | Val accuracy: {acc:.4f}\n")

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
            print(f"✓ Saved best classifier (accuracy: {best_acc:.4f})")

    print(f"\nDone. Best accuracy: {best_acc:.4f}")
    print(f"Saved to: {SAVE_PATH}")

def load_classifier():
    tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)
    model     = AutoModelForSequenceClassification.from_pretrained(SAVE_PATH).to(DEVICE)
    model.eval()
    return model, tokenizer

def predict_intent(text: str, model, tokenizer) -> str:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits
    return ID_TO_INTENT[logits.argmax(dim=-1).item()]

if __name__ == "__main__":
    train()

    print("\nTesting classifier:")
    clf_model, clf_tok = load_classifier()
    tests = [
        "what time does the store close",
        "i love spending time with my family",
        "please send me the report by tomorrow",
        "i will get that done for you",
    ]
    for t in tests:
        print(f"  '{t}' → {predict_intent(t, clf_model, clf_tok)}")