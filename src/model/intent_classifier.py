import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, "data/intent")
REPORTS_DIR  = os.path.join(BASE_DIR, "reports")
TRAIN_CSV    = os.path.join(DATA_DIR, "combined_train.csv")
VAL_CSV      = os.path.join(DATA_DIR, "combined_val.csv")
TEST_CSV     = os.path.join(DATA_DIR, "combined_test.csv")
SAVE_PATH    = os.path.join(BASE_DIR, "model/checkpoints/intent_classifier")
RESULTS_JSON = os.path.join(REPORTS_DIR, "intent_classifier_evaluation.json")

MODEL_NAME = "distilbert-base-uncased"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS     = 3
BATCH_SIZE = 16
LR         = 2e-5
MAX_LENGTH = 128

# Must match preprocessing/load_datasets.py order (stable IDs)
LABEL_ORDER = [
    "inform", "question", "directive", "commissive",
    "greeting", "farewell", "gratitude", "complaint", "confirmation",
]
NUM_LABELS   = len(LABEL_ORDER)
INTENT_TO_ID = {lab: i for i, lab in enumerate(LABEL_ORDER)}
ID_TO_INTENT = {i: lab for i, lab in enumerate(LABEL_ORDER)}

print(f"Training intent classifier on: {DEVICE}")


def save_label_mapping(path_dir: str) -> None:
    payload = {"label_order": LABEL_ORDER}
    os.makedirs(path_dir, exist_ok=True)
    with open(os.path.join(path_dir, "intent_labels.json"), "w") as f:
        json.dump(payload, f, indent=2)


def load_label_order_from_checkpoint(path_dir: str) -> list[str]:
    p = os.path.join(path_dir, "intent_labels.json")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Missing {p}; train the classifier or add intent_labels.json"
        )
    with open(p) as f:
        return list(json.load(f)["label_order"])


def build_id_mapping(label_order: list[str]) -> tuple[dict[int, str], dict[str, int]]:
    return {i: lab for i, lab in enumerate(label_order)}, {
        lab: i for i, lab in enumerate(label_order)
    }


class IntentDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, text_col: str = "text"):
        df = pd.read_csv(csv_path).dropna(subset=["intent", text_col])
        unk = df["intent"].isin(INTENT_TO_ID.keys())
        if not unk.all():
            n = (~unk).sum()
            print(f"  Warning: dropping {n} rows with unknown intents")
            df = df[unk].copy()
        df["label"] = df["intent"].map(INTENT_TO_ID).astype(int)
        self.texts = df[text_col].astype(str).tolist()
        self.labels = df["label"].tolist()
        self.tok = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


def _dataset_labels(csv_path: str, text_col: str = "text") -> np.ndarray:
    df = pd.read_csv(csv_path).dropna(subset=["intent", text_col])
    df = df[df["intent"].isin(INTENT_TO_ID.keys())]
    return df["intent"].map(INTENT_TO_ID).astype(int).values


def class_weights_tensor(csv_path: str) -> torch.Tensor:
    y = _dataset_labels(csv_path)
    cw = compute_class_weight(
        "balanced",
        classes=np.arange(NUM_LABELS),
        y=y,
    )
    return torch.tensor(cw, dtype=torch.float32)


@torch.no_grad()
def evaluate_loader(
    model,
    tokenizer,
    data_loader,
    loss_fn=None,
):
    """Returns loss (if loss_fn given), logits list, labels list."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_logits = []
    all_labels = []

    for batch in data_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
        if loss_fn is not None:
            total_loss += loss_fn(logits, labels).item()
            n_batches += 1
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    cat_logits = torch.cat(all_logits, dim=0)
    cat_labels = torch.cat(all_labels, dim=0)
    avg_loss = (total_loss / n_batches) if n_batches else None
    return avg_loss, cat_logits.numpy(), cat_labels.numpy()


def train():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    for p in (TRAIN_CSV, VAL_CSV):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing {p}. Run src/preprocessing/load_datasets.py first."
            )

    save_label_mapping(SAVE_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    ).to(DEVICE)

    train_ds = IntentDataset(TRAIN_CSV, tokenizer)
    val_ds = IntentDataset(VAL_CSV, tokenizer)
    if len(train_ds) == 0:
        raise ValueError(f"No rows in {TRAIN_CSV}")

    weights = class_weights_tensor(TRAIN_CSV).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = max(len(train_loader) * EPOCHS, 1)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=min(500, total_steps // 10),
        num_training_steps=total_steps,
    )

    best_macro_f1 = -1.0

    print(f"Train batches: {len(train_loader)}, val batches: {len(val_loader)}")
    print(f"Class weights ({LABEL_ORDER[:3]} …): "
          f"{weights[:3].cpu().tolist()}")

    for epoch in range(EPOCHS):
        model.train()
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
            ).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % max(1, len(train_loader) // 8) == 0:
                print(
                    f"Epoch {epoch + 1} | Step {step}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        val_loss, logits_np, labels_np = evaluate_loader(
            model, tokenizer, val_loader, loss_fn=loss_fn
        )
        preds = logits_np.argmax(axis=-1)
        macro_f1 = f1_score(
            labels_np, preds, average="macro", labels=list(range(NUM_LABELS)),
            zero_division=0,
        )

        print(
            f"\nEpoch {epoch + 1} | Val loss: {val_loss:.4f} "
            f"| Macro-F1: {macro_f1:.4f}\n"
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            model.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
            save_label_mapping(SAVE_PATH)
            print(f"✓ Saved best checkpoint (macro-F1: {best_macro_f1:.4f})")

    print(f"\nDone. Best macro-F1 (val): {best_macro_f1:.4f}")
    print(f"Saved to: {SAVE_PATH}")


def load_classifier(save_path=None):
    path = save_path or SAVE_PATH
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE)
    model.eval()
    label_order = load_label_order_from_checkpoint(path)
    id_to_intent, intent_to_id = build_id_mapping(label_order)
    return model, tokenizer, id_to_intent, intent_to_id


def predict_intent(text: str, model, tokenizer, id_to_intent: dict[int, str]) -> str:
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits
    return id_to_intent[logits.argmax(dim=-1).item()]


def evaluate_test_and_save(save_path=None):
    ckpt = save_path or SAVE_PATH
    os.makedirs(REPORTS_DIR, exist_ok=True)
    if not os.path.exists(TEST_CSV):
        print(f"No test CSV at {TEST_CSV}; skipping evaluation.")
        return

    model, tokenizer, id_to_intent, intent_to_id = load_classifier(ckpt)
    test_ds = IntentDataset(TEST_CSV, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False)
    weights = class_weights_tensor(TRAIN_CSV).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    avg_loss, logits_np, labels_np = evaluate_loader(
        model, tokenizer, test_loader, loss_fn=loss_fn,
    )
    preds = logits_np.argmax(axis=-1)
    macro_f1 = f1_score(
        labels_np, preds, average="macro", labels=list(range(NUM_LABELS)),
        zero_division=0,
    )
    acc = float((preds == labels_np).mean())

    target_names = [id_to_intent[i] for i in range(len(id_to_intent))]
    report = classification_report(
        labels_np,
        preds,
        labels=list(range(len(target_names))),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    summary = {
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint":   os.path.abspath(ckpt),
        "train_csv":    TRAIN_CSV,
        "val_csv":      VAL_CSV,
        "test_csv":     TEST_CSV,
        "model_name":   MODEL_NAME,
        "num_labels":   len(target_names),
        "label_order":  LABEL_ORDER,
        "test_examples": len(test_ds),
        "test_loss":    round(float(avg_loss), 6) if avg_loss is not None else None,
        "accuracy":    round(acc, 6),
        "macro_f1":    round(float(macro_f1), 6),
        "weighted_f1": round(
            float(
                f1_score(
                    labels_np, preds,
                    average="weighted",
                    labels=list(range(NUM_LABELS)),
                    zero_division=0,
                )
            ), 6
        ),
        "classification_report": report,
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"✓ Evaluation saved to {RESULTS_JSON}")
    print(f"  Test accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f} | Loss: "
          f"{avg_loss:.4f}")

    cls_lines = classification_report(
        labels_np,
        preds,
        labels=list(range(len(target_names))),
        target_names=target_names,
        zero_division=0,
    )
    print(cls_lines)


if __name__ == "__main__":
    train()
    evaluate_test_and_save()

    print("\nSmoke tests:")
    clf_model, clf_tok, id_map, _ = load_classifier()
    tests = (
        ("what time does the store close", "question"),
        ("thank you so much", "gratitude"),
        ("see you tomorrow", "farewell"),
    )
    for t, hint in tests:
        pred = predict_intent(t.lower(), clf_model, clf_tok, id_map)
        print(f"  '{t}' → {pred}  (hint: {hint})")
