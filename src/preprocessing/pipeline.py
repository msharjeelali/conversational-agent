import spacy
import nltk
import json
import re
import os
import pandas as pd
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

nlp       = spacy.load("en_core_web_sm")
STOPWORDS = set(stopwords.words("english"))

DIALOG_INTENT_MAP = {
    0: None,        # dummy — skip
    1: "inform",
    2: "question",
    3: "directive",
    4: "commissive"
}

def refine_intent(text: str, base_intent: str) -> str:
    t = text.lower().strip()
    if any(t.startswith(g) for g in
           ["hi ", "hello", "hey ", "hi!", "hello!", "hey!", "hi,"]):
        return "greeting"
    if any(t.startswith(f) for f in
           ["bye", "goodbye", "see you", "take care", "farewell"]):
        return "farewell"
    if any(w in t for w in ["thank you", "thanks", "thank u", "thx"]):
        return "gratitude"
    if any(t.startswith(c) for c in
           ["yes,", "yes.", "yes!", "yeah", "correct,", "exactly", "right,"]):
        return "confirmation"
    return base_intent

def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s\?\!\.,']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def extract_entities(text: str) -> list:
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def build_pairs(dialogue: list, intents: list) -> list:
    pairs = []
    for i in range(len(dialogue) - 1):
        inp    = clean_text(dialogue[i])
        resp   = clean_text(dialogue[i + 1])
        intent = DIALOG_INTENT_MAP.get(
            intents[i] if i < len(intents) else 1
        )

        if not intent:
            continue
        if len(inp) <= 5 or len(resp) <= 5:
            continue

        intent   = refine_intent(inp, intent)
        entities = str(extract_entities(inp))

        pairs.append({
            "input":    inp,
            "response": resp,
            "intent":   intent,
            "entities": entities
        })
    return pairs

def process_split(path: str) -> pd.DataFrame:
    with open(path) as f:
        raw = json.load(f)
    all_pairs = []
    for item in raw:
        all_pairs.extend(build_pairs(item["dialog"], item["act"]))
    return pd.DataFrame(all_pairs)

if __name__ == "__main__":
    BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir       = os.path.join(BASE_DIR, "data/raw")
    processed_dir = os.path.join(BASE_DIR, "data/processed")
    os.makedirs(processed_dir, exist_ok=True)

    for split in ["train", "validation", "test"]:
        path = os.path.join(raw_dir, f"{split}.json")
        if not os.path.exists(path):
            print(f"Skipping {split} — {path} not found")
            continue

        df  = process_split(path)
        out = os.path.join(processed_dir, f"{split}.csv")
        df.to_csv(out, index=False)

        print(f"\n{split}: {len(df):,} pairs")
        print(f"  Intent distribution:")
        for intent, count in df["intent"].value_counts().items():
            pct = count / len(df) * 100
            print(f"    {intent:<14}: {count:>6,}  ({pct:.1f}%)")

    df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    print("\nSample row:")
    print(df.iloc[0].to_string())