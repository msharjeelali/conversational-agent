import spacy
import nltk
import json
import re
import pandas as pd
from nltk.corpus import stopwords


nltk.download("stopwords", quiet=True)
nlp = spacy.load("en_core_web_sm")
STOPWORDS = set(stopwords.words("english"))


INTENT_MAP = {
    0: "dummy",
    1: "inform",
    2: "question",
    3: "directive",
    4: "commissive"
}


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s\?\!\.,']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_entities(text: str) -> list:
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def build_pairs(dialogue: list, intents: list) -> list:
    pairs = []
    for i in range(len(dialogue) - 1):
        inp  = clean_text(dialogue[i])
        resp = clean_text(dialogue[i + 1])
        intent = INTENT_MAP.get(intents[i] if i < len(intents) else 1, "inform")
        if len(inp) > 5 and len(resp) > 5:
            pairs.append({
                "input": inp,
                "response": resp,
                "intent": intent,
                "entities": str(extract_entities(inp))
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
    import os
    os.makedirs("data/processed", exist_ok=True)
    for split in ["train", "validation", "test"]:
        df = process_split(f"data/raw/{split}.json")
        df.to_csv(f"data/processed/{split}.csv", index=False)
        print(f"{split}: {len(df)} pairs — saved to data/processed/{split}.csv")
    
    df = pd.read_csv("data/processed/train.csv")
    print("\nSample row:")
    print(df.iloc[0])