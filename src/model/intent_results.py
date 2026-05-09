import os
import json
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_PATH  = os.path.join(BASE_DIR, "model/checkpoints/intent_classifier")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

ID_TO_INTENT  = {0: "inform", 1: "question", 2: "directive", 3: "commissive"}
INTENT_LABELS = {"inform": 0, "question": 1, "directive": 2, "commissive": 3}

tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(SAVE_PATH).to(DEVICE)
model.eval()

def predict(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits
    return ID_TO_INTENT[logits.argmax(dim=-1).item()]

tests = [
    ("what time does the store close",          "question"),
    ("do you have any siblings",                "question"),
    ("where are you from",                      "question"),
    ("how old are you",                         "question"),
    ("what is your name",                       "question"),
    ("who are you",                             "question"),
    ("what do you like to do for fun",          "question"),
    ("have you ever been abroad",               "question"),
    ("what happened next",                      "question"),
    ("how does that work",                      "question"),
    ("i love spending time with my family",     "inform"),
    ("i have two kids",                         "inform"),
    ("i work as a teacher",                     "inform"),
    ("i live in lahore",                        "inform"),
    ("i enjoy reading books",                   "inform"),
    ("i studied computer science",              "inform"),
    ("i like to cook at home",                  "inform"),
    ("i moved here last year",                  "inform"),
    ("i enjoy watching movies",                 "inform"),
    ("i have a dog named max",                  "inform"),
    ("please send me the report by tomorrow",   "directive"),
    ("can you help me with this",               "directive"),
    ("stop doing that",                         "directive"),
    ("tell me what happened",                   "directive"),
    ("remind me tomorrow",                      "directive"),
    ("show me how to do it",                    "directive"),
    ("help me understand this",                 "directive"),
    ("make sure you finish it today",           "directive"),
    ("close the door please",                   "directive"),
    ("send it over when ready",                 "directive"),
    ("i will get that done for you",            "commissive"),
    ("i promise to call you",                   "commissive"),
    ("i am going to finish it today",           "commissive"),
    ("i will make sure it happens",             "commissive"),
    ("i plan to visit next week",               "commissive"),
    ("i will take care of it",                  "commissive"),
    ("i intend to fix this",                    "commissive"),
    ("i will be there on time",                 "commissive"),
    ("i promise to do better",                  "commissive"),
    ("i am going to finish this project today", "commissive"),
]

predictions = []
actuals     = []
results     = []

print(f"\n{'Input':<50} {'Expected':<12} {'Predicted':<12} {'Match'}")
print("-" * 80)

for text, expected in tests:
    predicted = predict(text)
    match     = "✓" if predicted == expected else "✗"
    predictions.append(INTENT_LABELS[predicted])
    actuals.append(INTENT_LABELS[expected])
    results.append({
        "input":     text,
        "expected":  expected,
        "predicted": predicted,
        "correct":   predicted == expected
    })
    print(f"{text:<50} {expected:<12} {predicted:<12} {match}")

print("\n" + "="*50)
print("CLASSIFICATION REPORT — Current Classifier")
print("="*50)
report_str  = classification_report(
    actuals, predictions,
    target_names=["inform", "question", "directive", "commissive"]
)
report_dict = classification_report(
    actuals, predictions,
    target_names=["inform", "question", "directive", "commissive"],
    output_dict=True
)
print(report_str)

os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)
output = {
    "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "classifier_version": "v1_blended_skill_talk",
    "dataset_used":       "blended_skill_talk (imbalanced — 99% inform)",
    "total_tests":        len(tests),
    "correct":            sum(1 for r in results if r["correct"]),
    "accuracy":           round(sum(1 for r in results if r["correct"]) / len(tests), 4),
    "classification_report": report_dict,
    "per_sample_results": results
}

path = os.path.join(BASE_DIR, "reports/intent_classifier_v2_results.json")
with open(path, "w") as f:
    json.dump(output, f, indent=2)

print(f"✓ Results saved to: reports/intent_classifier_v2_results.json")
print(f"Overall accuracy: {output['accuracy']*100:.1f}%")