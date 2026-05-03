import os
import json
import torch
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import math

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model/checkpoints/finetuned")
VAL_CSV    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/processed/validation.csv")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 200
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print(f"Device: {DEVICE}")
print(f"Loading model from: {CHECKPOINT}")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model     = AutoModelForCausalLM.from_pretrained(CHECKPOINT, dtype=torch.float32).to(DEVICE)
model.eval()

def generate_response(text: str) -> str:
    inp = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            inp,
            max_length=200,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
    return tokenizer.decode(out[:, inp.shape[-1]:][0], skip_special_tokens=True)

def compute_perplexity(text: str) -> float:
    enc = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        loss = model(**enc, labels=enc["input_ids"]).loss
    return math.exp(loss.item())

df = pd.read_csv(VAL_CSV).dropna().head(NUM_SAMPLES)

bleu_scores  = []
perplexities = []
smoother     = SmoothingFunction().method1

print(f"\nEvaluating on {NUM_SAMPLES} samples...\n")

for i, row in df.iterrows():
    inp  = str(row["input"])
    ref  = str(row["response"])

    pred = generate_response(inp)

    ref_tokens  = nltk.word_tokenize(ref.lower())
    pred_tokens = nltk.word_tokenize(pred.lower()) if pred.strip() else ["<empty>"]
    bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoother)
    bleu_scores.append(bleu)

    perplexities.append(compute_perplexity(ref))

    if (i + 1) % 50 == 0:
        print(f"Progress: {i+1}/{NUM_SAMPLES} | "
              f"Avg BLEU so far: {sum(bleu_scores)/len(bleu_scores):.4f} | "
              f"Avg PPL so far: {sum(perplexities)/len(perplexities):.2f}")

avg_bleu = sum(bleu_scores) / len(bleu_scores)
avg_ppl  = sum(perplexities) / len(perplexities)

print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)
print(f"Samples evaluated : {NUM_SAMPLES}")
print(f"Average BLEU score: {avg_bleu:.4f}  (higher is better, 0–1)")
print(f"Average Perplexity: {avg_ppl:.2f}   (lower is better)")
print("="*50)

print("\nSample predictions:")
for _, row in df.head(5).iterrows():
    print(f"\n  Input   : {row['input']}")
    print(f"  Expected: {row['response']}")
    print(f"  Got     : {generate_response(str(row['input']))}")

results = {
    "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "samples_evaluated": NUM_SAMPLES,
    "avg_bleu":         round(avg_bleu, 4),
    "avg_perplexity":   round(avg_ppl,  2),
    "sample_predictions": []
}

for _, row in df.head(10).iterrows():
    pred = generate_response(str(row["input"]))
    results["sample_predictions"].append({
        "input":    row["input"],
        "expected": row["response"],
        "predicted": pred
    })

os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)
report_path = os.path.join(BASE_DIR, "reports/evaluation_results.json")

with open(report_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to: {report_path}")