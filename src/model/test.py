from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model/checkpoints/finetuned")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

model     = AutoModelForCausalLM.from_pretrained(CHECKPOINT, dtype=torch.float32).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

def chat(text):
    inp = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt").to(DEVICE)
    out = model.generate(
        inp,
        max_length=200,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )
    response = tokenizer.decode(out[:, inp.shape[-1]:][0], skip_special_tokens=True)
    return response

tests = [
    "hello how are you",
    "what do you like to do for fun",
    "do you have any kids",
]

for t in tests:
    print(f"You: {t}")
    print(f"Bot: {chat(t)}")
    print()