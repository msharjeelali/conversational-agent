from datasets import load_dataset
import json, os

dataset = load_dataset("blended_skill_talk")

os.makedirs("data/raw", exist_ok=True)
for split in ["train", "validation", "test"]:
    records = []
    for item in dataset[split]:
        dialog = list(item["previous_utterance"])
        for free, guided in zip(item["free_messages"], item["guided_messages"]):
            if free.strip():
                dialog.append(free)
            if guided.strip():
                dialog.append(guided)
        
        if len(dialog) >= 2:
            records.append({
                "dialog": dialog,
                "act": [1] * len(dialog)
            })

    with open(f"data/raw/{split}.json", "w") as f:
        json.dump(records, f)
    print(f"{split}: {len(records)} dialogues")