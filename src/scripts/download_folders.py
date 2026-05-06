import gdown
import os

MODELS = {
    "model/checkpoints/finetuned": "1tgVn0gWmkLDjzS8OFby1rBqIu_rDpdgt",
    "model/checkpoints/intent_classifier": "1Vel-VtcxTGZNq-FBEJdZLUABk9Ut7tP-",
}

for path, folder_id in MODELS.items():
    os.makedirs(path, exist_ok=True)
    print(f"Downloading to {path}...")
    gdown.download_folder(
        id=folder_id,
        output=path,
        quiet=False
    )
    print(f"✓ Done: {path}")

print("\nAll models downloaded. You can now run the API.")