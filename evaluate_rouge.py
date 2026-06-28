"""Compute ROUGE for the fine-tuned T5 summarizer on the SAMSum validation set.

Loads the saved model, generates summaries for a sample of the validation
dialogues, and scores them against the human reference summaries.
"""

import re
import sys

import pandas as pd
import torch
import evaluate
from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_DIR = "content/saved_summary_model"
N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 150
BATCH = 16

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device} | evaluating on {N_SAMPLES} validation samples")


def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip().lower()


tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
model.eval()

val = pd.read_csv("samsum-validation.csv").sample(n=N_SAMPLES, random_state=42).reset_index(drop=True)
val["dialogue"] = val["dialogue"].apply(clean_data)
val["summary"] = val["summary"].apply(clean_data)


def generate_batch(dialogues):
    inputs = tokenizer(list(dialogues), padding=True, truncation=True,
                       max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(input_ids=inputs["input_ids"],
                             attention_mask=inputs["attention_mask"],
                             max_length=150, num_beams=4, early_stopping=True)
    return tokenizer.batch_decode(ids, skip_special_tokens=True)


preds, refs = [], []
dialogues = val["dialogue"].tolist()
summaries = val["summary"].tolist()
for i in range(0, len(dialogues), BATCH):
    preds.extend(generate_batch(dialogues[i:i + BATCH]))
    refs.extend(summaries[i:i + BATCH])
    print(f"  {min(i + BATCH, len(dialogues))}/{len(dialogues)} done")

rouge = evaluate.load("rouge")
results = rouge.compute(predictions=preds, references=refs)

print("\n=== ROUGE ===")
for k, v in results.items():
    print(f"{k}: {v:.4f}")
