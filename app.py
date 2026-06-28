"""Gradio demo for the fine-tuned T5 dialogue summarizer.

Paste a conversation, get an abstractive summary. Loads the locally saved model.

Run:  python app.py
"""

import re

import gradio as gr
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_DIR = "content/saved_summary_model"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
model.eval()


def clean(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip().lower()


def summarize(dialogue):
    if not dialogue.strip():
        return "Please paste a conversation to summarize."
    inputs = tokenizer(clean(dialogue), padding="max_length", max_length=512,
                       truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(input_ids=inputs["input_ids"],
                             attention_mask=inputs["attention_mask"],
                             max_length=150, num_beams=4, early_stopping=True)
    return tokenizer.decode(ids[0], skip_special_tokens=True)


example = (
    "Amanda: I baked cookies. Do you want some?\n"
    "Jerry: Sure!\n"
    "Amanda: I'll bring you tomorrow :-)"
)

demo = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(lines=12, label="Conversation", placeholder="Paste a dialogue…"),
    outputs=gr.Textbox(lines=4, label="Summary"),
    title="📝 Dialogue Summarizer (fine-tuned T5)",
    description="Paste a multi-person conversation and get a concise abstractive summary.",
    examples=[[example]],
)

if __name__ == "__main__":
    demo.launch()
