# 📝 Dialogue Summarizer — Fine-tuned T5

> Fine-tuning T5 to write abstractive summaries of multi-person chat conversations.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-fine--tuning-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-FFD21E)

Fine-tunes **`t5-small`** on the **SAMSum** dataset (messenger-style dialogues paired
with human-written summaries) to produce concise, abstractive summaries — i.e. it
generates new sentences capturing the gist, rather than extracting existing ones.

## 📊 Results (ROUGE on validation set)

| Metric | Score |
|--------|-------|
| ROUGE-1 | **0.444** |
| ROUGE-2 | **0.198** |
| ROUGE-L | **0.373** |

*(evaluated on 150 SAMSum validation dialogues with beam search)*

ROUGE measures n-gram overlap with the human reference summaries (ROUGE-1 = unigrams,
ROUGE-2 = bigrams, ROUGE-L = longest common subsequence). ROUGE-1 ≈ 0.44 is in line
with published `t5-small` baselines on this dataset.

**Example**

> **Input (dialogue):** a multi-turn conversation between reporters and an expert about AI adoption, regulation, bias, and explainability.
>
> **Generated summary:** *"ai adoption has significantly increased over the past few years. experts highlight the importance of responsible ai development, including data privacy, security, and long-term societal impact."*

## 🧠 Method

```
SAMSum dialogues → clean → tokenize (input 512 / target 150) → fine-tune T5 (6 epochs)
   → generate with beam search → evaluate with ROUGE
```

| Step | Choice | Why |
|------|--------|-----|
| Base model | `t5-small` | Encoder-decoder built for text-to-text generation; small enough to fine-tune on a single GPU |
| Cleaning | lowercase, strip whitespace/HTML | Normalize noisy chat text |
| Tokenization | input 512 / target 150, pad-token labels → `-100` | Cap lengths; mask padding so it doesn't distort the loss |
| Training | HuggingFace `Trainer`, 6 epochs, warmup 500, weight decay 0.01 | Standard seq2seq fine-tuning setup |
| Generation | beam search (`num_beams=4`) | Higher-quality summaries than greedy decoding |
| Evaluation | ROUGE-1/2/L | Standard summarization metric vs. human references |

## 🚀 Usage

```bash
pip install -r requirements.txt

# launch the interactive demo (paste a dialogue, get a summary)
python app.py

# evaluate the trained model on the validation set
python evaluate_rouge.py 150     # number of validation samples to score
```

The full training pipeline is in `Text-summarizer.ipynb` (designed for a GPU runtime;
training t5-small for 6 epochs takes ~20 min on a T4).

## 🗂️ Files

```
Text-summarizer.ipynb   full pipeline: data → fine-tune → generate → ROUGE
app.py                   Gradio demo (paste dialogue → summary)
evaluate_rouge.py        standalone ROUGE evaluation on the saved model
samsum-*.csv             SAMSum train / validation / test splits
requirements.txt
```

> The fine-tuned model (~240 MB) is not committed (exceeds GitHub's file limit).
> Run the notebook to reproduce it, or load it locally from `saved_summary_model/`.

## 🔮 Possible improvements

- Try `t5-base` / `bart-large` for higher ROUGE
- Train on the full 14.7k SAMSum train set (this run sampled 4k)
- Add length penalty / no-repeat-ngram constraints to generation
