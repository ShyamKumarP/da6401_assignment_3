# DA6401 Assignment 3 — Transformer for German→English NMT

Implementation of "Attention Is All You Need" (Vaswani et al., 2017) from scratch using PyTorch.

---

## File Structure

```
da6401_assignment_3/
├── dataset.py        # Multi30k loading, vocab building, DataLoader helpers
├── model.py          # Full Transformer architecture (all components)
├── lr_scheduler.py   # Noam LR scheduler
├── train.py          # Training loop, BLEU eval, checkpointing
├── requirements.txt  # Dependencies
└── README.md
```

---

## Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Download spaCy language models (required for tokenisation)
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

---

## Training

```bash
python train.py
```

This:
1. Loads Multi30k (German→English, 29k/1k/1k train/val/test)  
2. Builds vocabularies from the training split only  
3. Trains the Transformer with the Noam LR schedule and label smoothing  
4. Saves the best checkpoint (`best_model.pt`) by validation loss  
5. Evaluates corpus-level BLEU on the test set  
6. Logs all metrics and curves to W&B  

---

## Inference

```python
from model import Transformer

model = Transformer()   # downloads weights, loads vocab automatically
model.eval()

english = model.infer("Ein Mann läuft durch einen Wald.")
print(english)

`torch`, `numpy`, `matplotlib`, `scikit-learn`, `wandb`, `datasets`, `spacy`, `tqdm`, `evaluate`, `gdown`
