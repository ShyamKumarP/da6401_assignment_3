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
```

**Before submission**: update `Transformer._GDRIVE_ID` in `model.py` with the file-ID of your trained weights on Google Drive.

---

## Architecture Details

### Design Choice — Pre-LayerNorm

We use **Pre-LayerNorm** (norm applied *before* each sub-layer) rather than Post-LN (the original paper). Pre-LN:
- Produces more stable gradients at initialisation  
- Reduces sensitivity to the initial learning rate during warmup  
- Reference: Xiong et al., "On Layer Normalization in the Transformer Architecture" (2020)  

The final LayerNorm after each stack (encoder/decoder) completes the Pre-LN convention.

### Hyperparameters (default)

| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| N (layers) | 3 |
| num_heads | 8 |
| d_ff | 512 |
| dropout | 0.1 |
| label_smoothing ε | 0.1 |
| warmup_steps | 4000 |
| optimizer | Adam (β₁=0.9, β₂=0.98, ε=1e-9) |
| batch_size | 128 |
| epochs | 30 |

> Note: These are resource-constrained defaults suitable for Multi30k. The original paper uses d_model=512, N=6 for larger datasets.

---

## Autograder Contract

All required signatures are preserved exactly:

| Symbol | File | Signature |
|--------|------|-----------|
| `scaled_dot_product_attention` | model.py | `(Q, K, V, mask) → (out, weights)` |
| `MultiHeadAttention.forward` | model.py | `(q, k, v, mask) → Tensor` |
| `PositionalEncoding.forward` | model.py | `(x) → Tensor` |
| `make_src_mask` | model.py | `(src, pad_idx) → BoolTensor [B,1,1,SL]` |
| `make_tgt_mask` | model.py | `(tgt, pad_idx) → BoolTensor [B,1,TL,TL]` |
| `Transformer.encode` | model.py | `(src, src_mask) → Tensor` |
| `Transformer.decode` | model.py | `(memory, src_mask, tgt, tgt_mask) → Tensor` |
| `Transformer.infer` | model.py | `(german_str) → english_str` |
| `greedy_decode` | train.py | `(model, src, src_mask, max_len, start_symbol, end_symbol, device) → Tensor` |
| `evaluate_bleu` | train.py | `(model, test_dl, tgt_vocab, device, max_len) → float` |
| `save_checkpoint` | train.py | `(model, optimizer, scheduler, epoch, path) → None` |
| `load_checkpoint` | train.py | `(path, model, optimizer, scheduler) → int` |

---

## W&B Experiments (Section 2)

The following ablations should be run and reported:

1. **Noam vs Fixed LR** — overlay train loss & val accuracy curves
2. **Scaling factor √(1/dₖ)** — log gradient norms of Q/K weights for first 1000 steps
3. **Attention Rollout & Head Specialisation** — heatmaps per head in last encoder layer
4. **Sinusoidal PE vs Learned Embeddings** — compare val BLEU, discuss extrapolation
5. **Label Smoothing ε=0.1 vs ε=0.0** — plot softmax confidence of correct token

---

## Permitted Libraries

`torch`, `numpy`, `matplotlib`, `scikit-learn`, `wandb`, `datasets`, `spacy`, `tqdm`, `evaluate`, `gdown`
