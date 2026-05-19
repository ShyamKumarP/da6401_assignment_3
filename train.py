#DA6401 Assignment 3: "Attention Is All You Need"

import os
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

import wandb
from tqdm import tqdm
import sacrebleu as _sacrebleu   

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import build_dataloaders
from lr_scheduler import NoamScheduler

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, pad_idx, smoothing = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, target):
        smooth_val = self.smoothing / (self.vocab_size - 2) 
        with torch.no_grad():
            dist = torch.full(
                (target.size(0), self.vocab_size),
                fill_value=smooth_val,
                device=logits.device,
            )
            dist.scatter_(1, target.unsqueeze(1), self.confidence)
            dist[:, self.pad_idx] = 0.0
            pad_positions = (target == self.pad_idx)
            dist[pad_positions] = 0.0
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(dist * log_probs).sum(dim=-1)

        loss = loss.masked_fill(pad_positions, 0.0)

        n_tokens = (~pad_positions).sum().clamp(min=1)
        return loss.sum() / n_tokens

def run_epoch(data_iter,model,loss_fn,optimizer,scheduler=None,epoch_num = 0,is_train = True,device = "cpu",):
    model.train(is_train)

    total_loss   = 0.0
    total_tokens = 0
    n_batches    = 0

    desc = f"Epoch {epoch_num} {'Train' if is_train else 'Val'}"
    pbar = tqdm(data_iter, desc=desc, leave=False)

    for batch in pbar:
        src, tgt = batch
        src = src.to(device)   # [batch, src_len]
        tgt = tgt.to(device)   # [batch, tgt_len]

        tgt_inp = tgt[:, :-1]   
        tgt_out = tgt[:, 1:]    

        src_mask = make_src_mask(src, pad_idx=1)
        tgt_mask = make_tgt_mask(tgt_inp, pad_idx=1)

        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_inp, src_mask, tgt_mask)
            batch_size, seq_len, vocab_size = logits.size()
            logits_flat = logits.contiguous().view(-1, vocab_size)
            target_flat = tgt_out.contiguous().view(-1)

            loss = loss_fn(logits_flat, target_flat)

        # Backward pass
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        n_non_pad = (tgt_out != 1).sum().item()
        total_loss   += loss.item() * n_non_pad
        total_tokens += n_non_pad
        n_batches    += 1

        current_lr = (optimizer.param_groups[0]["lr"] if optimizer else 0.0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

    avg_loss = total_loss / max(total_tokens, 1)

    # W&B logging
    if wandb.run is not None:
        log_dict = {
            f"{'train' if is_train else 'val'}/loss":       avg_loss,
            f"{'train' if is_train else 'val'}/perplexity": math.exp(min(avg_loss, 10)),
            "epoch": epoch_num,
        }
        if is_train and optimizer:
            log_dict["train/lr"] = optimizer.param_groups[0]["lr"]
        wandb.log(log_dict)

    return avg_loss

def greedy_decode(model,src,src_mask,max_len,start_symbol,end_symbol,device = "cpu",):
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)

    with torch.no_grad():
        memory = model.encode(src, src_mask)

        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=1).to(device)
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)

            if next_token.item() == end_symbol:
                break

    return ys   

def evaluate_bleu(model,test_dataloader,tgt_vocab,device = "cpu",max_len = 100,):
    model.eval()

    sos_idx = tgt_vocab.sos_idx
    eos_idx = tgt_vocab.eos_idx
    pad_idx = tgt_vocab.pad_idx

    hypotheses = []   # predicted translations (strings)
    refs_list  = []   # reference translations (strings), wrapped in a list

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU eval", leave=False):
            src = src.to(device)
            tgt = tgt.to(device)
            src_mask = make_src_mask(src, pad_idx=1).to(device)

            # Decode each sentence in the batch individually
            for i in range(src.size(0)):
                src_i   = src[i].unsqueeze(0)      
                src_m_i = src_mask[i].unsqueeze(0)  
                tgt_i   = tgt[i]                   

                output = greedy_decode(model, src_i, src_m_i,max_len=max_len,start_symbol=sos_idx,end_symbol=eos_idx,device=device,)
                pred_tokens = []
                for idx in output.squeeze(0).tolist()[1:]:
                    if idx == eos_idx:
                        break
                    tok = tgt_vocab.lookup_token(idx)
                    if tok not in ("<sos>", "<eos>", "<pad>", "<unk>"):
                        pred_tokens.append(tok)
                hypothesis = " ".join(pred_tokens)

                # Convert reference indices → string
                ref_tokens = []
                for idx in tgt_i.tolist():
                    if idx in (sos_idx, pad_idx):
                        continue
                    if idx == eos_idx:
                        break
                    tok = tgt_vocab.lookup_token(idx)
                    ref_tokens.append(tok)
                reference = " ".join(ref_tokens)

                hypotheses.append(hypothesis)
                refs_list.append([reference])

    bleu_obj   = _sacrebleu.corpus_bleu(hypotheses, refs_list)
    bleu_score = bleu_obj.score   

    if wandb.run is not None:
        wandb.log({"test/bleu": bleu_score})

    return bleu_score

#  checkpointz 
def save_checkpoint(model,optimizer,scheduler,epoch,path = "checkpoint.pt",):
    """
    Save model + optimiser + scheduler state to disk.
    """
    model_config = {
        "src_vocab_size": model.src_embedding.num_embeddings,
        "tgt_vocab_size": model.tgt_embedding.num_embeddings,
        "d_model":        model.d_model,
        "N":              len(model.encoder.layers),
        "num_heads":      model.encoder.layers[0].self_attn.num_heads,
        "d_ff":           model.encoder.layers[0].ffn.linear1.out_features,
        "dropout":        model.encoder.layers[0].dropout.p,
        "pad_idx":        model.pad_idx,
    }

    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "model_config":         model_config,
        },
        path,
    )


def load_checkpoint(path,model,optimizer = None,scheduler=None,):
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    return epoch


def run_training_experiment():
    """
        1. Initialise W&B
        2. Build dataset / vocabs
        3. Create DataLoaders
        4. Instantiate Transformer
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss
        8. Training loop with periodic validation and checkpointing
        9. Final BLEU evaluation on the test set
    """
    config = {
        "d_model":       256,
        "N":             3,
        "num_heads":     8,
        "d_ff":          512,
        "dropout":       0.1,
        "label_smoothing": 0.1,
        "batch_size":    128,
        "num_epochs":    30,
        "warmup_steps":  4000,
        "max_seq_len":   256,
        "min_freq":      2,
        "clip_grad":     1.0,
        "base_lr":       1.0,        # Noam scheduler uses scale × base_lr
    }

    wandb.init(project="da6401-a3",name="transformer-base",config=config,)
    cfg = wandb.config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    print("[Train] Building dataset and vocabulary …")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = build_dataloaders(
        batch_size=cfg.batch_size,
        min_freq=cfg.min_freq,
    )
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    pad_idx        = src_vocab.pad_idx
    print(f"[Train] src_vocab={src_vocab_size}  tgt_vocab={tgt_vocab_size}")

    model = Transformer(
        src_vocab_size = src_vocab_size,
        tgt_vocab_size = tgt_vocab_size,
        d_model        = cfg.d_model,
        N              = cfg.N,
        num_heads      = cfg.num_heads,
        d_ff           = cfg.d_ff,
        dropout        = cfg.dropout,
        max_seq_len    = cfg.max_seq_len,
        checkpoint_path= None,       # don't download during training
        gdrive_id      = None,
        pad_idx        = pad_idx,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Trainable parameters: {n_params:,}")
    wandb.config.update({"n_params": n_params})

    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.base_lr,betas=(0.9, 0.98),eps=1e-9,)

    scheduler = NoamScheduler(optimizer,d_model=cfg.d_model,warmup_steps = cfg.warmup_steps,)

    loss_fn = LabelSmoothingLoss(vocab_size = tgt_vocab_size,pad_idx = pad_idx,smoothing = cfg.label_smoothing,)
    best_val_loss = float("inf")
    best_ckpt_path = "best_model.pt"

    for epoch in range(cfg.num_epochs):
        t0 = time.time()

        # Training epoch
        train_loss = run_epoch(train_loader, model, loss_fn,optimizer, scheduler,epoch_num=epoch, is_train=True, device=device)

        # Validation epoch (no gradient updates)
        val_loss = run_epoch(val_loader, model, loss_fn,None, None,epoch_num=epoch, is_train=False, device=device,)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"time={elapsed:.1f}s"
        )

        # Save periodic checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, path=f"ckpt_epoch{epoch}.pt")

        # Save best checkpoint (by validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path=best_ckpt_path)
            print(f"  ↳ New best validation loss: {best_val_loss:.4f} — saved to {best_ckpt_path}")

    print("[Train] Loading best checkpoint for BLEU evaluation …")
    load_checkpoint(best_ckpt_path, model)
    model.eval()

    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"[Train] Test BLEU: {bleu:.2f}")
    wandb.log({"test/bleu": bleu})

    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
