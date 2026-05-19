
import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(Q,K,V,mask = None,):
    d_k = Q.size(-1)  
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn_w = F.softmax(scores, dim=-1)

    # Replace any NaN that arises when an entire row is -inf
    attn_w = torch.nan_to_num(attn_w, nan=0.0)
    output = torch.matmul(attn_w, V)
    return output, attn_w

def make_src_mask(src, pad_idx = 1,):
    src_mask = (src == pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask


def make_tgt_mask(tgt, pad_idx = 1,):
    batch_size, tgt_len = tgt.size()
    device = tgt.device

    causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device),diagonal=1,)  # shape: [tgt_len, tgt_len]

    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_mask = causal_mask.unsqueeze(0).unsqueeze(0) | pad_mask                           # shape: [batch, 1, tgt_len, tgt_len]
    return tgt_mask

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def _split_heads(self, x):
        batch_size, seq_len, d_model = x.size()                         # Reshape to [batch, seq_len, num_heads, d_k]
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)       # Transpose to [batch, num_heads, seq_len, d_k]
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        batch_size, num_heads, seq_len, d_k = x.size()                  # Transpose back to [batch, seq_len, num_heads, d_k]
        x = x.transpose(1, 2).contiguous()                             # Merge last two dims, finally [batch, seq_len, d_model]
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self,query,key,value,mask = None,):
        Q = self._split_heads(self.W_q(query))  
        K = self._split_heads(self.W_k(key))   
        V = self._split_heads(self.W_v(value)) 
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask=mask)
        attn_output = self._merge_heads(attn_output)
        output = self.W_o(attn_output)
        output = self.dropout(output)

        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)      
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)* (-math.log(10000.0) / d_model)) 

        pe[:, 0::2] = torch.sin(position * div_term)   # even indices
        pe[:, 1::2] = torch.cos(position * div_term)   # odd  indices

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout = 0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(self, x, src_mask):
        residual = x
        x = self.norm1(x)
        x = residual + self.self_attn(x, x, x, mask=src_mask)
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout = 0.1):
        super().__init__()
        self.self_attn   = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn         = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x,memory,src_mask,tgt_mask,):
        residual = x
        x = self.norm1(x)
        x = residual + self.self_attn(x, x, x, mask=tgt_mask)
        residual = x
        x = self.norm2(x)
        x = residual + self.cross_attn(x, memory, memory, mask=src_mask)
        residual = x
        x = self.norm3(x)
        x = residual + self.ffn(x)
        return x

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self,x,memory,src_mask,tgt_mask,):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    _DEFAULT_SRC_VOCAB = 7854  
    _DEFAULT_TGT_VOCAB = 5893  
    _GDRIVE_ID = "YOUR_GDRIVE_FILE_ID_HERE"  

    def __init__(self,src_vocab_size = None,tgt_vocab_size = None,d_model  = 256,N  = 3,num_heads  = 8,d_ff   = 512,dropout = 0.1,max_seq_len = 256,checkpoint_path = "best_model.pt",gdrive_id = "1tvyO62kwO0WLfb2vtrsmpwhAaZgN3oz1",pad_idx = 1,) -> None:
        super().__init__()
        from dataset import build_dataloaders
        import spacy

        # Load tokenizers
        import subprocess, sys
        for model_name in ("de_core_news_sm", "en_core_web_sm"):
            try:
                spacy.load(model_name)
            except OSError:
                print(f"[Transformer] spaCy model '{model_name}' not found — downloading …")
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", model_name],
                    check=True,
                )
 
        self._de_nlp = spacy.load("de_core_news_sm")
        self._en_nlp = spacy.load("en_core_web_sm")

        _, _, _, src_vocab, tgt_vocab = build_dataloaders(batch_size=1)
        self._src_vocab = src_vocab
        self._tgt_vocab = tgt_vocab

        # Resolve vocab sizes
        if src_vocab_size is None:
            src_vocab_size = len(src_vocab)
        if tgt_vocab_size is None:
            tgt_vocab_size = len(tgt_vocab)

        self.pad_idx = pad_idx

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        self.d_model = d_model
        self._emb_scale = math.sqrt(d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)

        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(dec_layer, N)

        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

        # Download and load weights
        _gdrive_id = gdrive_id or self._GDRIVE_ID
        if _gdrive_id and _gdrive_id != "YOUR_GDRIVE_FILE_ID_HERE":
            if not os.path.isfile(checkpoint_path):
                print(f"[Transformer] Downloading weights from Google Drive …")
                gdown.download(
                    id=_gdrive_id,
                    output=checkpoint_path,
                    quiet=False,
                )
            if os.path.isfile(checkpoint_path):
                print(f"[Transformer] Loading weights from {checkpoint_path} …")
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                # Support both bare state-dicts and our save_checkpoint format
                if "model_state_dict" in checkpoint:
                    self.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.load_state_dict(checkpoint)
                print("[Transformer] Weights loaded successfully.")

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def encode(self,src, src_mask,):
        src_emb = self.pos_encoding(self.src_embedding(src) * self._emb_scale)
        memory = self.encoder(src_emb, src_mask)
        return memory

    def decode(self,memory,src_mask,tgt,tgt_mask,):
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt) * self._emb_scale)
        dec_out = self.decoder(tgt_emb, memory, src_mask, tgt_mask)
        logits  = self.output_projection(dec_out)
        return logits

    def forward(self,src,tgt,src_mask,tgt_mask,):
        memory = self.encode(src, src_mask)
        logits = self.decode(memory, src_mask, tgt, tgt_mask)
        return logits

    def infer(self, src_sentence, max_len = 100):
        self.eval()
        device = next(self.parameters()).device

        src_vocab = self._src_vocab
        tgt_vocab = self._tgt_vocab

        # Tokenise German sentence
        if self._de_nlp is not None:
            tokens = [tok.text.lower() for tok in self._de_nlp.tokenizer(src_sentence)]
        else:
            tokens = src_sentence.lower().split()

        src_indices = (
            [src_vocab.sos_idx]
            + [src_vocab.lookup_index(t) for t in tokens]
            + [src_vocab.eos_idx]
        )
        src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)

        src_mask = make_src_mask(src_tensor, pad_idx=self.pad_idx)

        with torch.no_grad():
            # Encode
            memory = self.encode(src_tensor, src_mask)

            # Greedy decod
            sos_idx = tgt_vocab.sos_idx
            eos_idx = tgt_vocab.eos_idx
            ys = torch.tensor([[sos_idx]], dtype=torch.long, device=device)

            for _ in range(max_len):
                tgt_mask = make_tgt_mask(ys, pad_idx=self.pad_idx)
                logits = self.decode(memory, src_mask, ys, tgt_mask)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

                ys = torch.cat([ys, next_token], dim=1)

                if next_token.item() == eos_idx:
                    break

        output_indices = ys.squeeze(0).tolist()[1:]
        output_tokens  = []
        for idx in output_indices:
            if idx == eos_idx:
                break
            tok = tgt_vocab.lookup_token(idx)
            if tok not in ("<sos>", "<eos>", "<pad>", "<unk>"):
                output_tokens.append(tok)

        translation = " ".join(output_tokens)
        for punct in [".", ",", "!", "?", ";", ":", "'", "n't", "'s", "'re", "'ve", "'ll", "'m"]:
            translation = translation.replace(f" {punct}", punct)

        return translation
