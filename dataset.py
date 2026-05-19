import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import spacy

#  special token
UNK_TOKEN = "<unk>"   # index 0
PAD_TOKEN = "<pad>"   # index 1
SOS_TOKEN = "<sos>"   # index 2
EOS_TOKEN = "<eos>"   # index 3

SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]

class Vocabulary:
    def __init__(self, min_freq = 2):
        self.min_freq = min_freq
        # stoi: string → index
        self.stoi = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        # itos: index → string
        self.itos = {idx: tok for tok, idx in self.stoi.items()}

    def build_from_sentences(self, tokenized_sentences):
        freq = {}
        for tokens in tokenized_sentences:
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1

        for token, count in sorted(freq.items()):
            if count >= self.min_freq and token not in self.stoi:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx] = token

    def __len__(self):
        return len(self.stoi)

    def lookup_token(self, idx):
        return self.itos.get(idx, UNK_TOKEN)

    def lookup_index(self, token):
        return self.stoi.get(token, self.stoi[UNK_TOKEN])

    @property
    def pad_idx(self):
        return self.stoi[PAD_TOKEN]

    @property
    def sos_idx(self):
        return self.stoi[SOS_TOKEN]

    @property
    def eos_idx(self):
        return self.stoi[EOS_TOKEN]

    @property
    def unk_idx(self):
        return self.stoi[UNK_TOKEN]


#  MULTI30K DATASET CLASS
class Multi30kDataset(Dataset):
    def __init__(self,split = "train", src_vocab = None, tgt_vocab = None, min_freq = 2,):
        self.split = split
        self.min_freq = min_freq

        # Load dataset from HuggingFace
        dataset = load_dataset("bentrevett/multi30k")
        self.raw_data = dataset[split]

        # Load spaCy tokenizers
        # German: de_core_news_sm  |  English: en_core_web_sm
        # Auto-download if not present (handles autograder environments).
        import subprocess, sys
        for model_name in ("de_core_news_sm", "en_core_web_sm"):
            try:
                spacy.load(model_name)
            except OSError:
                print(f"[dataset] spaCy model '{model_name}' not found — downloading …")
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", model_name],
                    check=True,
                )
 
        self.de_nlp = spacy.load("de_core_news_sm")
        self.en_nlp = spacy.load("en_core_web_sm")

        self.src_tokenized = [
            self._tokenize_de(ex["de"]) for ex in self.raw_data
        ]
        self.tgt_tokenized = [
            self._tokenize_en(ex["en"]) for ex in self.raw_data
        ]

        if src_vocab is None:
            self.src_vocab = Vocabulary(min_freq=min_freq)
            self.src_vocab.build_from_sentences(self.src_tokenized)
        else:
            self.src_vocab = src_vocab

        if tgt_vocab is None:
            self.tgt_vocab = Vocabulary(min_freq=min_freq)
            self.tgt_vocab.build_from_sentences(self.tgt_tokenized)
        else:
            self.tgt_vocab = tgt_vocab

        # Convert tokenized sentences to index lists
        self.src_data, self.tgt_data = self.process_data()

    # Tokenizers
    def _tokenize_de(self, text):
        return [tok.text.lower() for tok in self.de_nlp.tokenizer(text)]

    def _tokenize_en(self, text):
        return [tok.text.lower() for tok in self.en_nlp.tokenizer(text)]

    def build_vocab(self):
        self.src_vocab = Vocabulary(min_freq=self.min_freq)
        self.src_vocab.build_from_sentences(self.src_tokenized)

        self.tgt_vocab = Vocabulary(min_freq=self.min_freq)
        self.tgt_vocab.build_from_sentences(self.tgt_tokenized)

    def process_data(self):
        src_data = []
        tgt_data = []

        sos = self.src_vocab.sos_idx  # same index for both (both == 2)
        eos_src = self.src_vocab.eos_idx
        eos_tgt = self.tgt_vocab.eos_idx

        for src_tokens, tgt_tokens in zip(self.src_tokenized, self.tgt_tokenized):
            # Convert tokens → indices, with <sos> and <eos>
            src_indices = (
                [sos]
                + [self.src_vocab.lookup_index(t) for t in src_tokens]
                + [eos_src]
            )
            tgt_indices = (
                [sos]
                + [self.tgt_vocab.lookup_index(t) for t in tgt_tokens]
                + [eos_tgt]
            )
            src_data.append(torch.tensor(src_indices, dtype=torch.long))
            tgt_data.append(torch.tensor(tgt_indices, dtype=torch.long))

        return src_data, tgt_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]

def collate_fn(batch, src_pad_idx = 1, tgt_pad_idx = 1):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=src_pad_idx)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_pad_idx)
    return src_batch, tgt_batch


def build_dataloaders(batch_size = 128, min_freq = 2, num_workers = 0,):
    train_dataset = Multi30kDataset(split="train", min_freq=min_freq)
    src_vocab = train_dataset.src_vocab
    tgt_vocab = train_dataset.tgt_vocab

    # Validation and test reuse the training vocab
    val_dataset  = Multi30kDataset(split="validation", src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    test_dataset = Multi30kDataset(split="test",       src_vocab=src_vocab, tgt_vocab=tgt_vocab)

    pad_idx = src_vocab.pad_idx

    def _collate(batch):
        return collate_fn(batch, src_pad_idx=pad_idx, tgt_pad_idx=pad_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=_collate, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=_collate, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        collate_fn=_collate, num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
