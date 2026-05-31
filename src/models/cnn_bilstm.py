from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader, Dataset

from ..data import clean_text

PAD_ID, UNK_ID = 0, 1
MAX_VOCAB = 500
MAX_LEN = 1500
BATCH = 32
EMB_DIM = 64
FILTERS = 128
KERNELS = [3, 5, 7]
HIDDEN = 256
DROP = 0.3
EPOCHS = 20
LR = 3e-4
THR = 0.40

SEARCH_GRID = {
    "emb_dim": [32, 64, 128],
    "filters": [64, 128, 256],
    "hidden":  [128, 256, 512],
    "dropout": [0.2, 0.3, 0.5],
    "lr":      [1e-4, 3e-4, 5e-4],
    "epochs":  [15, 20, 25],
}


def tok_chars(text: str):
    return list(clean_text(text).lower())


def build_vocab(texts, mx: int = MAX_VOCAB, mf: int = 1):
    c = Counter()
    for t in texts:
        c.update(tok_chars(t))
    v = {"<PAD>": 0, "<UNK>": 1}
    for ch, fr in c.most_common(mx - 2):
        if fr >= mf:
            v[ch] = len(v)
    return v


def to_ids(text: str, vocab, ml: int = MAX_LEN):
    ids = [vocab.get(c, 1) for c in tok_chars(text)][:ml]
    ln = len(ids)
    return ids + [0] * (ml - ln), min(ln, ml)


class CNNDataset(Dataset):
    def __init__(self, df, vocab, ml=MAX_LEN):
        self.t = df["text"].astype(str).tolist()
        self.l = df["label"].astype(int).tolist()
        self.v, self.ml = vocab, ml

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        ids, ln = to_ids(self.t[i], self.v, self.ml)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "lengths": torch.tensor(ln, dtype=torch.long),
            "labels": torch.tensor(self.l[i], dtype=torch.long),
        }


class CharCNNBiLSTM(nn.Module):
    def __init__(self, vs, ed=EMB_DIM, nf=FILTERS, ks=KERNELS, hd=HIDDEN, dr=DROP):
        super().__init__()
        self.emb = nn.Embedding(vs, ed, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(ed, nf, k, padding=k // 2),
                nn.BatchNorm1d(nf), nn.ReLU(),
                nn.MaxPool1d(4, 4),
            ) for k in ks
        ])
        self.lstm = nn.LSTM(nf * len(ks), hd, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hd * 2, 1)
        self.drop = nn.Dropout(dr)
        self.fc = nn.Linear(hd * 2, 2)

    def forward(self, x, lengths):
        x = self.emb(x).transpose(1, 2)
        outs = [c(x) for c in self.convs]
        ml = min(o.size(2) for o in outs)
        x = torch.cat([o[:, :, :ml] for o in outs], 1).transpose(1, 2)
        x, _ = self.lstm(x)
        w = torch.softmax(self.attn(x).squeeze(-1), 1)
        x = torch.bmm(w.unsqueeze(1), x).squeeze(1)
        return self.fc(self.drop(x))


def make_loader(df, vocab, bs=BATCH, shuf=False, ml=MAX_LEN):
    return DataLoader(CNNDataset(df, vocab, ml), bs, shuffle=shuf)


@torch.no_grad()
def evaluate(model, loader, device, thr=THR):
    model.eval()
    ps, ts = [], []
    for b in loader:
        logits = model(b["input_ids"].to(device), b["lengths"].to(device))
        probs = torch.softmax(logits, -1)[:, 1]
        ps.extend((probs >= thr).long().cpu().tolist())
        ts.extend(b["labels"].tolist())
    return (
        accuracy_score(ts, ps),
        precision_score(ts, ps, average="macro", zero_division=0),
        recall_score(ts, ps, average="macro", zero_division=0),
        f1_score(ts, ps, average="macro"),
        ps, ts,
    )


def fit(model, train_loader, val_loader, device,
        epochs: int = EPOCHS, patience: int = 4, lr: float = LR):
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, "max", 0.5, 2)
    crit = nn.CrossEntropyLoss()
    best_f1, best_state, wait = 0.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        tl, n = 0.0, 0
        for b in train_loader:
            opt.zero_grad()
            loss = crit(model(b["input_ids"].to(device), b["lengths"].to(device)),
                        b["labels"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(b["labels"])
            n += len(b["labels"])
        a, p, r, f1, _, _ = evaluate(model, val_loader, device)
        sched.step(f1)
        print(f"  Ep {ep}: loss={tl/n:.4f} val_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stop ep {ep}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_f1
