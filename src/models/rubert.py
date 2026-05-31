import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

MODEL_NAME = "sberbank-ai/ruBERT-base"
MAX_LEN = 256
BATCH = 16
EPOCHS = 4
LR = 3e-5
WD = 0.01
WARMUP = 0.1

SEARCH_GRID = {
    "max_len":    [128, 256, 384],
    "batch_size": [8, 16, 32],
    "lr":         [2e-5, 3e-5, 5e-5],
    "epochs":     [3, 4, 5],
}


class BertDataset(Dataset):
    def __init__(self, texts, labels, tok, ml=MAX_LEN):
        self.t, self.l, self.tok, self.ml = texts, labels, tok, ml

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        e = self.tok(self.t[i], truncation=True, padding="max_length",
                     max_length=self.ml, return_tensors="pt")
        return {
            "input_ids": e["input_ids"].squeeze(0),
            "attention_mask": e["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.l[i]), dtype=torch.long),
        }


def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def mk_loader(texts, labels, tok, bs=BATCH, shuf=False, ml=MAX_LEN):
    return DataLoader(BertDataset(texts, labels, tok, ml), bs, shuffle=shuf)


def build_model(device):
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2).to(device)


def train_epoch(model, loader, opt, sched, device):
    model.train()
    tl, n = 0.0, 0
    for b in loader:
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=b["input_ids"].to(device),
                    attention_mask=b["attention_mask"].to(device),
                    labels=b["labels"].to(device))
        out.loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        tl += out.loss.item()
        n += 1
    return tl / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ps, ls = [], []
    for b in loader:
        out = model(input_ids=b["input_ids"].to(device),
                    attention_mask=b["attention_mask"].to(device),
                    labels=b["labels"].to(device))
        ps.extend(torch.argmax(out.logits, 1).cpu().tolist())
        ls.extend(b["labels"].tolist())
    return (
        accuracy_score(ls, ps),
        precision_score(ls, ps, average="macro", zero_division=0),
        recall_score(ls, ps, average="macro", zero_division=0),
        f1_score(ls, ps, average="macro"),
        np.array(ps), np.array(ls),
    )


def fit(model, train_loader, val_loader, device, epochs=EPOCHS, lr=LR):
    opt = AdamW(model.parameters(), lr=lr, weight_decay=WD)
    ts = epochs * len(train_loader)
    sched = get_linear_schedule_with_warmup(opt, int(ts * WARMUP), ts)
    best_f1, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, opt, sched, device)
        a, p, r, f1, _, _ = evaluate(model, val_loader, device)
        print(f"  Ep {ep}: loss={loss:.4f} val_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_f1
