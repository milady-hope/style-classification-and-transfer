import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          get_linear_schedule_with_warmup)

MODEL_NAME = "ai-forever/ruT5-base"
MAX_LEN_SRC = 512
MAX_LEN_TGT = 512
BATCH = 4
EPOCHS = 8
LR = 3e-5
WD = 0.01
WARMUP = 0.1
LABEL_SMOOTHING = 0.1

PREFIX = "Приведи текст к научному стилю: "

NUM_BEAMS = 4
LENGTH_PENALTY = 0.95
NO_REPEAT_NGRAM = 3
REPETITION_PENALTY = 1.08


class T5Dataset(Dataset):
    def __init__(self, src, tgt, tok, use_prefix=True,
                 max_src=MAX_LEN_SRC, max_tgt=MAX_LEN_TGT):
        self.src, self.tgt, self.tok = src, tgt, tok
        self.use_prefix = use_prefix
        self.max_src, self.max_tgt = max_src, max_tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        s = (PREFIX if self.use_prefix else "") + self.src[i]
        se = self.tok(s, truncation=True, padding="max_length",
                      max_length=self.max_src, return_tensors="pt")
        te = self.tok(self.tgt[i], truncation=True, padding="max_length",
                      max_length=self.max_tgt, return_tensors="pt")
        labels = te.input_ids.squeeze(0)
        labels[labels == self.tok.pad_token_id] = -100
        return {
            "input_ids": se.input_ids.squeeze(0),
            "attention_mask": se.attention_mask.squeeze(0),
            "labels": labels,
        }


def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def build_model(device):
    return AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)


def mk_loader(src, tgt, tok, use_prefix=True, bs=BATCH, shuf=False):
    return DataLoader(T5Dataset(src, tgt, tok, use_prefix), bs, shuffle=shuf)


def train_epoch(model, loader, opt, sched, device):
    model.train()
    tl, n = 0.0, 0
    for b in loader:
        opt.zero_grad(set_to_none=True)
        labels = b["labels"].to(device)
        out = model(input_ids=b["input_ids"].to(device),
                    attention_mask=b["attention_mask"].to(device),
                    labels=labels)
        loss = F.cross_entropy(
            out.logits.view(-1, out.logits.size(-1)),
            labels.view(-1),
            label_smoothing=LABEL_SMOOTHING,
            ignore_index=-100,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        tl += loss.item()
        n += 1
    return tl / max(n, 1)


@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    tl, n = 0.0, 0
    for b in loader:
        labels = b["labels"].to(device)
        out = model(input_ids=b["input_ids"].to(device),
                    attention_mask=b["attention_mask"].to(device),
                    labels=labels)
        loss = F.cross_entropy(
            out.logits.view(-1, out.logits.size(-1)),
            labels.view(-1),
            label_smoothing=LABEL_SMOOTHING,
            ignore_index=-100,
        )
        tl += loss.item()
        n += 1
    return tl / max(n, 1)


@torch.no_grad()
def generate(model, tok, src_texts, device, use_prefix=True, bs=BATCH):
    model.eval()
    out = []
    for i in range(0, len(src_texts), bs):
        batch = src_texts[i:i + bs]
        if use_prefix:
            batch = [PREFIX + t for t in batch]
        enc = tok(batch, truncation=True, padding=True,
                  max_length=MAX_LEN_SRC, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        gen = model.generate(
            **enc,
            max_new_tokens=MAX_LEN_TGT,
            num_beams=NUM_BEAMS,
            length_penalty=LENGTH_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM,
            repetition_penalty=REPETITION_PENALTY,
            early_stopping=True,
        )
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


def fit(model, train_loader, val_loader, device,
        epochs: int = EPOCHS, lr: float = LR):
    opt = AdamW(model.parameters(), lr=lr, weight_decay=WD)
    ts = epochs * len(train_loader)
    sched = get_linear_schedule_with_warmup(opt, int(ts * WARMUP), ts)
    best_loss, best_state = float("inf"), None
    for ep in range(1, epochs + 1):
        tl = train_epoch(model, train_loader, opt, sched, device)
        vl = eval_loss(model, val_loader, device)
        print(f"  Ep {ep}: train_loss={tl:.4f}  val_loss={vl:.4f}")
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_loss
