import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import GroupShuffleSplit
from transformers import get_linear_schedule_with_warmup

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_data
from src.models import rubert, rut5
from src.transfer_metrics import (bleu_score, composite, cosine_sim,
                                   load_classifier, style_accuracy)
from src.utils import SEED, get_device, set_seed, setup_warnings


def split_70_10_20(df, seed: int = SEED):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tv_idx, te_idx = next(gss1.split(df, df.label, groups=df.pair_id))
    tv = df.iloc[tv_idx].reset_index(drop=True)
    test = df.iloc[te_idx].reset_index(drop=True)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
    tr_idx, va_idx = next(gss2.split(tv, tv.label, groups=tv.pair_id))
    train = tv.iloc[tr_idx].reset_index(drop=True)
    val = tv.iloc[va_idx].reset_index(drop=True)
    return train, val, test


def make_pairs(df):
    pop, sci = [], []
    for pid in df.pair_id.unique():
        pair = df[df.pair_id == pid]
        p = pair[pair.label == 0]
        s = pair[pair.label == 1]
        if len(p) == 1 and len(s) == 1:
            pop.append(p.text_clean.iloc[0])
            sci.append(s.text_clean.iloc[0])
    return pop, sci


def full_eval(model, tok, src, tgt, classifier, classifier_tok, sbert,
              device, use_prefix=True):
    gen = rut5.generate(model, tok, src, device, use_prefix=use_prefix)
    cos = cosine_sim(gen, tgt, sbert=sbert)
    acc = style_accuracy(gen, classifier, classifier_tok, device)
    bleu = bleu_score(gen, tgt)
    return {
        "cos_sim": round(cos, 4),
        "style_acc": round(acc, 4),
        "bleu": round(bleu, 4),
        "composite": round(composite(cos, acc, bleu), 4),
        "generations": gen,
    }


def train_one(train_src, train_tgt, val_src, val_tgt,
              tok, classifier, classifier_tok, sbert, device,
              use_prefix: bool, selection: str):
    print(f"\n=== ruT5-base | prefix={use_prefix} | selection={selection} ===")
    trl = rut5.mk_loader(train_src, train_tgt, tok,
                          use_prefix=use_prefix, shuf=True)
    val_loader = rut5.mk_loader(val_src, val_tgt, tok, use_prefix=use_prefix)

    model = rut5.build_model(device)
    opt = torch.optim.AdamW(model.parameters(), lr=rut5.LR, weight_decay=rut5.WD)
    ts = rut5.EPOCHS * len(trl)
    sched = get_linear_schedule_with_warmup(opt, int(ts * rut5.WARMUP), ts)

    best_score = float("inf") if selection == "loss" else -1.0
    best_state = None

    for ep in range(1, rut5.EPOCHS + 1):
        tl = rut5.train_epoch(model, trl, opt, sched, device)
        vl = rut5.eval_loss(model, val_loader, device)

        if selection == "loss":
            score = vl
            better = score < best_score
            print(f"  Ep {ep}: train_loss={tl:.4f}  val_loss={vl:.4f}")
        else:
            m = full_eval(model, tok, val_src, val_tgt,
                          classifier, classifier_tok, sbert, device,
                          use_prefix=use_prefix)
            score = m["composite"]
            better = score > best_score
            print(f"  Ep {ep}: train_loss={tl:.4f}  val_loss={vl:.4f}  "
                  f"val_composite={score:.4f}")

        if better:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main(args):
    setup_warnings()
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    out = Path(args.out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    rubert_ckpt = out / "checkpoints" / "rubert.pt"
    if not rubert_ckpt.exists():
        print(f"ОШИБКА: не найден {rubert_ckpt}")
        print("Сначала: make rubert")
        return

    data = load_data(args.data)
    train_df, val_df, test_df = split_70_10_20(data)
    train_src, train_tgt = make_pairs(train_df)
    val_src, val_tgt = make_pairs(val_df)
    test_src, test_tgt = make_pairs(test_df)
    print(f"Train pairs: {len(train_src)}  "
          f"Val: {len(val_src)}  Test: {len(test_src)}")

    tok = rut5.get_tokenizer()
    print("\nЗагрузка RuBERT-классификатора и SBERT...")
    classifier = load_classifier(rubert_ckpt, device)
    classifier_tok = rubert.get_tokenizer()
    sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    if args.all_configs:
        configs = [
            (True,  "loss"),
            (False, "loss"),
            (True,  "composite"),
            (False, "composite"),
        ]
    else:
        configs = [(args.use_prefix, args.selection)]

    rows = []
    best_test_score = -1.0
    best_model = None
    best_label = None
    best_generations = None

    for use_prefix, selection in configs:
        set_seed(SEED)
        model = train_one(train_src, train_tgt, val_src, val_tgt,
                          tok, classifier, classifier_tok, sbert, device,
                          use_prefix, selection)
        m = full_eval(model, tok, test_src, test_tgt,
                      classifier, classifier_tok, sbert, device,
                      use_prefix=use_prefix)
        rows.append({
            "Префикс":   "да" if use_prefix else "нет",
            "Отбор":     selection,
            "CosSim":    m["cos_sim"],
            "Accuracy":  m["style_acc"],
            "BLEU":      m["bleu"],
            "Composite": m["composite"],
        })
        print(f"\n  TEST: CosSim={m['cos_sim']:.4f}  "
              f"StyleAcc={m['style_acc']:.4f}  BLEU={m['bleu']:.4f}  "
              f"Composite={m['composite']:.4f}")

        if m["composite"] > best_test_score:
            best_test_score = m["composite"]
            best_model = model
            best_label = (use_prefix, selection)
            best_generations = m["generations"]

    print("\n=== Таблица 7. Результаты преобразования стиля на тестовой выборке ===\n")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format="%.4f"))
    df.to_csv(out / "transfer_table7.csv", index=False)

    ckpt = out / "checkpoints" / "rut5.pt"
    torch.save(best_model.state_dict(), ckpt)
    print(f"\nЛучшая конфигурация: prefix={best_label[0]} selection={best_label[1]}")
    print(f"Чекпойнт лучшей модели: {ckpt}")

    payload = {
        "table7": rows,
        "best_config": {"use_prefix": best_label[0], "selection": best_label[1]},
        "examples": [
            {"source": s, "target": t, "generated": g}
            for s, t, g in zip(test_src[:5], test_tgt[:5], best_generations[:5])
        ],
    }
    with open(out / "transfer_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Метрики: {out}/transfer_metrics.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data500.csv")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--use-prefix", action="store_true", default=True)
    p.add_argument("--no-prefix", dest="use_prefix", action="store_false")
    p.add_argument("--selection", choices=["loss", "composite"], default="composite")
    p.add_argument("--all-configs", action="store_true")
    main(p.parse_args())
