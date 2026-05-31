import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, GroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import clean_text, load_data, split_by_pairs
from src.features import (FEATURE_NAMES, ablate_clip, ablate_no_pronouns,
                          ablate_no_punct, extract_features)
from src.metrics import plot_cm
from src.models import baseline, cnn_bilstm, rubert, svm
from src.utils import LABEL_NAMES, SEED, get_device, set_seed, setup_warnings


def save_metrics(out, name, payload):
    with open(out / f"{name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Метрики: {out}/{name}_metrics.json")


def save_mistakes(out, name, test_df, labels, preds):
    mistakes = []
    for i, (t, p) in enumerate(zip(labels, preds)):
        if int(t) != int(p) and len(mistakes) < 10:
            mistakes.append({
                "text": test_df.text.iloc[i],
                "true": int(t),
                "pred": int(p),
                "pair_id": str(test_df.pair_id.iloc[i]),
            })
    with open(out / f"mistakes_{name}.json", "w", encoding="utf-8") as f:
        json.dump(mistakes, f, ensure_ascii=False, indent=2)


def rubert_grid_search_one(train_df, tok, device, config, n_splits=3):
    Xt = train_df.text_clean.tolist()
    yt = train_df.label.to_numpy()
    gt = train_df.pair_id.to_numpy()
    f1s = []
    gkf = GroupKFold(n_splits=n_splits)
    for tri, vai in gkf.split(np.zeros(len(yt)), yt, gt):
        trl = rubert.mk_loader([Xt[i] for i in tri], yt[tri], tok,
                                bs=config["batch_size"], shuf=True,
                                ml=config["max_len"])
        val = rubert.mk_loader([Xt[i] for i in vai], yt[vai], tok,
                                bs=config["batch_size"], ml=config["max_len"])
        m = rubert.build_model(device)
        rubert.fit(m, trl, val, device,
                   epochs=config["epochs"], lr=config["lr"])
        _, _, _, f1, _, _ = rubert.evaluate(m, val, device)
        f1s.append(float(f1))
    return float(np.mean(f1s)), float(np.std(f1s))


def rubert_grid_search(train_df, tok, device):
    base = {"max_len": rubert.MAX_LEN, "batch_size": rubert.BATCH,
            "lr": rubert.LR, "epochs": rubert.EPOCHS}
    print("\nRuBERT grid search. База:", base)
    results = []
    mean_f1, std_f1 = rubert_grid_search_one(train_df, tok, device, base)
    print(f"  base: F1={mean_f1:.4f}±{std_f1:.4f}")
    results.append({"varied": "base", **base,
                    "f1_mean": round(mean_f1, 4), "f1_std": round(std_f1, 4)})
    for param, values in rubert.SEARCH_GRID.items():
        for v in values:
            if v == base[param]:
                continue
            cfg = {**base, param: v}
            mean_f1, std_f1 = rubert_grid_search_one(train_df, tok, device, cfg)
            print(f"  {param}={v}: F1={mean_f1:.4f}±{std_f1:.4f}")
            results.append({"varied": param, **cfg,
                            "f1_mean": round(mean_f1, 4),
                            "f1_std": round(std_f1, 4)})
    return results


def rubert_cv(train_df, tok, device, n_splits=5):
    Xt = train_df.text_clean.tolist()
    yt = train_df.label.to_numpy()
    gt = train_df.pair_id.to_numpy()
    rows = []
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tri, vai) in enumerate(gkf.split(np.zeros(len(yt)), yt, gt), 1):
        print(f"\n  RuBERT fold {fold}")
        trl = rubert.mk_loader([Xt[i] for i in tri], yt[tri], tok, shuf=True)
        val = rubert.mk_loader([Xt[i] for i in vai], yt[vai], tok)
        m = rubert.build_model(device)
        rubert.fit(m, trl, val, device)
        a, p, r, f1, _, _ = rubert.evaluate(m, val, device)
        rows.append({"fold": fold, "acc": float(a), "f1": float(f1)})
    df = pd.DataFrame(rows)
    print(f"  RuBERT CV: Acc={df.acc.mean():.4f}±{df.acc.std():.4f}  "
          f"F1={df.f1.mean():.4f}±{df.f1.std():.4f}")
    return df


def rubert_multi_seed(train_df, test_df, tok, device, seeds=(42, 137, 1337)):
    trl = rubert.mk_loader(train_df.text_clean.tolist(), train_df.label.tolist(),
                            tok, shuf=True)
    tel = rubert.mk_loader(test_df.text_clean.tolist(), test_df.label.tolist(), tok)
    rows = []
    for s in seeds:
        set_seed(s)
        m = rubert.build_model(device)
        rubert.fit(m, trl, tel, device)
        a, p, r, f1, _, _ = rubert.evaluate(m, tel, device)
        rows.append({"seed": int(s), "acc": float(a), "f1": float(f1)})
        print(f"  RuBERT seed={s}: acc={a:.4f}  f1={f1:.4f}")
    set_seed(SEED)
    return pd.DataFrame(rows)


def rubert_shap(model, train_df, test_df, device, out, n=10):
    import shap
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    tok_shap = AutoTokenizer.from_pretrained(rubert.MODEL_NAME, use_fast=False)
    mask = tok_shap.mask_token or "[MASK]"
    masker = shap.maskers.Text(tok_shap, mask_token=mask)

    def predict(texts):
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, np.ndarray):
            texts = texts.tolist()
        texts = [str(t) if t else "" for t in texts]
        model.eval()
        out_arr = []
        with torch.no_grad():
            for i in range(0, len(texts), 8):
                batch = texts[i:i + 8]
                enc = tok_shap(batch, truncation=True, padding=True,
                               max_length=256, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits
                out_arr.append(F.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(out_arr)

    explainer = shap.Explainer(predict, masker, algorithm="partition")
    sorted_test = test_df.copy()
    sorted_test["len"] = sorted_test.text_clean.apply(len)
    sorted_test = sorted_test.sort_values("len")
    sci = sorted_test[sorted_test.label == 1].head(n // 2)
    pop = sorted_test[sorted_test.label == 0].head(n // 2)
    explain_df = pd.concat([pop, sci]).reset_index(drop=True)
    shap_values = explainer(explain_df.text_clean.tolist())

    token_map = {}
    for i in range(len(explain_df)):
        sv_i = shap_values[i, :, 1]
        tokens = tok_shap.tokenize(explain_df.text_clean.iloc[i])
        n_tok = min(len(tokens), len(sv_i.values) - 2)
        for j in range(n_tok):
            token_map.setdefault(tokens[j], []).append(float(sv_i.values[j + 1]))
    token_means = {t: float(np.mean(v)) for t, v in token_map.items() if len(v) >= 3}
    top_neg = sorted(token_means.items(), key=lambda x: x[1])[:10]
    top_pos = sorted(token_means.items(), key=lambda x: x[1], reverse=True)[:10]
    return {"top_neg": top_neg, "top_pos": top_pos}


def run_rubert(train_df, test_df, device, out, args):
    print("\n" + "=" * 60 + "\nRuBERT\n" + "=" * 60)
    tok = rubert.get_tokenizer()

    grid_results = rubert_grid_search(train_df, tok, device) if args.grid_search else []
    cv_df = rubert_cv(train_df, tok, device)

    trl = rubert.mk_loader(train_df.text_clean.tolist(), train_df.label.tolist(),
                            tok, shuf=True)
    tel = rubert.mk_loader(test_df.text_clean.tolist(), test_df.label.tolist(), tok)
    model = rubert.build_model(device)
    rubert.fit(model, trl, tel, device)
    a, p, r, f1, preds, labs = rubert.evaluate(model, tel, device)
    print(f"\nRuBERT (тест) — Acc: {a:.3f}  P: {p:.3f}  R: {r:.3f}  F1: {f1:.3f}")
    print(classification_report(labs, preds, digits=4,
                                target_names=[LABEL_NAMES[0], LABEL_NAMES[1]]))
    plot_cm(labs, preds, "RuBERT (тест)", str(out / "figures" / "cm_rubert.png"))

    torch.save(model.state_dict(), out / "checkpoints" / "rubert.pt")

    ms_df = rubert_multi_seed(train_df, test_df, tok, device) if args.multi_seed else pd.DataFrame()
    shap_top = rubert_shap(model, train_df, test_df, device, out) if args.shap else {}

    save_metrics(out, "rubert", {
        "test": {"acc": float(a), "precision": float(p),
                 "recall": float(r), "f1": float(f1)},
        "cv": cv_df.to_dict(orient="records"),
        "multi_seed": ms_df.to_dict(orient="records") if len(ms_df) else [],
        "grid_search": grid_results,
        "shap": shap_top,
    })
    save_mistakes(out, "rubert", test_df, labs, preds)


def cnn_grid_search_one(train_df, device, config, n_splits=3):
    f1s = []
    gkf = GroupKFold(n_splits=n_splits)
    for tri, vai in gkf.split(np.zeros(len(train_df)), train_df.label, train_df.pair_id):
        trf = train_df.iloc[tri].reset_index(drop=True)
        vaf = train_df.iloc[vai].reset_index(drop=True)
        vocab = cnn_bilstm.build_vocab(trf.text.tolist())
        trl = cnn_bilstm.make_loader(trf, vocab, shuf=True)
        val = cnn_bilstm.make_loader(vaf, vocab)
        m = cnn_bilstm.CharCNNBiLSTM(
            len(vocab),
            ed=config["emb_dim"], nf=config["filters"],
            hd=config["hidden"], dr=config["dropout"],
        ).to(device)
        cnn_bilstm.fit(m, trl, val, device,
                       epochs=config["epochs"], lr=config["lr"])
        _, _, _, f1, _, _ = cnn_bilstm.evaluate(m, val, device)
        f1s.append(float(f1))
    return float(np.mean(f1s)), float(np.std(f1s))


def cnn_grid_search(train_df, device):
    base = {"emb_dim": cnn_bilstm.EMB_DIM, "filters": cnn_bilstm.FILTERS,
            "hidden": cnn_bilstm.HIDDEN, "dropout": cnn_bilstm.DROP,
            "lr": cnn_bilstm.LR, "epochs": cnn_bilstm.EPOCHS}
    print("\nCNN grid search. База:", base)
    results = []
    mean_f1, std_f1 = cnn_grid_search_one(train_df, device, base)
    print(f"  base: F1={mean_f1:.4f}±{std_f1:.4f}")
    results.append({"varied": "base", **base,
                    "f1_mean": round(mean_f1, 4), "f1_std": round(std_f1, 4)})
    for param, values in cnn_bilstm.SEARCH_GRID.items():
        for v in values:
            if v == base[param]:
                continue
            cfg = {**base, param: v}
            mean_f1, std_f1 = cnn_grid_search_one(train_df, device, cfg)
            print(f"  {param}={v}: F1={mean_f1:.4f}±{std_f1:.4f}")
            results.append({"varied": param, **cfg,
                            "f1_mean": round(mean_f1, 4),
                            "f1_std": round(std_f1, 4)})
    return results


def cnn_cv(train_df, device, n_splits=5):
    rows = []
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tri, vai) in enumerate(
        gkf.split(np.zeros(len(train_df)), train_df.label, train_df.pair_id), 1
    ):
        print(f"\n  CNN fold {fold}")
        trf = train_df.iloc[tri].reset_index(drop=True)
        vaf = train_df.iloc[vai].reset_index(drop=True)
        vocab = cnn_bilstm.build_vocab(trf.text.tolist())
        trl = cnn_bilstm.make_loader(trf, vocab, shuf=True)
        val = cnn_bilstm.make_loader(vaf, vocab)
        m = cnn_bilstm.CharCNNBiLSTM(len(vocab)).to(device)
        cnn_bilstm.fit(m, trl, val, device)
        a, p, r, f1, _, _ = cnn_bilstm.evaluate(m, val, device)
        rows.append({"fold": fold, "acc": float(a), "f1": float(f1)})
    df = pd.DataFrame(rows)
    print(f"  CNN CV: Acc={df.acc.mean():.4f}±{df.acc.std():.4f}  "
          f"F1={df.f1.mean():.4f}±{df.f1.std():.4f}")
    return df


def cnn_multi_seed(train_df, test_df, device, seeds=(42, 137, 1337)):
    rows = []
    for s in seeds:
        set_seed(s)
        vocab = cnn_bilstm.build_vocab(train_df.text.tolist())
        trl = cnn_bilstm.make_loader(train_df, vocab, shuf=True)
        tel = cnn_bilstm.make_loader(test_df, vocab)
        m = cnn_bilstm.CharCNNBiLSTM(len(vocab)).to(device)
        cnn_bilstm.fit(m, trl, tel, device, patience=5)
        a, p, r, f1, _, _ = cnn_bilstm.evaluate(m, tel, device)
        rows.append({"seed": int(s), "acc": float(a), "f1": float(f1)})
        print(f"  CNN seed={s}: acc={a:.4f}  f1={f1:.4f}")
    set_seed(SEED)
    return pd.DataFrame(rows)


def cnn_shap(model, vocab, train_df, test_df, device, n_bg=50, n_expl=20):
    import shap
    from src.interpret import CNNEmbeddingWrapper, texts_to_tensor

    set_seed(SEED)
    bg_idx = np.random.choice(len(train_df), n_bg, replace=False)
    bg_texts = train_df.text_clean.iloc[bg_idx].tolist()
    expl_texts = test_df.text_clean.iloc[:n_expl].tolist()

    bg_tensor = texts_to_tensor(bg_texts, vocab, cnn_bilstm.MAX_LEN, device=device)
    expl_tensor = texts_to_tensor(expl_texts, vocab, cnn_bilstm.MAX_LEN, device=device)

    wrapped = CNNEmbeddingWrapper(model)
    with torch.no_grad():
        bg_embs = model.emb(bg_tensor)
        expl_embs = model.emb(expl_tensor)

    explainer = shap.GradientExplainer(wrapped, bg_embs)
    shap_vals = explainer.shap_values(expl_embs)
    model.eval()
    sv = (shap_vals[1] if isinstance(shap_vals, list) else shap_vals).sum(axis=-1)

    char_sums, char_cnts = {}, {}
    for i in range(n_expl):
        chars = list(clean_text(expl_texts[i]).lower())[:cnn_bilstm.MAX_LEN]
        for j, ch in enumerate(chars):
            if j < sv.shape[1]:
                char_sums[ch] = char_sums.get(ch, 0.0) + float(sv[i, j])
                char_cnts[ch] = char_cnts.get(ch, 0) + 1

    char_mean = {ch: char_sums[ch] / char_cnts[ch]
                 for ch in char_sums if char_cnts[ch] >= 20}
    sorted_chars = sorted(char_mean.items(), key=lambda x: x[1])
    return {"top_neg": sorted_chars[:10], "top_pos": sorted_chars[-10:][::-1]}


def run_cnn(train_df, test_df, device, out, args):
    print("\n" + "=" * 60 + "\nChar-CNN-BiLSTM\n" + "=" * 60)

    grid_results = cnn_grid_search(train_df, device) if args.grid_search else []
    cv_df = cnn_cv(train_df, device)

    vocab = cnn_bilstm.build_vocab(train_df.text.tolist())
    trl = cnn_bilstm.make_loader(train_df, vocab, shuf=True)
    tel = cnn_bilstm.make_loader(test_df, vocab)
    model = cnn_bilstm.CharCNNBiLSTM(len(vocab)).to(device)
    cnn_bilstm.fit(model, trl, tel, device, patience=5)
    a, p, r, f1, preds, labs = cnn_bilstm.evaluate(model, tel, device)
    print(f"\nCNN (тест) — Acc: {a:.3f}  P: {p:.3f}  R: {r:.3f}  F1: {f1:.3f}")
    print(classification_report(labs, preds, digits=4,
                                target_names=[LABEL_NAMES[0], LABEL_NAMES[1]]))
    plot_cm(labs, preds, "Char-CNN-BiLSTM (тест)",
            str(out / "figures" / "cm_cnn.png"))

    torch.save({"state_dict": model.state_dict(), "vocab": vocab},
               out / "checkpoints" / "cnn_bilstm.pt")

    ms_df = cnn_multi_seed(train_df, test_df, device) if args.multi_seed else pd.DataFrame()
    shap_top = cnn_shap(model, vocab, train_df, test_df, device) if args.shap else {}

    save_metrics(out, "cnn_bilstm", {
        "test": {"acc": float(a), "precision": float(p),
                 "recall": float(r), "f1": float(f1)},
        "cv": cv_df.to_dict(orient="records"),
        "multi_seed": ms_df.to_dict(orient="records") if len(ms_df) else [],
        "grid_search": grid_results,
        "shap": shap_top,
    })
    save_mistakes(out, "cnn_bilstm", test_df, labs, preds)


def svm_shap(model, train_df, test_df, out):
    import shap
    vct = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    X_train = vct.transform(train_df.text_clean)
    X_test = vct.transform(test_df.text_clean)
    feat_names = np.array(vct.get_feature_names_out())

    explainer = shap.LinearExplainer(clf, X_train)
    shap_vals = explainer.shap_values(X_test)
    X_dense = X_test.toarray()
    masked = shap_vals.copy()
    masked[X_dense == 0] = 0

    plt.figure(figsize=(10, 8))
    shap.summary_plot(masked, X_test, feature_names=feat_names,
                      max_display=20, show=False)
    plt.title("SHAP Summary — SVM")
    plt.tight_layout()
    plt.savefig(out / "figures" / "shap_svm_summary.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    mean_shap = shap_vals.mean(axis=0)
    top_neg = [(str(feat_names[i]), float(mean_shap[i]))
               for i in np.argsort(mean_shap)[:10]]
    top_pos = [(str(feat_names[i]), float(mean_shap[i]))
               for i in np.argsort(mean_shap)[-10:][::-1]]
    return {"top_neg": top_neg, "top_pos": top_pos}


def run_svm(train_df, test_df, device, out, args):
    print("\n" + "=" * 60 + "\nSVM\n" + "=" * 60)

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(train_df.text_clean, train_df.label, train_df.pair_id))
    pipe = svm.build_pipeline()
    grid = GridSearchCV(pipe, svm.PARAM_GRID, cv=splits,
                        scoring="f1_macro", refit=True, n_jobs=-1, verbose=1)
    grid.fit(train_df.text_clean, train_df.label)
    print(f"Best params: {grid.best_params_}  Best CV F1: {grid.best_score_:.4f}")
    model = grid.best_estimator_

    cv_rows = []
    for fold, (tr, va) in enumerate(splits, 1):
        model.fit(train_df.text_clean.iloc[tr], train_df.label.iloc[tr])
        pred = model.predict(train_df.text_clean.iloc[va])
        cv_rows.append({
            "fold": fold,
            "acc": float(accuracy_score(train_df.label.iloc[va], pred)),
            "f1": float(f1_score(train_df.label.iloc[va], pred, average="macro")),
        })
    cv_df = pd.DataFrame(cv_rows)
    print(f"SVM CV: Acc={cv_df.acc.mean():.4f}±{cv_df.acc.std():.4f}  "
          f"F1={cv_df.f1.mean():.4f}±{cv_df.f1.std():.4f}")

    model.fit(train_df.text_clean, train_df.label)
    preds = model.predict(test_df.text_clean)
    a = accuracy_score(test_df.label, preds)
    p = precision_score(test_df.label, preds, average="macro", zero_division=0)
    r = recall_score(test_df.label, preds, average="macro", zero_division=0)
    f1 = f1_score(test_df.label, preds, average="macro")
    print(f"\nSVM (тест) — Acc: {a:.3f}  P: {p:.3f}  R: {r:.3f}  F1: {f1:.3f}")
    print(classification_report(test_df.label, preds, digits=4,
                                target_names=[LABEL_NAMES[0], LABEL_NAMES[1]]))
    plot_cm(test_df.label, preds, "SVM (тест)",
            str(out / "figures" / "cm_svm.png"))

    joblib.dump(model, out / "checkpoints" / "svm.joblib")

    shap_top = svm_shap(model, train_df, test_df, out) if args.shap else {}

    save_metrics(out, "svm", {
        "test": {"acc": float(a), "precision": float(p),
                 "recall": float(r), "f1": float(f1)},
        "cv": cv_df.to_dict(orient="records"),
        "grid_search": {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in grid.best_params_.items()},
        "shap": shap_top,
    })
    save_mistakes(out, "svm", test_df, test_df.label.to_numpy(), preds)


def baseline_coefficients(model, out):
    coefs = model.named_steps["lr"].coef_[0]
    df = pd.DataFrame({"feature": FEATURE_NAMES, "coef": coefs})
    df["abs"] = df.coef.abs()
    df = df.sort_values("coef")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d73027" if v < 0 else "#4575b4" for v in df.coef]
    ax.barh(df.feature, df.coef, color=colors)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Коэффициент")
    ax.set_title("Коэффициенты LogReg")
    plt.tight_layout()
    plt.savefig(out / "figures" / "baseline_coefs.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    return df.sort_values("abs", ascending=False).to_dict(orient="records")


def baseline_ablations(model, train_df, test_df, splits, y_train, y_test):
    transforms = {
        "Все признаки":            lambda s: s.astype(str),
        "Без местоимений":         lambda s: s.astype(str).apply(ablate_no_pronouns),
        "Без пунктуации":          lambda s: s.astype(str).apply(ablate_no_punct),
        "Урезание до 200 токенов": lambda s: s.astype(str).apply(ablate_clip),
    }
    rows = []
    for name, t in transforms.items():
        X_tr = extract_features(t(train_df.text_clean))
        X_te = extract_features(t(test_df.text_clean))
        cv_f1s = []
        for tr, va in splits:
            model.fit(X_tr[tr], y_train[tr])
            cv_f1s.append(f1_score(y_train[va], model.predict(X_tr[va]),
                                   average="macro"))
        model.fit(X_tr, y_train)
        yp = model.predict(X_te)
        rows.append({
            "Конфигурация": name,
            "CV F1": f"{np.mean(cv_f1s):.3f}±{np.std(cv_f1s):.3f}",
            "Accuracy": f"{accuracy_score(y_test, yp):.3f}",
            "Precision": f"{precision_score(y_test, yp, average='macro', zero_division=0):.3f}",
            "Recall": f"{recall_score(y_test, yp, average='macro', zero_division=0):.3f}",
            "F1 (тест)": f"{f1_score(y_test, yp, average='macro'):.3f}",
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df.to_dict(orient="records")


def run_baseline(train_df, test_df, device, out, args):
    print("\n" + "=" * 60 + "\nBaseline (LogReg)\n" + "=" * 60)

    X_train = extract_features(train_df.text_clean)
    y_train = train_df.label.values
    groups = train_df.pair_id.values
    X_test = extract_features(test_df.text_clean)
    y_test = test_df.label.values

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X_train, y_train, groups))
    grid = GridSearchCV(baseline.build_pipeline(), baseline.PARAM_GRID,
                        cv=splits, scoring="f1_macro", refit=True, n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}  Best CV F1: {grid.best_score_:.4f}")
    model = grid.best_estimator_

    cv_rows = []
    for fold, (tr, va) in enumerate(splits, 1):
        model.fit(X_train[tr], y_train[tr])
        pred = model.predict(X_train[va])
        cv_rows.append({
            "fold": fold,
            "acc": float(accuracy_score(y_train[va], pred)),
            "f1": float(f1_score(y_train[va], pred, average="macro")),
        })
    cv_df = pd.DataFrame(cv_rows)
    print(f"Baseline CV: Acc={cv_df.acc.mean():.4f}±{cv_df.acc.std():.4f}  "
          f"F1={cv_df.f1.mean():.4f}±{cv_df.f1.std():.4f}")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    a = accuracy_score(y_test, preds)
    p = precision_score(y_test, preds, average="macro", zero_division=0)
    r = recall_score(y_test, preds, average="macro", zero_division=0)
    f1 = f1_score(y_test, preds, average="macro")
    print(f"\nBaseline (тест) — Acc: {a:.3f}  P: {p:.3f}  R: {r:.3f}  F1: {f1:.3f}")
    print(classification_report(y_test, preds, digits=4,
                                target_names=[LABEL_NAMES[0], LABEL_NAMES[1]]))
    plot_cm(y_test, preds, "Baseline (LogReg) — тест",
            str(out / "figures" / "cm_baseline.png"))

    joblib.dump(model, out / "checkpoints" / "baseline.joblib")

    coefs = baseline_coefficients(model, out)
    print("\nТаблица 2. Абляции baseline-модели:")
    abl = baseline_ablations(model, train_df, test_df, splits, y_train, y_test)

    save_metrics(out, "baseline", {
        "test": {"acc": float(a), "precision": float(p),
                 "recall": float(r), "f1": float(f1)},
        "cv": cv_df.to_dict(orient="records"),
        "grid_search": grid.best_params_,
        "coefficients": coefs,
        "ablations": abl,
    })
    save_mistakes(out, "baseline", test_df, y_test, preds)


def main(args):
    setup_warnings()
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    out = Path(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    data = load_data(args.data)
    train_df, test_df = split_by_pairs(data, test_size=0.2, seed=SEED)
    print(f"Train: {len(train_df)}  Test: {len(test_df)}")

    runners = {
        "rubert":   run_rubert,
        "cnn":      run_cnn,
        "svm":      run_svm,
        "baseline": run_baseline,
    }
    for name in args.models:
        if name in runners:
            runners[name](train_df, test_df, device, out, args)
        else:
            print(f"[skip] Неизвестная модель: {name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data500.csv")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--models", nargs="+",
                   default=["rubert", "cnn", "svm", "baseline"])
    p.add_argument("--grid-search", action="store_true")
    p.add_argument("--multi-seed", action="store_true")
    p.add_argument("--shap", action="store_true")
    main(p.parse_args())