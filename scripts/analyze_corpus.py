import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_data
from src.utils import LABEL_NAMES, setup_warnings

_TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё0-9'-]+")


def main(args):
    setup_warnings()
    out = Path(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    data = load_data(args.data)
    data["len_words"] = data["text"].apply(lambda s: len(_TOKEN.findall(str(s))))
    data["len_chars"] = data["text"].apply(lambda s: len(str(s)))
    data["n_sent"] = data["text"].apply(
        lambda s: max(len(re.split(r"[.!?…]+", str(s))) - 1, 1)
    )
    data["avg_sent_len"] = data["len_words"] / data["n_sent"]

    stats = data.groupby("label").agg(
        n_texts=("text", "count"),
        total_sents=("n_sent", "sum"),
        mean_chars=("len_chars", "mean"),
        mean_words=("len_words", "mean"),
        median_words=("len_words", "median"),
        mean_sent_len=("avg_sent_len", "mean"),
    ).round(2)
    stats.index = [LABEL_NAMES[i] for i in stats.index]

    print("Таблица 1. Характеристики корпуса\n")
    print(stats.to_string())
    stats.to_csv(out / "corpus_stats.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for lbl, name in LABEL_NAMES.items():
        axes[0].hist(data[data.label == lbl].len_words, bins=25, alpha=0.55, label=name)
    axes[0].set_title("Распределение длины текстов (слова)")
    axes[0].set_xlabel("Длина"); axes[0].set_ylabel("Кол-во"); axes[0].legend()
    axes[0].grid(True, alpha=0.25)
    axes[1].boxplot(
        [data[data.label == 0].len_words, data[data.label == 1].len_words],
        labels=[LABEL_NAMES[0], LABEL_NAMES[1]], showfliers=False,
    )
    axes[1].set_title("Boxplot длин по классам")
    axes[1].set_ylabel("Длина (слова)"); axes[1].grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "figures" / "corpus_lengths.png", dpi=150, bbox_inches="tight")
    plt.close()

    sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    pair_sims = []
    for pid in data.pair_id.unique():
        pair = data[data.pair_id == pid]
        if len(pair) == 2:
            e = sbert.encode(pair.text.tolist())
            pair_sims.append(float(cosine_similarity([e[0]], [e[1]])[0, 0]))

    print(f"\nКосинусная близость пар: mean={np.mean(pair_sims):.3f}, "
          f"median={np.median(pair_sims):.3f}")

    plt.figure(figsize=(6, 3))
    plt.hist(pair_sims, bins=30, edgecolor="white", alpha=0.8)
    plt.axvline(np.mean(pair_sims), color="red", ls="--",
                label=f"mean={np.mean(pair_sims):.3f}")
    plt.title("Семантическая близость пар (cosine sim)")
    plt.xlabel("Cosine similarity"); plt.ylabel("Кол-во пар")
    plt.legend(); plt.tight_layout()
    plt.savefig(out / "figures" / "pair_similarity.png", dpi=150, bbox_inches="tight")
    plt.close()

    metrics = {
        "stats": {k: list(v.values()) for k, v in stats.to_dict().items()},
        "pair_similarity": {
            "mean": float(np.mean(pair_sims)),
            "median": float(np.median(pair_sims)),
            "n_pairs": len(pair_sims),
        },
    }
    with open(out / "corpus_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nГотово. Артефакты: {out}/corpus_stats.csv, "
          f"{out}/corpus_metrics.json, {out}/figures/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data500.csv")
    p.add_argument("--out-dir", default="results")
    main(p.parse_args())
