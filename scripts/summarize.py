import argparse
import json
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import plot_cv_folds


MODEL_ORDER = [
    ("RuBERT",          "rubert_metrics.json"),
    ("Char-CNN-BiLSTM", "cnn_bilstm_metrics.json"),
    ("SVM",             "svm_metrics.json"),
    ("Baseline",        "baseline_metrics.json"),
]


def main(args):
    out = Path(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    loaded = {}
    for name, fn in MODEL_ORDER:
        path = out / fn
        if not path.exists():
            print(f"[!] Не найдено: {path} (пропуск)")
            continue
        with open(path, encoding="utf-8") as f:
            loaded[name] = json.load(f)

    if not loaded:
        print("Нет ни одного файла метрик в", out)
        return

    rows = []
    for name in [n for n, _ in MODEL_ORDER if n in loaded]:
        t = loaded[name]["test"]
        rows.append({
            "Модель":     name,
            "Accuracy":   t["acc"],
            "Precision":  t["precision"],
            "Recall":     t["recall"],
            "F1 (macro)": t["f1"],
        })
    summary = pd.DataFrame(rows)
    print("Таблица 3. Результаты классификации на тестовой выборке\n")
    print(summary.to_string(index=False, float_format="%.3f"))
    summary.to_csv(out / "summary.csv", index=False)

    all_cv = {
        name: pd.DataFrame(loaded[name]["cv"])
        for name, _ in MODEL_ORDER
        if name in loaded and loaded[name].get("cv")
    }
    if all_cv:
        plot_cv_folds(all_cv, str(out / "figures" / "cv_folds_all_models.png"))

    aggregate = {
        "test": summary.to_dict(orient="records"),
        "cv": {name: df.to_dict(orient="records") for name, df in all_cv.items()},
        "multi_seed": {
            name: loaded[name].get("multi_seed", [])
            for name in loaded if loaded[name].get("multi_seed")
        },
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)
    print(f"\nСводка: {out}/summary.csv  и  {out}/metrics.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="results")
    main(p.parse_args())
