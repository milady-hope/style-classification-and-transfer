from pathlib import Path

import numpy as np
import torch
from sacrebleu import corpus_bleu
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification

from .models import rubert

WEIGHTS = {"cos_sim": 0.5, "style_acc": 0.3, "bleu": 0.2}


def bleu_score(hyps, refs) -> float:
    return corpus_bleu(hyps, [refs]).score / 100.0


def cosine_sim(hyps, refs, sbert=None) -> float:
    if sbert is None:
        sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    h = sbert.encode(hyps)
    r = sbert.encode(refs)
    sims = [float(cosine_similarity([h[i]], [r[i]])[0, 0]) for i in range(len(h))]
    return float(np.mean(sims))


@torch.no_grad()
def style_accuracy(generated, classifier, tokenizer, device,
                   target_label: int = 1, bs: int = 16) -> float:
    classifier.eval()
    preds = []
    for i in range(0, len(generated), bs):
        batch = generated[i:i + bs]
        enc = tokenizer(batch, truncation=True, padding=True,
                        max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = classifier(**enc).logits
        preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
    return float(sum(p == target_label for p in preds) / len(preds))


def load_classifier(checkpoint_path, device):
    model = AutoModelForSequenceClassification.from_pretrained(
        rubert.MODEL_NAME, num_labels=2).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def composite(cos_sim: float, style_acc: float, bleu: float) -> float:
    return (WEIGHTS["cos_sim"] * cos_sim
            + WEIGHTS["style_acc"] * style_acc
            + WEIGHTS["bleu"] * bleu)
