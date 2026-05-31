import re

import nltk
import numpy as np
import pandas as pd

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords as _nltk_stopwords

RUS_STOPWORDS = set(_nltk_stopwords.words("russian"))

PRONOUNS = {
    "я", "меня", "мне", "мной",
    "ты", "тебя", "тебе", "тобой",
    "вы", "вас", "вам", "вами",
    "он", "его", "ему", "им",
    "она", "её", "ее", "ей", "ею",
    "оно",
    "мы", "нас", "нам", "нами",
    "они", "их", "ими",
    "себя", "себе", "собой",
}

DISCOURSE_MARKERS = [
    "рассмотрим", "покажем", "докажем", "обозначим", "пусть",
    "следовательно", "итак", "таким образом", "однако", "впрочем",
    "например", "в частности", "с другой стороны", "заметим",
    "поэтому", "давайте", "представим",
]

FEATURE_NAMES = [
    "pron_share", "excl_share", "ques_share", "elli_share",
    "latin_share", "digit_share", "marker_rate", "marker_any",
    "avg_sent_len", "std_sent_len", "text_len_norm",
    "stop_share", "ttr",
]

_TOK = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", re.UNICODE)
_SENT = re.compile(r"[.!?…]+")
_LAT = re.compile(r"[A-Za-z]")
_CYR = re.compile(r"[А-Яа-яЁё]")
_DIG = re.compile(r"\d")


def tokenize(text: str):
    return _TOK.findall(text.lower())


def split_sents(text: str):
    s = [x.strip() for x in _SENT.split(text) if x.strip()]
    return s if s else [text.strip()]


def _marker_count(text_lc: str, marker: str) -> int:
    if " " in marker:
        return text_lc.count(marker)
    return len(re.findall(rf"\b{re.escape(marker)}\b", text_lc))


def extract_features_one(text) -> np.ndarray:
    text = str(text) if text else ""
    tl = text.lower()
    tokens = tokenize(tl)
    n_tok = max(len(tokens), 1)
    n_ch = max(len(text), 1)

    sents = split_sents(text)
    slens = [len(tokenize(s)) for s in sents]

    lat = len(_LAT.findall(text))
    cyr = len(_CYR.findall(text))
    let = max(lat + cyr, 1)

    mc = [_marker_count(tl, m) for m in DISCOURSE_MARKERS]
    ms = sum(mc)

    return np.array([
        sum(1 for t in tokens if t in PRONOUNS) / n_tok,
        text.count("!") / n_ch,
        text.count("?") / n_ch,
        (text.count("…") + text.count("...")) / n_ch,
        lat / let,
        len(_DIG.findall(text)) / n_ch,
        ms / n_tok,
        float(ms > 0),
        float(np.mean(slens)) if slens else 0.0,
        float(np.std(slens)) if len(slens) > 1 else 0.0,
        np.log(n_tok + 1),
        sum(1 for t in tokens if t in RUS_STOPWORDS) / n_tok,
        len(set(tokens)) / n_tok,
    ], dtype=np.float32)


def extract_features(texts) -> np.ndarray:
    if isinstance(texts, pd.Series):
        texts = texts.values
    return np.vstack([extract_features_one(t) for t in texts])


_PRON_RE = re.compile(
    r"\b(" + "|".join(map(re.escape, sorted(PRONOUNS, key=len, reverse=True))) + r")\b",
    re.IGNORECASE,
)


def ablate_no_pronouns(t) -> str:
    return re.sub(r"\s+", " ", _PRON_RE.sub(" ", str(t))).strip()


def ablate_no_punct(t) -> str:
    t = str(t).replace("...", "…")
    t = re.sub(r"[!?…]+", " ", t)
    t = t.replace("«", "").replace("»", "")
    return re.sub(r"\s+", " ", t).strip()


def ablate_clip(t, n: int = 200) -> str:
    toks = _TOK.findall(str(t))
    return " ".join(toks[:n]) if len(toks) > n else str(t)
