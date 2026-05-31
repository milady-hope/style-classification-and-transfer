import re
from typing import Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .utils import SEED

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w{2,}\b")
_MSPACE = re.compile(r"\s+")


def clean_text(s) -> str:
    s = str(s).replace("\u200b", "")
    s = _URL.sub(" ", s)
    s = _EMAIL.sub(" ", s)
    s = s.replace("\t", " ").replace("\xa0", " ")
    return _MSPACE.sub(" ", s).strip()


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = ["text", "label", "pair_id"]
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df["pair_id"] = df["pair_id"].astype(str)
    df["text_clean"] = df["text"].map(clean_text)
    return df


def split_by_pairs(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df["text_clean"], df["label"], groups=df["pair_id"]))
    train_df = df.iloc[tr].reset_index(drop=True)
    test_df = df.iloc[te].reset_index(drop=True)
    assert len(set(train_df["pair_id"]) & set(test_df["pair_id"])) == 0
    return train_df, test_df
