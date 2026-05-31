from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..utils import SEED

PARAM_GRID = {
    "lr__C": [0.01, 0.1, 1.0, 10.0],
    "lr__class_weight": [None, "balanced"],
}


def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(solver="liblinear", max_iter=2000, random_state=SEED)),
    ])
