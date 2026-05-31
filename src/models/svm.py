from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ..utils import SEED

PARAM_GRID = {
    "tfidf__ngram_range": [(3, 5), (3, 6)],
    "tfidf__min_df": [2, 3],
    "clf__C": [0.5, 1.0, 2.0],
}


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", sublinear_tf=True)),
        ("clf", LinearSVC(max_iter=5000, random_state=SEED)),
    ])
