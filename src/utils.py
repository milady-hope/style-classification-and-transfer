import random
import warnings

import numpy as np

SEED = 42

LABEL_NAMES = {0: "Науч.-поп.", 1: "Научный"}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device():
    try:
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        return None


def setup_warnings() -> None:
    warnings.filterwarnings("ignore")
