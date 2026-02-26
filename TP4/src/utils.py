# TP4/src/utils.py
from __future__ import annotations
from dataclasses import dataclass
import time
import random
import os
import numpy as np
import torch
from sklearn.metrics import f1_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@dataclass
class Timer:
    t0: float = 0.0
    t1: float = 0.0

    def __enter__(self) -> "Timer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.t1 = time.perf_counter()

    @property
    def elapsed_s(self) -> float:
        return self.t1 - self.t0


def compute_metrics(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> dict:
    preds = logits.argmax(dim=1).cpu().numpy()
    y_np = y.cpu().numpy()
    acc = (preds == y_np).mean()
    macro_f1 = f1_score(y_np, preds, average="macro", zero_division=0)
    return {"acc": float(acc), "macro_f1": float(macro_f1)}
