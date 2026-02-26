# TP4/src/train.py
from __future__ import annotations
import argparse
import yaml
import torch
import torch.nn as nn
import time

from data import load_cora
from models import MLP
from utils import set_seed, Timer, compute_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    set_seed(int(cfg["seed"]))

    device_str = cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    data = load_cora()
    x = data.x.to(device)
    y = data.y.to(device)

    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    model = MLP(
        in_dim=data.num_features,
        hidden_dim=int(cfg["mlp"]["hidden_dim"]),
        out_dim=data.num_classes,
        dropout=float(cfg["mlp"]["dropout"]),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    epochs = int(cfg["epochs"])
    print("device:", device)
    print("epochs:", epochs)

    total_train_s = 0.0
    train_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        with Timer() as t:
            logits = model(x)
            loss = criterion(logits[train_mask], y[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_train_s += t.elapsed_s

        model.eval()
        with torch.no_grad():
            logits = model(x)

            m_train = compute_metrics(logits[train_mask], y[train_mask], data.num_classes)
            m_val = compute_metrics(logits[val_mask], y[val_mask], data.num_classes)
            m_test = compute_metrics(logits[test_mask], y[test_mask], data.num_classes)

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"epoch={epoch:03d} "
                f"loss={loss.item():.4f} "
                f"train_acc={m_train['acc']:.4f} val_acc={m_val['acc']:.4f} test_acc={m_test['acc']:.4f} "
                f"train_f1={m_train['macro_f1']:.4f} val_f1={m_val['macro_f1']:.4f} test_f1={m_test['macro_f1']:.4f} "
                f"epoch_time_s={t.elapsed_s:.4f}"
            )

    print(f"total_train_time_s={total_train_s:.4f}")
    train_loop_time = time.time() - train_start
    print(f"train_loop_time={train_loop_time:.4f}")


if __name__ == "__main__":
    main()
    