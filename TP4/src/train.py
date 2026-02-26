# TP4/src/train.py
"""
Train MLP, GCN, or GraphSAGE on Cora.

Usage:
    python TP4/src/train.py --config TP4/configs/baseline_mlp.yaml --model mlp
    python TP4/src/train.py --config TP4/configs/gcn.yaml --model gcn
    python TP4/src/train.py --config TP4/configs/sage_sampling.yaml --model sage
"""
from __future__ import annotations
import argparse
import os
import sys

import torch
import torch.nn as nn
import yaml

# Allow running from repo root: python TP4/src/train.py
sys.path.insert(0, os.path.dirname(__file__))

from data import load_cora
from models import MLP, GCN, GraphSAGE
from utils import Timer, compute_metrics, set_seed


def build_model(args_model: str, cfg: dict, num_features: int, num_classes: int) -> nn.Module:
    if args_model == "mlp":
        c = cfg["mlp"]
        return MLP(
            in_dim=num_features,
            hidden_dim=c["hidden_dim"],
            out_dim=num_classes,
            dropout=c["dropout"],
        )
    elif args_model == "gcn":
        c = cfg["gcn"]
        return GCN(
            in_dim=num_features,
            hidden_dim=c["hidden_dim"],
            out_dim=num_classes,
            dropout=c["dropout"],
        )
    elif args_model == "sage":
        c = cfg["sage"]
        return GraphSAGE(
            in_dim=num_features,
            hidden_dim=c["hidden_dim"],
            out_dim=num_classes,
            dropout=c["dropout"],
        )
    else:
        raise ValueError(f"Unknown model: {args_model}")


def train_full_batch(model, cora_data, device, cfg, model_name: str):
    """Full-batch training for MLP and GCN."""
    x = cora_data.x.to(device)
    y = cora_data.y.to(device)
    use_edge = model_name != "mlp"  # MLP has no graph structure
    edge_index = cora_data.edge_index.to(device) if use_edge else None

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        if use_edge:
            out = model(x, edge_index)
        else:
            out = model(x)
        loss = criterion(out[cora_data.train_mask.to(device)], y[cora_data.train_mask.to(device)])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == cfg["epochs"]:
            model.eval()
            with torch.no_grad():
                if use_edge:
                    logits = model(x, edge_index)
                else:
                    logits = model(x)
            logits = logits.cpu()
            y_cpu = cora_data.y
            train_m = compute_metrics(logits[cora_data.train_mask], y_cpu[cora_data.train_mask], cora_data.num_classes)
            val_m = compute_metrics(logits[cora_data.val_mask], y_cpu[cora_data.val_mask], cora_data.num_classes)
            test_m = compute_metrics(logits[cora_data.test_mask], y_cpu[cora_data.test_mask], cora_data.num_classes)
            print(
                f"epoch={epoch:>3d}  loss={loss.item():.4f}"
                f"  train_acc={train_m['acc']:.4f}  val_acc={val_m['acc']:.4f}"
                f"  test_acc={test_m['acc']:.4f}  test_f1={test_m['macro_f1']:.4f}"
            )
    return test_m


def train_minibatch(model, cora_data, raw_data, device, cfg):
    """Mini-batch training for GraphSAGE using NeighborLoader."""
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.sampler import NeighborSampler

    sage_cfg = cfg["sage"]
    # Explicit NeighborSampler avoids needing pyg-lib or torch-sparse
    sampler = NeighborSampler(raw_data, num_neighbors=sage_cfg["num_neighbors"])
    loader = NeighborLoader(
        raw_data,
        num_neighbors=sage_cfg["num_neighbors"],
        batch_size=sage_cfg["batch_size"],
        input_nodes=raw_data.train_mask,
        neighbor_sampler=sampler,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            # Only supervise seed nodes (first batch_size rows)
            out = out[: batch.batch_size]
            target = batch.y[: batch.batch_size]
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * batch.batch_size
            total_examples += batch.batch_size

        if epoch % 10 == 0 or epoch == cfg["epochs"]:
            model.eval()
            # Full-batch evaluation for consistency
            x = cora_data.x.to(device)
            edge_index = raw_data.edge_index.to(device)
            with torch.no_grad():
                logits = model(x, edge_index).cpu()
            y_cpu = cora_data.y
            avg_loss = total_loss / total_examples
            train_m = compute_metrics(logits[cora_data.train_mask], y_cpu[cora_data.train_mask], cora_data.num_classes)
            val_m = compute_metrics(logits[cora_data.val_mask], y_cpu[cora_data.val_mask], cora_data.num_classes)
            test_m = compute_metrics(logits[cora_data.test_mask], y_cpu[cora_data.test_mask], cora_data.num_classes)
            print(
                f"epoch={epoch:>3d}  loss={avg_loss:.4f}"
                f"  train_acc={train_m['acc']:.4f}  val_acc={val_m['acc']:.4f}"
                f"  test_acc={test_m['acc']:.4f}  test_f1={test_m['macro_f1']:.4f}"
            )
    return test_m


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GNN on Cora")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--model", required=True, choices=["mlp", "gcn", "sage"], help="Model architecture")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Load Cora (high-level wrapper)
    cora_data = load_cora()

    # Load raw PyG Data object (needed for edge_index and NeighborLoader)
    import os as _os
    from torch_geometric.datasets import Planetoid
    root = _os.environ.get("PYG_DATA_ROOT", _os.path.expanduser("~/.cache/pyg_data"))
    dataset = Planetoid(root=root, name="Cora")
    raw_data = dataset[0]

    model = build_model(args.model, cfg, cora_data.num_features, cora_data.num_classes)
    model = model.to(device)
    print(f"model: {model.__class__.__name__}  params={sum(p.numel() for p in model.parameters())}")

    print(f"\n--- Training ({args.model.upper()}) ---")
    with Timer() as total_timer:
        with Timer() as loop_timer:
            if args.model == "sage":
                test_metrics = train_minibatch(model, cora_data, raw_data, device, cfg)
            else:
                # Attach edge_index to cora_data for convenience
                cora_data.edge_index = raw_data.edge_index
                test_metrics = train_full_batch(model, cora_data, device, cfg, args.model)

    print(f"\ntotal_train_time_s: {total_timer.elapsed_s:.2f}")
    print(f"train_loop_time_s:  {loop_timer.elapsed_s:.2f}")
    print(f"test_acc:           {test_metrics['acc']:.4f}")
    print(f"test_macro_f1:      {test_metrics['macro_f1']:.4f}")

    # Save checkpoint (Exercise 5)
    runs_dir = os.path.join(os.path.dirname(__file__), "..", "runs")
    os.makedirs(runs_dir, exist_ok=True)
    ckpt_name = f"{args.model}.pt"
    ckpt_path = os.path.join(runs_dir, ckpt_name)
    torch.save(
        {
            "model": args.model,
            "config_path": args.config,
            "state_dict": model.state_dict(),
            "num_features": cora_data.num_features,
            "num_classes": cora_data.num_classes,
            "cfg": cfg,
        },
        ckpt_path,
    )
    print(f"\ncheckpoint saved -> {ckpt_path}")


if __name__ == "__main__":
    main()
