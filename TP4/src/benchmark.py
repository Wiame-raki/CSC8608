# TP4/src/benchmark.py
"""
Inference latency benchmark for a trained checkpoint.

Usage:
    python TP4/src/benchmark.py --config TP4/configs/baseline_mlp.yaml --model mlp --ckpt TP4/runs/mlp.pt
    python TP4/src/benchmark.py --config TP4/configs/gcn.yaml --model gcn --ckpt TP4/runs/gcn.pt
    python TP4/src/benchmark.py --config TP4/configs/sage_sampling.yaml --model sage --ckpt TP4/runs/sage.pt
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from data import load_cora
from models import MLP, GCN, GraphSAGE


def sync_if_cuda(device: torch.device) -> None:

    if device.type == "cuda":
        torch.cuda.synchronize()


def build_model_from_ckpt(ckpt: dict) -> torch.nn.Module:
    model_name = ckpt["model"]
    cfg = ckpt["cfg"]
    num_features = ckpt["num_features"]
    num_classes = ckpt["num_classes"]

    if model_name == "mlp":
        c = cfg["mlp"]
        m = MLP(in_dim=num_features, hidden_dim=c["hidden_dim"], out_dim=num_classes, dropout=c["dropout"])
    elif model_name == "gcn":
        c = cfg["gcn"]
        m = GCN(in_dim=num_features, hidden_dim=c["hidden_dim"], out_dim=num_classes, dropout=c["dropout"])
    elif model_name == "sage":
        c = cfg["sage"]
        m = GCN(in_dim=num_features, hidden_dim=c["hidden_dim"], out_dim=num_classes, dropout=c["dropout"])
        # Use GraphSAGE
        m = GraphSAGE(in_dim=num_features, hidden_dim=c["hidden_dim"], out_dim=num_classes, dropout=c["dropout"])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return m


def forward_once(model: torch.nn.Module, x: torch.Tensor, edge_index, model_name: str):
    if model_name == "mlp":
        return model(x)
    else:
        return model(x, edge_index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference latency")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=["mlp", "gcn", "sage"])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--warmup", type=int, default=10, help="Warmup forward passes")
    parser.add_argument("--runs", type=int, default=100, help="Timed forward passes")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"model:  {args.model}")

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    model = build_model_from_ckpt(ckpt)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    # Load Cora data
    cora_data = load_cora()
    import os as _os
    from torch_geometric.datasets import Planetoid
    root = _os.environ.get("PYG_DATA_ROOT", _os.path.expanduser("~/.cache/pyg_data"))
    dataset = Planetoid(root=root, name="Cora")
    raw_data = dataset[0]

    x = cora_data.x.to(device)
    edge_index = raw_data.edge_index.to(device)
    num_nodes = x.shape[0]

    # Warmup
    print(f"\nWarmup ({args.warmup} passes) …")
    with torch.no_grad():
        for _ in range(args.warmup):
            forward_once(model, x, edge_index, args.model)
    sync_if_cuda(device)

    # Timed runs
    print(f"Timing ({args.runs} passes) …")
    times = []
    with torch.no_grad():
        for _ in range(args.runs):
            sync_if_cuda(device)
            t0 = time.perf_counter()
            forward_once(model, x, edge_index, args.model)
            sync_if_cuda(device)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)

    print(f"\n{'='*45}")
    print(f"  model            : {args.model}")
    print(f"  device           : {device}")
    print(f"  num_nodes        : {num_nodes}")
    print(f"  warmup_passes    : {args.warmup}")
    print(f"  timed_passes     : {args.runs}")
    print(f"  avg_forward_ms   : {avg_ms:.3f}")
    print(f"  min_forward_ms   : {min_ms:.3f}")
    print(f"  max_forward_ms   : {max_ms:.3f}")
    print(f"  ms_per_node      : {avg_ms / num_nodes * 1000:.4f}  (µs)")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()
