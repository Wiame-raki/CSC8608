# TP4/src/data.py (mise à jour)
import os
from dataclasses import dataclass
import torch
from torch_geometric.datasets import Planetoid


@dataclass
class CoraData:
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    num_features: int
    num_classes: int


def load_cora() -> CoraData:
    root = os.environ.get("PYG_DATA_ROOT", os.path.expanduser("~/.cache/pyg_data"))
    dataset = Planetoid(root=root, name="Cora")
    data = dataset[0]

    return CoraData(
        x=data.x,
        y=data.y,
        edge_index=data.edge_index,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
        num_features=dataset.num_node_features,
        num_classes=dataset.num_classes,
    )