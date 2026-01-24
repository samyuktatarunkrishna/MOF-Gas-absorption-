from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

THRESHOLD = "threshold_0.10"
RUNS = ["run_0", "run_1", "run_2", "run_3"]
DATA_DIR = Path("data/processed/pyg") / THRESHOLD

OUT_DIR = Path("outputs/gnn_gcn_shuffled") / THRESHOLD
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

class GCNRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.lin(x).squeeze(-1)
        return x

def shuffle_edges(edge_index, edge_weight, num_nodes):
    # Shuffle targets while keeping sources: breaks structure but preserves degree-ish and weights distribution
    src = edge_index[0].clone()
    dst = edge_index[1].clone()
    perm = torch.randperm(dst.numel())
    dst = dst[perm]
    w = edge_weight.clone()[perm]
    # Ensure indices stay in range (they will)
    dst = dst.clamp(0, num_nodes - 1)
    return torch.stack([src, dst], dim=0), w

def rmse(y_true, y_pred):
    return float(torch.sqrt(F.mse_loss(y_pred, y_true)).detach().cpu())

def mae(y_true, y_pred):
    return float(F.l1_loss(y_pred, y_true).detach().cpu())

def r2(y_true, y_pred):
    y = y_true.detach().cpu().numpy()
    p = y_pred.detach().cpu().numpy()
    ss_res = ((y - p) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum() + 1e-12
    return float(1.0 - ss_res / ss_tot)

@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_weight)
    y = data.y
    pred_m = pred[mask]
    y_m = y[mask]
    return {
        "rmse": rmse(y_m, pred_m),
        "mae": mae(y_m, pred_m),
        "r2": r2(y_m, pred_m),
    }

def train_one(run: str, epochs: int = 300, lr: float = 1e-3, wd: float = 1e-4):
    set_seed(SEED)
    data = torch.load(DATA_DIR / f"{run}.pt", map_location="cpu")

    # shuffle edges ONCE per run
    edge_index, edge_weight = shuffle_edges(data.edge_index, data.edge_weight, data.num_nodes)
    data.edge_index = edge_index
    data.edge_weight = edge_weight

    data = data.to(DEVICE)

    model = GCNRegressor(in_dim=data.num_node_features, hidden=64, dropout=0.2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val = float("inf")
    best_state = None
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(data.x, data.edge_index, data.edge_weight)
        loss = F.mse_loss(pred[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        val_m = evaluate(model, data, data.val_mask)
        history.append({"epoch": ep, "train_mse": float(loss.detach().cpu()), **{f"val_{k}": v for k, v in val_m.items()}})

        if val_m["rmse"] < best_val:
            best_val = val_m["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    val_final = evaluate(model, data, data.val_mask)
    test_final = evaluate(model, data, data.test_mask)

    (OUT_DIR / "history").mkdir(parents=True, exist_ok=True)
    Path(OUT_DIR / "history" / f"{run}.json").write_text(json.dumps(history, indent=2))
    torch.save(best_state, OUT_DIR / f"best_model_{run}.pt")

    print(f"\n{run} VAL : {val_final}")
    print(f"{run} TEST: {test_final}")
    return val_final, test_final

def main():
    all_val, all_test = [], []
    for run in RUNS:
        v, t = train_one(run)
        all_val.append(v)
        all_test.append(t)

    def agg(lst, key):
        vals = np.array([d[key] for d in lst], dtype=float)
        return float(vals.mean()), float(vals.std(ddof=1))

    summary = {
        "threshold": THRESHOLD,
        "val": {k: {"mean": agg(all_val, k)[0], "std": agg(all_val, k)[1]} for k in ["rmse", "mae", "r2"]},
        "test": {k: {"mean": agg(all_test, k)[0], "std": agg(all_test, k)[1]} for k in ["rmse", "mae", "r2"]},
    }
    Path(OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== MEAN ± STD across runs (SHUFFLED) ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

