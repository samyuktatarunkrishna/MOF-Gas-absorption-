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
OUT_DIR = Path("outputs/gnn_gcn") / THRESHOLD
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

class GCNRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.3):
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
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x).squeeze(-1)
        return x

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
    }, pred.cpu()

def train_one(run: str, epochs: int = 300, lr: float = 1e-3, wd: float = 1e-4):
    set_seed(SEED)
    data = torch.load(DATA_DIR / f"{run}.pt", map_location="cpu").to(DEVICE)
    model = GCNRegressor(in_dim=data.num_node_features).to(DEVICE)
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

        val_metrics, _ = evaluate(model, data, data.val_mask)
        history.append({
            "epoch": ep,
            "train_mse": float(loss.detach().cpu()),
            **{f"val_{k}": v for k, v in val_metrics.items()}
        })

        if val_metrics["rmse"] < best_val:
            best_val = val_metrics["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Reload best model
    model.load_state_dict(best_state)
    val_final, _ = evaluate(model, data, data.val_mask)
    test_final, test_pred = evaluate(model, data, data.test_mask)

    # Save predictions for downstream use
    pred_dict = {
        "refcode": [data.refcode[i] for i in range(len(data.refcode)) if data.test_mask[i]],
        "y_true": data.y[data.test_mask].cpu().numpy().tolist(),
        "y_pred": test_pred[data.test_mask].cpu().numpy().tolist(),
    }
    Path(OUT_DIR / f"predictions_{run}.json").write_text(json.dumps(pred_dict, indent=2))

    # Save history + model
    (OUT_DIR / "history").mkdir(parents=True, exist_ok=True)
    Path(OUT_DIR / "history" / f"{run}.json").write_text(json.dumps(history, indent=2))
    torch.save(best_state, OUT_DIR / f"best_model_{run}.pt")

    print(f"\n{run} VAL : {val_final}")
    print(f"{run} TEST: {test_final}")
    return {
    "val": val_final,
    "test": test_final,
    "refcodes_test": [data.refcode[i] for i in range(len(data.refcode)) if data.test_mask[i]],
}

def main():
    all_val, all_test = [], []
    for run in RUNS:
        out = train_one(run)
        v, t = out["val"], out["test"]
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
    print("\n=== MEAN ± STD across runs ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()