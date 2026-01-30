import torch
import numpy as np
from pathlib import Path
from train_gnn_gcn import GCNRegressor
from tqdm import tqdm

THRESHOLD = "threshold_0.10"
RUNS = ["run_0", "run_1", "run_2", "run_3"]
DEVICE = "cpu"
OUT_DIR = Path("outputs/gnn_gcn") / THRESHOLD
DATA_DIR = Path("data/processed/pyg") / THRESHOLD

def main():
    preds = []
    base_shape = None

    for run in tqdm(RUNS):
        data_path = DATA_DIR / f"{run}.pt"
        model_path = OUT_DIR / f"best_model_{run}.pt"

        data = torch.load(data_path, map_location=DEVICE)
        model = GCNRegressor(in_dim=data.num_node_features)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        pred = model(data.x, data.edge_index, data.edge_weight)
        pred_np = pred.detach().cpu().numpy()

        if base_shape is None:
            base_shape = pred_np.shape
        elif pred_np.shape != base_shape:
            print(f"[WARNING] {run} prediction shape {pred_np.shape} != {base_shape}")
            continue  # Skip inconsistent predictions

        preds.append(pred_np)

    if len(preds) == 0:
        raise ValueError("No consistent predictions found across runs.")

    pred_stack = np.stack(preds, axis=0)  # [runs, nodes]
    mean_pred = pred_stack.mean(axis=0)
    std_pred = pred_stack.std(axis=0)

    np.savez(OUT_DIR / "aggregated_predictions.npz", mean=mean_pred, std=std_pred)
    print("[aggregate_predictions] Saved mean and std predictions.")

if __name__ == "__main__":
    main()
