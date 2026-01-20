import json, random
import pandas as pd
from pathlib import Path

ID_COL = "refcode"
SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

IN_PATH = "data/processed/MOFCSD_with_co2_labels.csv"
OUT_PATH = "data/processed/splits_seed42.json"

def main():
    df = pd.read_csv(IN_PATH)
    df = df[df["co2_uptake"].notna()].copy()

    ids = df[ID_COL].astype(str).tolist()
    random.seed(SEED)
    random.shuffle(ids)

    n = len(ids)
    n_train = int(TRAIN_FRAC * n)
    n_val = int(VAL_FRAC * n)

    splits = {
        "seed": SEED,
        "train": ids[:n_train],
        "val": ids[n_train:n_train+n_val],
        "test": ids[n_train+n_val:],
    }

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path(OUT_PATH).write_text(json.dumps(splits, indent=2))

    print("Labeled MOFs:", n)
    print("Train/Val/Test:", len(splits["train"]), len(splits["val"]), len(splits["test"]))
    print("Saved:", OUT_PATH)

if __name__ == "__main__":
    main()

