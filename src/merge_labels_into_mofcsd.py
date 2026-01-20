
import pandas as pd
from pathlib import Path

ID_COL = "refcode"

MOF_PATH = "data/raw/MOFCSD.csv"
LABEL_PATH = "data/processed/labels_co2_crafted_298K_1bar.csv"
OUT_PATH = "data/processed/MOFCSD_with_co2_labels.csv"

def main():
    mof = pd.read_csv(MOF_PATH)
    print("MOF rows:", len(mof), "cols:", len(mof.columns))

    print("Loading:", LABEL_PATH)
    lab = pd.read_csv(LABEL_PATH)
    print("Label rows:", len(lab), "cols:", len(lab.columns))

    mof[ID_COL] = mof[ID_COL].astype(str)
    lab[ID_COL] = lab[ID_COL].astype(str)

    merged = mof.merge(lab[[ID_COL, "co2_uptake"]], on=ID_COL, how="left")

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Total MOFs:", len(merged))
    print("Labeled MOFs:", int(merged["co2_uptake"].notna().sum()))

if __name__ == "__main__":
    main()


