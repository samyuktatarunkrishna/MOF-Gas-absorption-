import pandas as pd
from pathlib import Path

def main():
    ID_COL = "refcode"
    mof = pd.read_csv("data/raw/MOFCSD.csv")
    mof[ID_COL] = mof[ID_COL].astype(str)
    mof_ids = set(mof[ID_COL].tolist())

    crafted_path = Path("data/processed/crafted_refcodes_filtered.txt")
    assert crafted_path.exists(), "Missing crafted_refcodes_filtered.txt. Run crafted_id_list.py first."

    crafted_ids = set(crafted_path.read_text().splitlines())
    overlap = sorted(mof_ids & crafted_ids)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "overlap_refcodes.txt").write_text("\n".join(overlap))

    print("MOFCSD refcodes:", len(mof_ids))
    print("Filtered CRAFTED refcodes:", len(crafted_ids))
    print("OVERLAP:", len(overlap))
    print("Saved: data/processed/overlap_refcodes.txt")

if __name__ == "__main__":
    main()
