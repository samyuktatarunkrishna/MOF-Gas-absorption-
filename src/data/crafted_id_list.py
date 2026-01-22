from pathlib import Path
import re

NUMERIC_STYLE = re.compile(r"^\d+N\d+$")  # e.g., 05000N2, 07010N3

def main():
    crafted_parent = Path("data/external/crafted")
    iso_dir = next(crafted_parent.rglob("ISOTHERM_FILES"))

    all_ids = set()
    refcodes = set()

    for fp in iso_dir.glob("*.csv"):
        parts = fp.stem.split("_")
        if len(parts) < 5:
            continue
        token = parts[1]  # 2nd token
        all_ids.add(token)

        # keep only CSD-refcode-like tokens (exclude 05000N2-style)
        if not NUMERIC_STYLE.match(token):
            refcodes.add(token)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "crafted_ids_all.txt").write_text("\n".join(sorted(all_ids)))
    (out_dir / "crafted_refcodes_filtered.txt").write_text("\n".join(sorted(refcodes)))

    print("Detected ISOTHERM_FILES:", iso_dir)
    print("All tokens found:", len(all_ids))
    print("Filtered refcodes (non-\\d+N\\d+):", len(refcodes))
    print("Sample filtered refcodes:", sorted(refcodes)[:30])

if __name__ == "__main__":
    main()
