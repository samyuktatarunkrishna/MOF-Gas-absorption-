from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# ===== settings (keep consistent for thesis) =====
ID_COL = "refcode"
TARGET_GAS = "CO2"
TARGET_TEMP = 298
TARGET_P_BAR = 1.0

# Choose ONE consistent simulation setting
# Based on your folder listing, these exist widely:
CHARGE_SCHEME = "DDEC"     # e.g., DDEC / EQeq / Qeq / NEUTRAL / PACMOF / MPNN
FORCEFIELD = "UFF"         # UFF or DREIDING

CRAFTED_PARENT = Path("data/external/crafted")
ISO_DIR = next(CRAFTED_PARENT.rglob("ISOTHERM_FILES"))

OVERLAP_PATH = Path("data/processed/overlap_refcodes.txt")
OUT_PATH = Path("data/processed/labels_co2_crafted_298K_1bar.csv")

def read_isotherm(fp: Path) -> pd.DataFrame:
    # CRAFTED is usually CSV with headers; this is robust just in case
    for sep in [",", "\t", r"\s+"]:
        try:
            df = pd.read_csv(fp, sep=sep, engine="python")
            if df.shape[1] >= 2:
                return df
        except Exception:
            pass
    raise ValueError(f"Could not parse isotherm file: {fp}")

def pick_cols(df: pd.DataFrame) -> tuple[str, str]:
    # Try to find pressure & uptake columns by name
    p_cands = [c for c in df.columns if "press" in c.lower() or c.lower() in ["p", "pressure"]]
    u_cands = [c for c in df.columns if any(k in c.lower() for k in ["uptake", "loading", "amount", "adsorb"])]
    if p_cands and u_cands:
        return p_cands[0], u_cands[0]

    # Fallback: first two numeric columns
    num_cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(5, int(0.5 * len(df))):
            num_cols.append(c)
    if len(num_cols) >= 2:
        return num_cols[0], num_cols[1]

    # Last resort
    return df.columns[0], df.columns[1]

def uptake_at_pressure(df: pd.DataFrame, p_target: float) -> float:
    pcol, ucol = pick_cols(df)
    p = pd.to_numeric(df[pcol], errors="coerce").to_numpy()
    u = pd.to_numeric(df[ucol], errors="coerce").to_numpy()
    m = np.isfinite(p) & np.isfinite(u)
    p, u = p[m], u[m]
    if len(p) < 2:
        return np.nan

    idx = np.argsort(p)
    p, u = p[idx], u[idx]

    # interpolate if inside range, else nearest
    if p_target < p.min() or p_target > p.max():
        return float(u[np.argmin(np.abs(p - p_target))])
    return float(np.interp(p_target, p, u))

def parse_filename(fp: Path) -> dict:
    # Expect: CHARGE_REFCODE_FF_GAS_TEMP.csv
    parts = fp.stem.split("_")
    if len(parts) < 5:
        return {}
    charge, refcode, ff, gas, temp = parts[0], parts[1], parts[2], parts[3], parts[4]
    try:
        temp = int(temp)
    except Exception:
        return {}
    return {"charge": charge, "refcode": refcode, "ff": ff, "gas": gas, "temp": temp}

def main():
    assert ISO_DIR.exists(), f"Missing {ISO_DIR}"
    assert OVERLAP_PATH.exists(), f"Missing {OVERLAP_PATH}. Run overlap script first."

    overlap = set(OVERLAP_PATH.read_text().splitlines())

    rows = []
    scanned = 0
    matched = 0

    for fp in ISO_DIR.glob("*.csv"):
        meta = parse_filename(fp)
        if not meta:
            continue
        scanned += 1

        if meta["gas"].upper() != TARGET_GAS:
            continue
        if meta["temp"] != TARGET_TEMP:
            continue
        if meta["charge"].upper() != CHARGE_SCHEME.upper():
            continue
        if meta["ff"].upper() != FORCEFIELD.upper():
            continue
        if meta["refcode"] not in overlap:
            continue

        matched += 1
        df = read_isotherm(fp)
        y = uptake_at_pressure(df, TARGET_P_BAR)

        rows.append({
            ID_COL: meta["refcode"],
            "co2_uptake": y,
            "T_K": TARGET_TEMP,
            "P_bar": TARGET_P_BAR,
            "charge_scheme": CHARGE_SCHEME,
            "forcefield": FORCEFIELD,
            "source": "CRAFTED-2.0.1",
            "file": fp.name
        })

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["co2_uptake"]).drop_duplicates(subset=[ID_COL])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print("Scanned isotherm files:", scanned)
    print("Matched (after filters + overlap):", matched)
    print("Final labels saved:", len(out))
    print("Saved to:", OUT_PATH)

if __name__ == "__main__":
    main()
