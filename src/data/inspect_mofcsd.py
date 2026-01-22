import pandas as pd

path = "data/raw/MOFCSD.csv"
df = pd.read_csv(path)

print("Rows:", len(df))
print("Cols:", len(df.columns))
print("\nFirst 30 columns:\n", list(df.columns[:30]))

# show columns that look like identifiers
keywords = ["id", "mof", "ref", "csd", "name", "code", "uid"]
id_like = [c for c in df.columns if any(k in c.lower() for k in keywords)]
print("\nID-like columns:\n", id_like)

# show uniqueness ratio for the top candidates
cands = id_like[:15]
print("\nUniqueness check (first 15 ID-like cols):")
for c in cands:
    series = df[c].astype(str)
    nunique = series.nunique(dropna=True)
    print(f"{c:30s} unique={nunique:8d}  nonnull={series.notna().sum():8d}")
