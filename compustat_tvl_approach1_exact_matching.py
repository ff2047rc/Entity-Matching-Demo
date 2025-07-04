from pathlib import Path
import pandas as pd

# Set relative paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# File paths
comp_file = DATA_DIR / "compustat_sample.csv"
tvl_file = DATA_DIR / "tvl_sample.csv"

# Load data
comp = pd.read_csv(comp_file)
tvl = pd.read_csv(tvl_file)

# Inspect relevant columns
print("Compustat columns:", comp.columns.tolist())
print("TVL columns:", tvl.columns.tolist())

# Standardize and harmonize CUSIP formats
comp["cusip_std"] = comp["cusip"].astype(str).str.upper().str.replace(r"\W", "", regex=True).str.strip()
tvl["cusip_std"] = tvl["cusip"].astype(str).str.upper().str.replace(r"\W", "", regex=True).str.strip()

# Remove entries with missing or invalid CUSIP
comp = comp[comp["cusip_std"].str.len() >= 6]
tvl = tvl[tvl["cusip_std"].str.len() >= 6]

# Inner join on standardized CUSIP
matches = pd.merge(comp, tvl, on="cusip_std", how="inner", suffixes=("_comp", "_tvl"))

# Drop potential duplicates (optional, for one-to-one matching only)
matches = matches.drop_duplicates(subset=["cusip_std"])

# Log diagnostics
n_comp = len(comp)
n_tvl = len(tvl)
n_match = len(matches)

attrition = pd.DataFrame({
    "Source": ["Compustat", "TVL", "Matched"],
    "N_Records": [n_comp, n_tvl, n_match],
    "Share of Compustat": [1.0, None, n_match / n_comp],
    "Share of TVL": [None, 1.0, n_match / n_tvl]
})

# Save outputs
matches.to_csv(OUTPUT_DIR / "compustat_tvl_approach1_matched_exact_by_cusip.csv", index=False)
attrition.to_csv(OUTPUT_DIR / "compustat_tvl_approach1_attrition_table_exact_by_cusip.csv", index=False)

print(attrition)
print(f"\nFinished. Full results saved to {OUTPUT_DIR}/")
