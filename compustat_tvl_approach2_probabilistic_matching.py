import pandas as pd
from pathlib import Path
from rapidfuzz import fuzz

# Set relative paths
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# File paths
comp_file = DATA_DIR / "compustat_sample.csv"
tvl_file  = DATA_DIR / "tvl_sample.csv"

# Load data
comp = pd.read_csv(comp_file)
tvl  = pd.read_csv(tvl_file)

# Set column names for matching
COMP_NAME_COL = "conm"         # Compustat company name
TVL_NAME_COL  = "tv_org_name"  # TVL company name

# Canonicalise names
def canon(text: str) -> str:
    import re, unicodedata
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(inc|corp|corporation|ltd|llc|co|sa|plc)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

comp["name_canon"] = comp[COMP_NAME_COL].map(canon)
tvl["name_canon"]  = tvl[TVL_NAME_COL].map(canon)

comp["block"] = comp["name_canon"].str.split().str[0]
tvl["block"]  = tvl["name_canon"].str.split().str[0]

# Build block dictionary from TVL side
block_dict = (
    tvl.reset_index()[["index", "block", "name_canon"]]
    .groupby("block")
    .apply(lambda df: list(zip(df["index"], df["name_canon"])))
    .to_dict()
)

# RapidFuzz token_set matching
THRESH = 80 # Arbitrary threshold for fuzzy matching; You can adjust this based on your needs

matches = []

for idx, row in comp.iterrows():
    candidates = block_dict.get(row["block"], [])
    best_idx, best_score = None, 0
    for tvl_idx, cand_name in candidates:
        score = fuzz.token_set_ratio(row["name_canon"], cand_name)
        if score > best_score:
            best_idx, best_score = tvl_idx, score

    if best_score >= THRESH:
        tvl_row = tvl.loc[best_idx]
        matches.append({
            "COMP_NAME":         row[COMP_NAME_COL],
            "TVL_NAME":          tvl_row[TVL_NAME_COL],
            "FUZZY_SCORE":       best_score,
            "link_source":       "fuzzy_token_set"
        })

fuzzy_df = pd.DataFrame(matches)

# Diagnostics
attrition = pd.DataFrame({
    "Step": ["Compustat input", "TVL input", "Fuzzy matched"],
    "Rows": [len(comp), len(tvl), len(fuzzy_df)]
})

# Save outputs
fuzzy_csv  = OUTPUT_DIR / "compustat_tvl_approach2_matched_fuzzy_by_name.csv"
attr_csv   = OUTPUT_DIR / "compustat_tvl_approach2_attrition_table_fuzzy_match.csv"

fuzzy_df.to_csv(fuzzy_csv,  index=False)
attrition.to_csv(attr_csv, index=False)

# Console summary
pd.set_option("display.max_rows", None)          # show all matches
print("\n=== ATTRITION TABLE ===")
print(attrition.to_string(index=False))

print(f"\nFinished. Full results saved to {OUTPUT_DIR}/")
