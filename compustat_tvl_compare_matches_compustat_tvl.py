import pandas as pd
from pathlib import Path

# Set relative paths
BASE     = Path(__file__).resolve().parent
OUT_DIR  = BASE / "outputs"

# File paths
exact_fp = OUT_DIR / "compustat_tvl_approach1_matched_exact_by_cusip.csv"
fuzzy_fp = OUT_DIR / "compustat_tvl_approach2_matched_fuzzy_by_name.csv"
llm_fp   = OUT_DIR / "compustat_tvl_approach3_match_gpt4o_review.csv"

# Load gold standard (exact match) and build a comparable key
exact = (
    pd.read_csv(exact_fp)
    .assign(key=lambda d: d["conm"].str.lower().str.strip() + " | "
                          + d["tv_org_name"].str.lower().str.strip())
)

gold_set = set(exact["key"])
n_gold   = len(gold_set)

# Evaluate Probabilistic / Fuzzy
fuzzy = (
    pd.read_csv(fuzzy_fp)
    .assign(key=lambda d: d["COMP_NAME"].str.lower().str.strip() + " | "
                          + d["TVL_NAME"].str.lower().str.strip())
)

fuzzy_preds = set(fuzzy["key"])
tp_fuzzy    = len(gold_set & fuzzy_preds)
precision_fuzzy = tp_fuzzy / len(fuzzy_preds) if fuzzy_preds else 0
coverage_fuzzy  = tp_fuzzy / n_gold

# Evaluate LLM (count only rows GPT labelled "Yes")
llm = (
    pd.read_csv(llm_fp)
    .query("Answer.str.lower() == 'yes'", engine="python")
    .assign(key=lambda d: d["conm"].str.lower().str.strip() + " | "
                          + d["tv_org_name"].str.lower().str.strip())
)

llm_preds = set(llm["key"])
tp_llm    = len(gold_set & llm_preds)
precision_llm = tp_llm / len(llm_preds) if llm_preds else 0
coverage_llm  = tp_llm / n_gold

# Show comparison table
summary = pd.DataFrame({
    "Approach": ["Exact (Gold)", "Probabilistic / Fuzzy", "LLM"],
    "True Positives": [n_gold, tp_fuzzy, tp_llm],
    "Predicted Pairs": [n_gold, len(fuzzy_preds), len(llm_preds)],
    "Coverage (Recall)": [1.0, coverage_fuzzy, coverage_llm],
    "Precision":        [1.0, precision_fuzzy, precision_llm]
})

print(summary.to_string(index=False, float_format="{:.2%}".format))
print(f"\nFinished. Full results saved to {OUT_DIR}/")
