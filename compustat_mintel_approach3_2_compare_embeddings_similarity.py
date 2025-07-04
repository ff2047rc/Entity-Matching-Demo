from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# Set relative paths
BASE      = Path(__file__).resolve().parent
OUT_DIR  = BASE / "outputs"

# File paths
MINTEL_EMB_FILE     = OUT_DIR / "mintel_embeddings.parquet"
COMPUSTAT_EMB_FILE  = OUT_DIR / "compustat_embeddings.parquet"
OUTPUT_FILE  = OUT_DIR / "mintel_vs_compustat_topK.csv"

TOP_K = 1  # You can change this value as needed

# Load Mintel and Compustat embeddings
print("Loading embeddings...")
mintel_df = pd.read_parquet(MINTEL_EMB_FILE)
compustat_df = pd.read_parquet(COMPUSTAT_EMB_FILE)

print(f"Mintel records: {len(mintel_df)}")
print(f"Compustat records: {len(compustat_df)}")

# Cosine similarity computation
print("Computing cosine similarities...")

mintel_embs = np.vstack(mintel_df["embedding"].values)
compustat_embs = np.vstack(compustat_df["embedding"].values)

# Calculate similarity matrix
similarity_matrix = cosine_similarity(mintel_embs, compustat_embs)

# Get top-K matches for each Mintel row
results = []
for i, mintel_row in tqdm(mintel_df.iterrows(), total=len(mintel_df), desc="Matching"):
    top_indices = similarity_matrix[i].argsort()[-TOP_K:][::-1]
    for idx in top_indices:
        results.append({
            "mintel_index": i,
            "mintel_company": mintel_df.loc[i, "company"],
            "ultimate_company": mintel_df.loc[i, "ultimate_company"],
            "compustat_conm": compustat_df.loc[idx, "conm"],
            "cosine_score": similarity_matrix[i, idx]
        })

# Save results to CSV
result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_FILE, index=False)
print(f"Top-K match results saved to: {OUTPUT_FILE}")
