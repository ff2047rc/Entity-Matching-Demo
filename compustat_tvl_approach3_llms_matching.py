import os, json, time, logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import pandas as pd

# Set relative paths
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR    = BASE_DIR / "debug"
for p in (OUTPUT_DIR, LOG_DIR): p.mkdir(exist_ok=True)

# File paths
COMP_FILE  = DATA_DIR / "compustat_sample.csv"
TVL_FILE   = DATA_DIR / "tvl_sample.csv"
OUTPUT_CSV = OUTPUT_DIR / "compustat_tvl_approach3_match_gpt4o_review.csv"

# Debug logging setup 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "debug_messages.log"),
              logging.StreamHandler()]
)

#  OpenAI key 
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise EnvironmentError("OPENAI_API_KEY not set in environment.")

#  Throttle settings 
TOKEN_LIMIT  = 3_900_000
MAX_WORKERS  = 50

# Prompt template 
FORMAT_INSTR = json.dumps({
    "conm":        "Compustat company name",
    "tv_org_name": "TVL company name",
    "Answer":      "\"Yes\" or \"No\"",
    "Explanation": "Brief justification"
}, indent=4)

PROMPT = f"""
Decide whether the two company names refer to the **same economic entity**.
Return a JSON object using the schema below.  Answer must be "Yes" or "No".

Schema:
{FORMAT_INSTR}
"""

# Function to ask the LLM for matching 
def ask_llm(conm, tv_name):
    messages = [
        {"role": "system", "content": "You are an expert on corporate ownership and branding."},
        {"role": "user", "content": f"{PROMPT}\nconm: {conm}\ntv_org_name: {tv_name}"}
    ]
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content), response.usage.total_tokens

# Function to build Cartesian pairs of Compustat and TVL names
# This option is useful when you want to compare all combinations of names from both datasets, 
# especially when you don't have paired data or when you want to explore potential matches exhaustively.
# Note: This can be computationally expensive, so it's advisable to pre-filter the datasets to reduce the number of pairs.
# For example, you might filter by industry, country, or other criteria to limit the number of combinations.
def build_cartesian_pairs(n_comp: int = 150, n_tvl: int = 150) -> pd.DataFrame:
    """
    Return a de-duplicated Cartesian grid of Compustat vs TVL names.
    ATTENTION!!! It's advisable to pre-filter the datasets to avoid excessive costs - such as by industry, country, 
    embbedding comparison, fuzzy matching, etc.
    """

    # Load—and keep only the first n unique names
    comp = (
        pd.read_csv(COMP_FILE)
        .drop_duplicates("conm")        # avoid repeating identical CONM
        .head(n_comp)                   # sample cap
    )
    tvl = (
        pd.read_csv(TVL_FILE)
        .drop_duplicates("tv_org_name") # avoid repeating identical TVL name
        .head(n_tvl)
    )

    # Sanity check on required columns
    if "conm" not in comp.columns or "tv_org_name" not in tvl.columns:
        raise ValueError("Both datasets must contain 'conm' and 'tv_org_name' columns.")

    # Cartesian product (without repetition)
    comp["key"] = 1
    tvl["key"]  = 1
    grid = comp.merge(tvl, on="key", suffixes=("_comp", "_tvl")).drop(columns="key")

    logging.warning(
        "Cartesian pairs (deduplicated): %s Compustat × %s TVL → %s rows",
        len(comp), len(tvl), len(grid)
    )
    logging.warning("Down-stream GPT cost still grows quadratically; pre-filter if possible.")

    return grid

# Function to match pairs using GPT-4o-mini
# This function uses a thread pool to parallelize the matching process, 
# which can significantly speed up the operation, especially when dealing with a large number of pairs.
def match_with_gpt(pairs_df):
    results = []
    token_window = 0
    total_tokens = 0
    t0 = time.time()

    meta_cols = [c for c in pairs_df.columns if c not in {"conm", "tv_org_name"}]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        fut_map = {
            pool.submit(ask_llm, row["conm"], row["tv_org_name"]): idx
            for idx, row in pairs_df.iterrows()
        }

        for fut in as_completed(fut_map):
            idx = fut_map[fut]
            result, tokens = fut.result()
            token_window += tokens
            total_tokens += tokens

            if token_window > TOKEN_LIMIT:
                elapsed = time.time() - t0
                if elapsed < 60:
                    time.sleep(60 - elapsed)
                t0 = time.time()
                token_window = tokens

            row_out = [pairs_df.iloc[idx][c] for c in meta_cols] + [
                pairs_df.iloc[idx]["conm"],
                pairs_df.iloc[idx]["tv_org_name"],
                result.get("Answer", "Err"),
                result.get("Explanation", "")
            ]
            results.append(row_out)

    out_cols = meta_cols + ["conm", "tv_org_name", "Answer", "Explanation"]
    logging.info("GPT Matching complete. Total tokens used: %s", total_tokens)
    return pd.DataFrame(results, columns=out_cols)

# Main function to orchestrate the matching process
def main():
    logging.info("Step 1: Build Cartesian candidate set")
    pairs_df = build_cartesian_pairs()

    logging.info("Step 2: Match each pair using GPT-4o-mini")
    matched_df = match_with_gpt(pairs_df)

    logging.info("Step 3: Save results to CSV")
    matched_df.to_csv(OUTPUT_CSV, index=False)
    logging.info("Finished. Full results saved to %s", OUTPUT_CSV)
    

# Console summary
if __name__ == "__main__":
    main()
