import os, json, time, logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import pandas as pd


# Set relative paths
BASE        = Path(__file__).resolve().parent
DATA_DIR    = BASE / "data"
OUT_DIR     = BASE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# File paths
INPUT_FILE  = OUT_DIR / "mintel_vs_compustat_topK.csv"
OUTPUT_FILE = OUT_DIR / "mintel_vs_compustat_validated.csv"


# Debug logging setup 
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

#  OpenAI key 
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise EnvironmentError("OPENAI_API_KEY not set.")

# GPT model and throttle settings
MODEL              = "gpt-4o"
MAX_WORKERS        = 100
TOKEN_LIMIT_MIN    = 3_900_000

# System role for GPT
SYSTEM_ROLE = (
    "You are an expert in corporate structures and brand ownership. "
    "Given two entity names, decide whether they refer to the same company "
    "(including parent–subsidiary or brand relationships). "
    "Respond strictly in JSON: "
    '{ "Answer": "Yes" | "No", "Explanation": "<brief reason>" }'
)

# Validation function
def validate_pair(name1: str, name2: str) -> dict:
    """Calls GPT-4o and returns a match verdict in structured format."""
    prompt = f"Entity 1: {name1}\nEntity 2: {name2}"
    try:
        resp = openai.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_ROLE},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
        )
        raw    = resp.choices[0].message.content
        tokens = resp.usage.total_tokens
        data   = json.loads(raw)
    except Exception as e:
        logging.error("OpenAI error on (%s, %s): %s", name1, name2, e)
        data   = {"Answer": "N/A", "Explanation": str(e)}
        tokens = 0

    data.setdefault("Answer", "N/A")
    data.setdefault("Explanation", "")
    data["tokens_used"] = tokens
    return data

# Run validation on the input DataFrame
def run_validation(input_path: Path, output_path: Path):
    df = pd.read_csv(input_path)

    # Construct entityOne as: "mintel_company | ultimate_company"
    df["entityOne"] = df["mintel_company"].astype(str).str.strip() + " | " + df["ultimate_company"].astype(str).str.strip()
    df["entityTwo"] = df["compustat_conm"].astype(str).str.strip()

    if not {"entityOne", "entityTwo"}.issubset(df.columns):
        raise ValueError("Input must contain or construct 'entityOne' and 'entityTwo' columns.")

    total_tokens     = 0
    token_bucket     = 0
    minute_start     = time.time()
    results_buffer   = [None] * len(df)

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(df))) as executor:
        future_to_idx = {
            executor.submit(validate_pair, row.entityOne, row.entityTwo): i
            for i, row in df.iterrows()
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()

            total_tokens += result["tokens_used"]
            token_bucket += result["tokens_used"]
            results_buffer[idx] = {
                "Answer": result["Answer"],
                "Explanation": result["Explanation"]
            }

            if token_bucket > TOKEN_LIMIT_MIN:
                elapsed = time.time() - minute_start
                wait_time = max(0, 60 - elapsed)
                logging.info("Throttling: sleeping %.1f seconds", wait_time)
                time.sleep(wait_time)
                minute_start = time.time()
                token_bucket = 0

    # Save output
    df_out = pd.concat([df.reset_index(drop=True), pd.DataFrame(results_buffer)], axis=1)
    df_out.to_csv(output_path, index=False)
    logging.info("Validation finished → %s (rows: %d)", output_path, len(df_out))
    logging.info("Total tokens used: %d", total_tokens)

# Console summary
if __name__ == "__main__":
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found → {INPUT_FILE}")

    run_validation(INPUT_FILE, OUTPUT_FILE)
