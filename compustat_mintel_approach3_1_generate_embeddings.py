import os, json, logging
from pathlib import Path
from typing import List

import openai
import pandas as pd
from tqdm import tqdm

# Set relative paths
BASE      = Path(__file__).resolve().parent
DATA_DIR  = BASE / "data"
OUT_DIR  = BASE / "outputs"

# File paths
MINTEL_CSV     = DATA_DIR / "mintel_sample.csv"
COMPUSTAT_CSV  = DATA_DIR / "compustat_sample.csv"
MINTEL_OUT     = OUT_DIR / "mintel_embeddings.parquet"
COMPUSTAT_OUT  = OUT_DIR / "compustat_embeddings.parquet"

# Debug logging setup 
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

#  OpenAI key 
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise EnvironmentError("OPENAI_API_KEY not set.")

# Helper: batching embeddings
MODEL       = "text-embedding-3-large"
BATCH_SIZE  = 200          # adjust to taste

def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    resp = openai.embeddings.create(model=MODEL, input=texts)
    return [o.embedding for o in resp.data]

def add_embeddings(df: pd.DataFrame, txt_col: str) -> pd.DataFrame:
    texts = df[txt_col].tolist()
    embeds = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE),
                  desc=f"Embedding {txt_col}",
                  unit_scale=True):
        chunk = texts[i : i + BATCH_SIZE]
        embeds.extend(get_embeddings_batch(chunk))

    df["embedding"] = embeds
    return df

#  Mintel 
mintel = pd.read_csv(MINTEL_CSV)

# Build embedding string: COMPANY | ULTIMATE_COMPANY
mintel["embedding_info"] = (
    mintel["company"].astype(str).str.strip().fillna("") + " | " +
    mintel["ultimate_company"].astype(str).str.strip().fillna("")
)

logging.info("Mintel rows: %s", len(mintel))
mintel = add_embeddings(mintel, "embedding_info")
mintel.to_parquet(MINTEL_OUT, index=False)
logging.info("Mintel embeddings saved → %s", MINTEL_OUT)

# Compustat
comp = pd.read_csv(COMPUSTAT_CSV)

comp["embedding_info"] = comp["conm"].astype(str).str.strip()
logging.info("Compustat rows: %s", len(comp))

comp = add_embeddings(comp, "embedding_info")
comp.to_parquet(COMPUSTAT_OUT, index=False)
logging.info("Compustat embeddings saved → %s", COMPUSTAT_OUT)
