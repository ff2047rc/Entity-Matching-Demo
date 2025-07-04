
# Entity‑Matching‑Demo

This repository accompanies our paper on **entity resolution across marketing-finance datasets**. We implement and compare **three approaches**—from exact‐ID linkage to probabilistic fuzzy logic and large‑language‑model (LLM)–assisted pipelines—using real‑world data from finance, ESG analytics, and product innovation.

We walk through two end‑to‑end case studies:

- **Compustat ↔ TVL (ESG Pulse)** – financial vs. sustainability sentiment (datasets share unique identifiers).
- **Compustat ↔ Mintel GNPD** – parent companies vs. branded product launches (no unique identifiers).

## Repository Layout

```text
Entity‑Matching‑Demo/
│
├── data/                               # Sample input CSV / parquet files
├── outputs/                            # Generated matches & diagnostics
│
├── compustat_tvl_approach1_exact_matching.py
├── compustat_tvl_approach2_probabilistic_matching.py
├── compustat_tvl_approach3_llms_matching.py
├── compustat_tvl_compare_matches_compustat_tvl.py
│
├── compustat_mintel_approach3_1_generate_embeddings.py
├── compustat_mintel_approach3_2_compare_embeddings_similarity.py
├── compustat_mintel_approach3_3_llms_matching.py
│
├── LICENSE
└── README.md
```

> **Quick‑start:**
>
> ```bash
> pip install -r requirements.txt
> ```

## Approach 1 · Exact matching on stable identifiers

**When to use:** Datasets share a clean, globally unique identifier (e.g., **CUSIP**, **ISIN**, **GVKEY**).

- **Script** `compustat_tvl_approach1_exact_matching.py`
- **Output** `outputs/compustat_tvl_approach1_matched_exact_by_cusip.csv`

This script standardises identifier strings, performs an inner join, drops duplicates, and logs an attrition table.

## Approach 2 · Probabilistic / fuzzy matching on names

**When to use:** No shared identifier, but names are similar enough to compare token‑wise.

- **Script** `compustat_tvl_approach2_probabilistic_matching.py`
- **Output** `outputs/compustat_tvl_approach2_matched_probabilistic_by_name.csv`
- **Key techniques** Canonicalisation → token blocking → `RapidFuzz.token_set_ratio` ≥ 80 → best‑score tie‑breaks.

The output CSV contains top‑scoring matches plus an attrition table for auditability.

## Approach 3 · LLM‑assisted semantic matching

Large language models can reason over aliases, multilingual forms, and parent–subsidiary relationships that defeat traditional fuzzy logic.

### 3A · Direct LLM match (Compustat ↔ TVL)

- **Script** `compustat_tvl_approach3_llms_matching.py`
- Creates a filtered Cartesian product of company names and asks **GPT‑4o‑mini** whether each pair refers to the same economic entity, throttling requests to stay within token budgets.

### 3B · Embedding‑first pipeline (Compustat ↔ Mintel)

1. **Generate embeddings** – `compustat_mintel_approach3_1_generate_embeddings.py`
   - Model: `text‑embedding‑3‑large`
   - Text string: "company | ultimate\_company" (Mintel) or `conm` (Compustat)
2. **Retrieve top‑K candidates via cosine similarity** – `compustat_mintel_approach3_2_compare_embeddings_similarity.py`
   - Writes `outputs/mintel_vs_compustat_topK.csv`
3. **Validate each pair with GPT‑4o** – `compustat_mintel_approach3_3_llms_matching.py`
   - Strict JSON verdict { "Answer": "Yes|No", "Explanation": "…" }
   - Produces `outputs/mintel_vs_compustat_validated.csv`

## Results summary

| Approach              | True Positives | Predicted Pairs | Coverage (Recall) | Precision |
| --------------------- | -------------- | --------------- | ----------------- | --------- |
| Exact (Gold)          | 100            | 100             | 100.00%           | 100.00%   |
| Probabilistic / Fuzzy | 86             | 90              | 86.00%            | 95.56%    |
| LLM                   | 97             | 103             | 97.00%            | 94.17%    |

## Sample data

Minimal, de‑identified slices live in `/data` so the pipelines can be run end‑to‑end out‑of‑the‑box:

* `compustat_sample.csv`
* `tvl_sample.csv`
* `mintel_sample.csv`

## Requirements

```text
pandas
numpy
scikit‑learn
rapidfuzz
tqdm
openai>=1.0.0
```

Python ≥ 3.8 is recommended. Install with `pip install -r requirements.txt` (or use a virtual‑env/conda env).

## Citation

If you use this code for academic or commercial work, please cite **"Does it Match? Diagnosing Entity Resolution Methods in Marketing & Finance"** (forthcoming) and link back to this repository.