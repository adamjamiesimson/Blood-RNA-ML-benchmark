# Dataset

The benchmark ships with its dataset in this folder — no separate download needed.

## GSE17048 — Whole Blood RNA Expression Profiles

- **Source:** [NCBI GEO · GSE17048](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE17048)
- **File:** `GSE17048_series_matrix.txt.gz` (GEO series matrix, gzip-compressed)

### Cohort

144 whole-blood samples:

- Healthy controls
- Relapsing-remitting MS (RRMS)
- Primary-progressive MS (PPMS)
- Secondary-progressive MS (SPMS)

### How it is used

The loader ([`msbench/data.py`](../msbench/data.py)) parses the series matrix, reads the condition of each
sample from the GEO metadata, and builds a samples × genes expression table. For the binary task the three MS
subtypes are collapsed into a single positive class:

- `Label = 0` → healthy control (HC)
- `Label = 1` → any MS subtype

Samples whose condition cannot be resolved are dropped.

### Using a different path

The default is `data/GSE17048_series_matrix.txt.gz`. Point elsewhere with:

```bash
python run_benchmark.py --data-file /path/to/GSE17048_series_matrix.txt.gz
```
