# wafer-guard

Audit and repair of the Fab-7 wafer anomaly detector (GDG wildcard, AI/ML task).

- `wafer_guard_fixed.ipynb` is the repaired pipeline. Section 8 has the evidence for each bug: shapes, ablations, CV.
- `AUDIT.md` is the write-up: the bugs, the fixes, the metrics and a go/no-go.
- `original/bnypi.ipynb` is the notebook as it was handed over.
- `secom_telemetry_modified.csv` is the dataset as provided.

## Setup

Python 3.12. A GPU isn't needed.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook wafer_guard_fixed.ipynb
```

Run all cells. The full notebook takes about 3 minutes on a laptop CPU. The committed outputs are the ones quoted in `AUDIT.md`. The seeds are fixed, so a CPU rerun with the pinned versions should give the same numbers. On a GPU they will differ slightly.

## Notes

- Data is sorted by `timestamp` and split 60/20/20 in time. Anything that gets fit (column filter, imputer, scaler, decision threshold) only sees rows from before the window it is applied to.
- PS-L legacy wafers are detected from the data itself (`sensor_0 == 0.0`). I didn't use the evaluator-only mask file, and it isn't in this repo.
- The model is the original supervised autoencoder, unchanged. Imbalance is handled with `pos_weight`. SMOTE only appears in the section 8 ablations.
- There are no separate tests. The shape asserts in the training loop and the evidence cells in section 8 are the checks.
- Libraries: PyTorch, scikit-learn, imbalanced-learn, pandas, numpy, joblib.
- Limitations are listed in `AUDIT.md`. The short version: with 17 defects in the test window, every metric has wide intervals.
