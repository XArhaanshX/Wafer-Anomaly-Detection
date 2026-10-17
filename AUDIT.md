# AUDIT: WAFER-GUARD pre-release review

**No-go.** The reported F1 of 0.99 came from a test set that was 50% defects, and 274 of its 293 defects were SMOTE points built from training wafers. With the pipeline repaired, the model gets F1 0.124 and AP 0.070 on the last two weeks of data (314 wafers, 17 defects). Flagging every wafer gets 0.103 and 0.054. Evidence for each point is in section 8 of `wafer_guard_fixed.ipynb`.

## The three silent bugs

**1. Data leakage (original cell 8).** SMOTE ran before `train_test_split`. For 92.8% of test defects, the nearest training row was a SMOTE point, at a median distance of 0.99 to 2.37. In a clean split the nearest real training wafer is 18.46 away. The imputer and scaler were also fit on all rows, and the random split let October lots predict July ones. *Fix:* I sort by `timestamp` and split 60/20/20 in time, keeping lots that share a start time together. The column filter, imputer and scaler are fit on train only, and the threshold is picked on val. SMOTE is replaced by `pos_weight`, which scored the same in CV. Fixing only the loss leaves F1 at 0.987, while moving SMOTE after the split drops it to 0.067 (8.2).

**2. Corrupted loss gradients (cells 10 and 14).** The loader gives `[B, 1, F]` and `forward` returns `[B, F]`, so `reconstructed - batch_x` broadcasts to `[64, 64, F]`. Its gradient is the same as MSE against the batch mean (max difference 2.18e-11), so the decoder learns to output the average wafer. Recon loss stays at 0.744 for all 40 epochs, and reconstructions differ across wafers by 0.004, versus 0.384 once fixed (8.1). *Fix:* I removed the channel axis and added asserts on the reconstruction and logit shapes.

**3. Invariant blindness on PS-L wafers (cell 8).** 146 wafers have `sensor_0 == 0.0`, and all of them have `sensor_1` in {100, 200, 300}. The original imputed and scaled these as numbers. That raised sensor_0's std from 74 to 881, squashing genuine variation 11.8x (8.5x for sensor_1). It also pushed the six truly missing sensor_0 values 3.8 std below the real mean, and turned the sentinel into a perfect PS-L flag at z = -3.11 (8.3). *Fix:* a fixed per-row rule blanks sensor_0 and sensor_1 on PS-L rows, and the code becomes three one-hot columns that skip imputation and scaling. The blanked cells go into the encoder as 0, the one-hot columns tell the model why, and the reconstruction loss skips them. That way no made-up value is ever used as a measurement or a target.

## Other defects

- The 0.5 threshold was never tuned. At natural prevalence it flags 1 of 314 test wafers.
- The gate used F1 on a balanced set, and the export always saved a preprocessor fit on test rows. The gate now runs on the time-ordered test set, with no export when it fails.
- 122 sensors are constant in train and 24 are at least 50% missing. Without clipping, |z| on the validation window reaches 1,836. I drop those sensors and clip to ±5, both chosen by walk-forward CV inside train + val (AP 0.045 to 0.058).

## True metrics at natural prevalence

The test window is 2008-10-02 to 2008-10-17: 314 wafers, 17 defects (5.41%). Brackets are 95% bootstrap intervals.

| | AP | ROC-AUC | F1 | precision | recall |
|---|---|---|---|---|---|
| repaired SAE (seed 1337) | 0.070 [0.042, 0.137] | 0.577 [0.444, 0.710] | 0.124 [0.067, 0.186] | 0.068 | 0.765 (13 of 17, 179 false alarms) |
| same, mean ± std over 5 seeds | 0.084 ± 0.010 | 0.618 ± 0.026 | 0.130 ± 0.013 | | |
| logistic regression | 0.088 | 0.638 | 0.117 | 0.063 | 0.765 |
| random forest | 0.077 | 0.608 | 0.145 | 0.083 | 0.588 |
| flag every wafer | 0.054 | 0.500 | 0.103 | 0.054 | 1.000 |

I didn't resample the test window to 6.6%, because defects cluster in time. Re-weighting its TPR and FPR to the dataset-wide 6.64% gives precision 0.083 and F1 0.149. AP is the main metric: ROC-AUC looks good when 94% of wafers pass, and an AP equal to the prevalence means no skill.

## Residual risks

- There are only 17 test and 11 validation defects. The ROC-AUC interval includes 0.5, and a validation ROC-AUC of 0.48 suggests drift.
- The classifier head overfits (train BCE 0.0007). Its scores aren't calibrated, and the chosen threshold is 2.2e-05.
- All 146 PS-L wafers passed, so the model can learn "PS-L means pass".
- I assumed electrical-test labels are available immediately. In production they arrive late, and training should only use labels known at the time.

## Go / no-go

No-go for Friday. The model is only slightly better than flagging every wafer, and logistic regression matches it. Before trying again, I'd want more labelled history, walk-forward retraining, and a gate on AP or recall at a fixed false-alarm budget.
