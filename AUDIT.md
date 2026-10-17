# AUDIT: WAFER-GUARD pre-release review

**No-go.** The reported F1 of 0.99 came from a test set that was 50% defects, and 274 of its 293 defects were SMOTE points built from training wafers. With the pipeline repaired, the model gets F1 0.124 and AP 0.070 on the last two weeks (314 wafers, 17 defects). Flagging every wafer gets 0.103 and 0.054. Locations are sections of `bnypi.ipynb`, and the evidence is in section 8 of `wafer_guard_fixed.ipynb`.

## Three silent bugs, most severe first

**1. Data leakage (§2).** SMOTE ran before `train_test_split`. For 92.8% of test defects the nearest training row was a SMOTE point, at a median distance of 0.99 to 2.37, against 18.46 in a clean split. The imputer and scaler saw all rows, and the random split let October lots predict July ones. Fixing only the loss keeps F1 at 0.987, while moving SMOTE after the split drops it to 0.067 (8.2). *Fix:* I sort by `timestamp` and split 60/20/20 in time, keeping lots that share a start time together. The column filter, imputer and scaler are fit on train only, and `pos_weight` replaces SMOTE (same CV score, 8.4).

**2. Corrupted loss gradients (§3 `make_loader`, §5 loss).** The reconstructions are `[B, F]` but the batch is `[B, 1, F]`, so the difference broadcasts to `[64, 64, F]`. The gradient equals MSE against the batch mean (max difference 2.18e-11), so the decoder outputs the average wafer. Recon loss stays at 0.744 for 40 epochs, and reconstructions vary across wafers by 0.004, versus 0.384 once fixed (8.1). *Fix:* I removed the channel axis and added asserts on the reconstruction and logit shapes.

**3. Invariant blindness on PS-L wafers (§2).** 146 wafers have `sensor_0 == 0.0`, all with `sensor_1` in {100, 200, 300}. Scaled as numbers, they raised sensor_0's std from 74 to 881, squashing real variation 11.8x (8.5x for sensor_1). They also put the 6 truly missing sensor_0 values 3.8 std below the real mean, and made the sentinel a perfect PS-L flag at z = -3.11 (8.3). *Fix:* a fixed per-row rule blanks sensor_0 and sensor_1 on PS-L rows and moves the code into three one-hot columns outside imputation and scaling. Blanked cells enter the encoder as 0, and the masked reconstruction loss skips them.

## Other defects

- **0.5 threshold (§6):** at natural prevalence it flags 1 of 314 test wafers. I now use the max-F1 point on the validation window, which comes before test and has natural prevalence. F1 because the gate uses it.
- **Gate and export (§7):** the gate used F1 on a balanced set, and the export always saved a preprocessor fit on test rows. The gate now runs on the time-ordered test set, with no export on failure.
- **Dead and drifting sensors (§2):** 122 sensors are constant and 24 are at least 50% missing in train, and unclipped val |z| reaches 1,836. I drop those sensors and clip to ±5, both chosen by walk-forward CV inside train + val (AP 0.045 to 0.058).

## True metrics at natural prevalence

The test window is 2008-10-02 to 10-17: 314 wafers, 17 defects (5.41%). Brackets are 95% bootstrap intervals.

| | AP | ROC-AUC | F1 | precision | recall |
|---|---|---|---|---|---|
| repaired SAE, seed 1337 | 0.070 [0.042, 0.137] | 0.577 [0.444, 0.710] | 0.124 [0.067, 0.186] | 0.068 | 0.765 |
| 5 seeds, mean ± std | 0.084 ± 0.010 | 0.618 ± 0.026 | 0.130 ± 0.013 | | |
| logistic regression | 0.088 | 0.638 | 0.117 | 0.063 | 0.765 |
| flag every wafer | 0.054 | 0.500 | 0.103 | 0.054 | 1.000 |

I didn't resample to 6.6%, because defects cluster in time. Re-weighting TPR and FPR to 6.64% gives precision 0.083 and F1 0.149. AP is the main metric: ROC-AUC flatters when 94% of wafers pass, and an AP equal to the prevalence means no skill.

## Residual risks

- With 17 test and 11 val defects, the ROC-AUC interval includes 0.5, and a val ROC-AUC of 0.48 suggests drift.
- The head overfits (train BCE 0.0007) and isn't calibrated: the 2.2e-05 threshold flags 192 of 314 test wafers.
- All PS-L wafers passed, so the model can learn "PS-L means pass".
- I assumed labels are available immediately. In production, training has to account for label delay.

## Go / no-go

No-go. The model is only slightly better than flagging every wafer, and logistic regression matches it (random forest: AP 0.077). Next steps: more labelled history, walk-forward retraining, and a gate on AP or recall at a fixed false-alarm budget.
