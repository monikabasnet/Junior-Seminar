# Executed verification

Verified November 13, 2026, using the actual local prepared UNSW-NB15 CSVs, Python 3.12 and the installed dependency versions pinned in requirements.txt.

## Passed

- Actual training, candidate comparison, calibration and official/strict test evaluation; audited run completes in approximately seven seconds including the supervised reference on this machine (runtime varies).
- Model persistence; reload produces exactly the same scores.
- Saved metrics exactly reproduce across all 82,332 testing rows.
- Feature-equivalent groups do not cross fitting/selection/calibration partitions (training assertion).
- Labels, attack categories, IDs and extra columns do not change scores.
- Missing/infinite numeric values and unseen categories are supported without refitting; training imputation statistics remain unchanged after inference on extreme values.
- Missing columns, empty inputs, negative values and nonnumeric values are rejected.
- Statistical investigation and text report generation run on actual records.
- Streamlit AppTest: all six pages execute without exceptions, both performance populations execute, and start/pause/reset replay controls update state correctly.
- All nine unittest checks passed after the technical extension.
- Local Streamlit HTTP health endpoint returned `ok`.
- Source files compile successfully.

## Limits of verification

The browser automation attempt timed out, so no pixel-level visual review or physical-browser upload/download click test is claimed. AppTest verifies runtime elements and replay state but not real browser timer cadence, mobile rendering, native file dialogs or download delivery. CSV inference and report-content generation were exercised directly; the browser file chooser was not.

These checks do not establish real-world cybersecurity effectiveness. Official test labels were already inspected during the first project iteration. The strict subset changes class/attack composition and still is not an external blind dataset. The known test-set false-alarm increase and lower strict-subset recall are explicitly documented.

## Re-run

```bash
./.venv/bin/python train_model.py
./.venv/bin/python -m unittest -v test_project.py
./.venv/bin/python -m streamlit run app.py
```

For final presentation, manually review the six pages in your browser, upload the test CSV, download both report formats, and try a CSV missing a required column. Present the overlap audit and both evaluation populations alongside headline results.


## Technical extension checks

- Passed independent metric arithmetic against hand-computable TN=3, FP=1, FN=2, TP=4 counts and zero-denominator cases.
- Passed false-positive / false-negative extraction against a known four-row example.
- Passed exact reproduction of the supervised reference metrics from its saved pipeline and calibrated threshold across all official test records.
- Passed supervised score invariance to label/category changes and verified scaler fitting sample count equals the fitting pool only.
- Confirmed original Isolation Forest metrics are exactly unchanged after adding the reference model.
- Both official and unique-input evaluation populations execute in the updated dashboard.
