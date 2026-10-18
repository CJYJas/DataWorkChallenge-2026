# DataWorks 2026: Source Code Submission

## Start here

This package contains source code, three trained ExtraTrees models, cleaned inputs, and executed results. It supports reservoir history matching and conditional 20-year production forecasting. This is the source-code deliverable; presentation, dashboard, video and any official forecast workbook are separate submissions.

The authoritative result set is the team's uploaded `Revised_results.zip` from its Colab run. All included result tables and JSON records are preserved byte-for-byte from that archive. Trained models retain their fitted trees and predictions; their joblib containers use lossless XZ compression to meet the upload limit. Repetitive per-case PNGs are omitted to meet the portal's 50 MB limit. Full numerical case results remain included and rerunning generates all figures.

## Five-minute review route

All paths below are relative to `verified_run/`.

| Order | File | Question answered |
|---|---|---|
| 1 | `03_Proxy/validation_summary.csv` | Why was ExtraTrees selected? |
| 2 | `04_Validation/method_summary.csv` and `05_Test/method_summary.csv` | How was the forecast method selected and evaluated? |
| 3 | `06_Field/time_hypotheses.png` | How well do model curves match observed production? |
| 4 | `06_Field/field_development_backtest.csv` and `06_Field/field_backtest_Q3_2005-01-01.png` | Where does field forecasting still struggle? |
| 5 | `06_Field/twenty_year_scenarios.png` and `06_Field/cumulative_watercut.png` | What conditional future scenarios were generated? |
| 6 | `08_Review/RUN_REVIEW.md` | What are the findings and limitations? |

No execution is required to inspect these results. Models are already present in `verified_run/03_Proxy/models/`.

## Method and interpretation

1. Audit 100 cases, static parameters, consistent dates and nonnegative production rates.
2. Train a proxy using four parameters plus elapsed time. Cases 1–70 are training, 71–85 validation and 86–100 test, following the inherited manifest. Official case assignments remain to be verified.
3. Search reservoir parameter combinations that minimize historical production mismatch. Compare equal-fluid and recent-balanced matching objectives.
4. Compare six forecasting methods on validation pseudo-fields at two forecast origins. Hide each target's future during matching and coefficient estimation. The proxy retains the full historical time range of TRAIN cases.
5. Lock the selected method before evaluating test cases. Apply history matching to the target field, retain diverse candidate parameter sets, and generate forecasts.

The selected recipe is `recent_balanced` with `hm_wor_constant_liquid`: matched parameters inform WOR coefficients, forecasts start from the last observed production, total liquid remains constant, and gas follows an estimated historical GOR. This is empirical extrapolation informed by matched parameters, not a new 20-year reservoir-simulator run.

Validation mean NRMSE is 0.027762 and test mean NRMSE is 0.027783. NRMSE uses TRAIN fluid maxima and is not an accuracy percentage. Direct WOR is almost tied on validation, so a material forecast advantage from history matching is not established. Field backtests vary by origin and show oil overprediction, particularly from 2005.

## Outputs and timing hypotheses

`07_Export_Q0/` contains 10 candidates under original query timing. `07_Export_Q3/` contains 10 candidates under a +3-month proxy-query hypothesis; original observed dates are retained. Each export contains 81 quarterly points per candidate from 2008-01-01 to 2028-01-01, including the starting point.

Both are alternative hypothesis sets, not a combined final best-10 submission. Date interpretation and the final official forecast selection remain unresolved. The source-code portal requirements do not establish an official forecast upload schema.

Oil and water rates are STB/day; gas is MSCF/day. Forecast cumulative oil/water are MMSTB and cumulative gas is Bscf. Forecast cumulative values start at zero at the forecast origin and exclude historical production. Scenario spread is not a calibrated P10/P50/P90 distribution or a reserves estimate. Do not sum alternative scenarios in Power BI.

## Environment and quick verification

The preserved models were trained with Python 3.13.16, NumPy 2.1.3, pandas 2.2.3, SciPy 1.16.3, scikit-learn 1.6.1 and joblib 1.6.0, as recorded in their manifest. Use Python 3.13 to most closely reproduce the original environment. Matplotlib 3.10.8 is pinned for plotting; its original version was not recorded. Packaging checks also use a separate Python 3.12 environment with the same numerical package pins; see `PACKAGING_CHECKS.json` for actual checks performed. Numerical differences across platforms may occur.

From this directory:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python verify_submission.py
python -m unittest test_pipeline.py
```

Do not load these models with a different scikit-learn version. The pipeline enforces that version check. Colab users can open `START_HERE_Colab.ipynb`, install dependencies, restart the session, then upload this complete submission ZIP at Step 1. Existing results can be reviewed without rerunning training.

## Reproduce all stages

Use a new output folder. Existing completed stages are intentionally not overwritten.

```bash
python pipeline.py setup --source Inputs_From_Phase4.zip --run new_run
python pipeline.py audit --run new_run
python pipeline.py train --run new_run
python pipeline.py validate --run new_run
python pipeline.py lock --run new_run
python pipeline.py test --run new_run
python pipeline.py field --run new_run
python pipeline.py export --run new_run --shift 0
python pipeline.py export --run new_run --shift 3
python summarize_results.py --run new_run
```

The folder numbering starts at `01_Audit`, followed by `03_Proxy`; there is no missing executable Step 02. Baseline diagnostics are included in the audit stage. The unchanged `pipeline.py` matches the preserved selection-lock fingerprint.

## Files and provenance

- `pipeline.py`: modelling stages and CLI.
- `START_HERE_Colab.ipynb`: interactive execution guide.
- `test_pipeline.py`: six focused tests, not a complete leakage audit.
- `verify_submission.py`: loads bundled models and checks exports.
- `summarize_results.py`: summarizes a completed run.
- `Inputs_From_Phase4.zip`: cleaned production inputs used by setup.
- `data/DATA_DICTIONARY.csv`: column descriptions and units.
- `verified_run/`: preserved results and trained models from the team's Colab run.
- `PACKAGE_PROVENANCE.json`: source archive hashes and omitted figures.
- `FILE_MANIFEST.csv`: file hashes for this package, excluding itself.
- `PACKAGING_CHECKS.json`: checks performed during final packaging.

`scripts/convert_clean_raw_case.py` is an optional raw Excel-to-Parquet utility. Install `pyarrow` and `python-calamine` separately if using it, then run `python scripts/convert_clean_raw_case.py --raw-data path/to/raw_data`. The original workbooks are not bundled. This utility is outside the modelling reproduction path and does not establish a verified raw-to-clean lineage for the supplied CSVs.

## Scientific limitations

Raw Phase 1 processing, operating-schedule comparability, date correction and instantaneous-rate interpretation remain unverified or assumed. Cumulative integration uses trapezoids under the instantaneous-rate assumption. Validation cases were reused for proxy and forecast-method selection. Test cases were previously inspected in earlier development, so the evaluation is not an untouched external test. Parameter solutions are non-unique. No independent 20-year truth or calibrated probabilities are available. Profitability and reserves have not been established.

The preserved `official_submission_ready: false` flags concern the overall competition deliverables and unresolved scientific/schema checks. They are retained honestly; packaging this source-code ZIP does not certify the entire competition submission.
