# optvance-cooling-analytics

CoolingHealthSentinel pre-onboarding project: a synthetic 12-month cooling
telemetry dataset for a 30 MW data center (Doha DC-1), an XGBoost regressor
predicting a Cooling Health Score, an IsolationForest anomaly detector, an
LSTM sequence model, SHAP explainability, a 3-tier alert system with an ROI
estimate, and a small NLP classifier for sustainability regulatory text
(Track B warm-up).

## Repository Structure

```
optvance-cooling-analytics/
  README.md
  requirements.txt
  data/
    generate_dataset.py              synthetic dataset generation script
    cooling_telemetry_doha_dc1.csv   generated dataset (35,040 rows)
    regulatory_excerpts.csv          NLP task dataset (60 rows)
    README_dataset.md                dataset documentation
    feature_registry.md              all 59 engineered features documented
    features_engineered.csv          dataset + Day 3 engineered features
  notebooks/
    01_eda.ipynb                     Days 1-2: EDA
    02_feature_engineering.ipynb     Day 3: feature pipeline
    03_gbr_model.ipynb               Day 4: GBR training + evaluation
    04_isolation_forest.ipynb        Day 5: anomaly detection
    05_lstm_model.ipynb              Day 6: LSTM training + evaluation
    06_shap_analysis.ipynb           Day 7: SHAP explainability
    07_alert_roi.ipynb               Day 8: alert design + ROI
    08_nlp_regwatch.ipynb            Day 9: regulatory classifier
  models/
    gbr_baseline.pkl                 baseline GBR (12 static features)
    gbr_engineered.pkl               GBR with 71 features (12 + 59 engineered)
    isolation_forest.pkl             trained anomaly detector
    lstm_model.keras                 trained LSTM
  report/
    Nassim_Oulhadj_PreOnboarding_Report.pdf
    report.ipynb                     source notebook for the PDF report
```

## Reproducing Everything From Scratch

```bash
# 1. Environment
python -m venv optvance_env
source optvance_env/bin/activate      # Windows: optvance_env\Scripts\activate
pip install -r requirements.txt

# NOTE FOR MAC USERS: XGBoost requires the libomp system library. 
# If you are on a Mac, install it via Homebrew before running pip install:
brew install libomp

# 2. NLTK stopwords corpus, needed only for notebook 08
python -c "import nltk; nltk.download('stopwords')"

# 3. Regenerate the dataset (deterministic, seed=42; already included in
#    data/ but this confirms reproducibility from the generator alone)
cd data
python generate_dataset.py
cd ..

# 4. Run the notebooks in order
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_gbr_model.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_isolation_forest.ipynb
jupyter nbconvert --to notebook --execute --inplace 05_lstm_model.ipynb
jupyter nbconvert --to notebook --execute --inplace 06_shap_analysis.ipynb
jupyter nbconvert --to notebook --execute --inplace 07_alert_roi.ipynb
jupyter nbconvert --to notebook --execute --inplace 08_nlp_regwatch.ipynb
```

Each notebook reads its inputs from `../data/` and `../models/` and writes its
own outputs back to those same folders, so running them in the numbered order
above reproduces the full pipeline end to end, including regenerating all
four model files in `models/`.

### A note on notebook 05 (LSTM) specifically

TensorFlow's default CPU execution is not fully deterministic even with fixed
random seeds, an early version of this notebook produced different results on
different runs of identical code, including one run with a severe, unstable
failure late in the test period. This was diagnosed and fixed: the first code
cell of `05_lstm_model.ipynb` sets `PYTHONHASHSEED`, enables
`tf.config.experimental.enable_op_determinism()`, forces single-threaded
execution, and disables oneDNN's reordering optimizations
(`TF_ENABLE_ONEDNN_OPTS=0`), which a TensorFlow log message explicitly flagged
as a source of floating-point round-off differences between runs. With that
configuration in place, the notebook's full training run was independently
re-executed twice and produced bit-identical results (RMSE 5.8870 both times)
before any result in that notebook was written down. If re-running this
notebook in a different environment, keep that first cell intact, removing
those settings will likely reintroduce run-to-run variability.

### Generating the PDF report

`report/report.ipynb` is itself a notebook (markdown + code cells) that loads
the saved datasets and models and regenerates every required figure inline,
it does not just copy images from the other notebooks. To regenerate the PDF:

```bash
cd report
jupyter nbconvert --to notebook --execute --inplace report.ipynb
jupyter nbconvert --to pdf report.ipynb --output Nassim_Oulhadj_PreOnboarding_Report.pdf
```

## Key Design Decisions Worth Knowing Before Reading Further

- **The dataset's anomaly design reconciles a genuine spec conflict.**
  Section 1.2's "~3% prevalence" and Section 1.3's "8 events of 2-6 hours"
  (which caps out at ~0.55%) cannot both be literally true. Per instructor
  guidance, the dataset carries two anomaly populations: 8 "structured"
  multi-hour, multi-channel failure-precursor events, and a "background"
  layer of ~2.7% isolated single-row sensor-noise perturbations that
  deliberately leave `cooling_health_score` untouched. See
  `data/README_dataset.md` for the full reconciliation and its validation.
- **The GREEN/AMBER/RED alert thresholds from the spec produce zero RED
  alerts** over the full year, because the GBR's predictions never drop below
  66.45 even during the worst event. `notebooks/07_alert_roi.ipynb` documents
  why and works the ROI calculation both under the literal threshold ($0) and
  a recalibrated one (~$700,000/year), rather than picking one silently.
- **The LSTM underperforms the GBR**, by design intent of comparing them, not
  by oversight, both on accuracy (RMSE 5.887 vs. 2.636) and latency (57.6ms vs.
  3.4ms per prediction). `notebooks/05_lstm_model.ipynb` traces this to a clear
  overfitting signature in the training curves rather than treating it as an
  unexplained result.

## Author

Nassim Oulhadj, CoolingHealthSentinel pre-onboarding assignment, OptvanceAI Arabia.
