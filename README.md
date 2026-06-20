# IoT Sensor Data Trend Prediction — Aquaponics Fish Pond Water Quality

End-to-end pipeline that ingests, cleans, engineers features for, models,
and evaluates real industrial IoT sensor data from an aquaponics fish farm.

## 1. Industrial context & target variable

**Domain:** Aquaculture environmental tracking. An ESP-32 microcontroller
with six water-quality probes (temperature, turbidity, dissolved oxygen,
pH, ammonia, nitrate) continuously monitors a freshwater catfish/tilapia
aquaponics pond and uploads readings to the cloud (ThingSpeak), alongside
manual fortnightly measurements of fish length and weight.

**Dataset:** [Sensor Based Aquaponics Fish Pond Datasets](https://www.kaggle.com/datasets/ogbuokiriblessing/sensor-based-aquaponics-fish-pond-datasets)
(Kaggle) — 12 separate pond files, `IoT_pond1.csv` .. `IoT_pond12.csv`.
Download any one and place it at `data/raw/IoT_pond1.csv` (or point
`RAW_FILE` in `utils.py` at a different pond). **If no real file is
present, `01_data_ingestion.py` auto-generates a synthetic dataset with the
same six sensors and the same categories of real-world defects**, so the
pipeline is fully testable without the Kaggle download. Swap in the real
CSV before your final run/demo.

**Target variable:** `dissolved_oxygen_mgl`, forecast `HORIZON_STEPS`
(default 6) steps ahead — roughly 30 minutes at the dataset's typical
~5-minute logging cadence. Dissolved Oxygen (DO) is the most safety-critical
water-quality parameter for fish survival; published aquaculture guidance
flags risk to fish health below ~3 mg/L. A 30-minute-ahead DO forecast is
directly actionable (time to turn on an aerator) and is therefore a more
meaningful "system failure" framing than a 1-step nowcast of a slow-moving
sensor.

## 2. Two ways to run it

- **Single Colab/Jupyter notebook** — `notebooks/iot_aquaponics_pipeline.ipynb`
  contains the entire pipeline (all 5 phases) as one self-contained
  notebook with no multi-file imports. Open it in Google Colab, optionally
  upload a real `IoT_pondN.csv`, and run all cells top to bottom. This is
  the easiest option for the demo video, since you narrate phase by phase
  as each cell runs.
- **Modular scripts** — `src/01_...` through `src/05_...`, for a
  "production pipeline" repo structure with distinct analytical phases as
  separate files.

Both implement identical logic.
```bash
pip install -r requirements.txt
cd src
python run_all.py
```

## 3. Pipeline structure (modular scripts)

```
src/
  utils.py                  paths, canonical schema, robust column auto-detection, sampling-rate inference
  01_data_ingestion.py      load a real pond CSV, or generate a realistic synthetic one
  02_data_cleaning.py       per-reading filters, resample to a regular grid, gap-aware interpolation
  03_feature_engineering.py lags, rolling windows, ammonia x temperature interaction, time features, target
  04_train_model.py         chronological split, baseline, Random Forest, neural net, DO-risk recall check
  05_evaluate_and_plot.py   metrics table + actual-vs-predicted (with risk threshold) / residual / importance plots
  run_all.py                runs all five stages in order
outputs/   model_metrics.json, metrics_table.csv, test_predictions.csv, feature_importance.csv, plots/, models/
```

## 4. Data cleaning — methodology and justification

The 12 pond files differ slightly in header text and the nominal "every 5
seconds" upload rate is rarely hit exactly by WiFi-connected field
hardware, so this pipeline is built to adapt rather than assume:

1. **Column auto-detection by keyword**, not hardcoded header strings —
   `IoT_pond3.csv` might spell a header slightly differently than
   `IoT_pond7.csv`; matching by keyword (`"oxygen"`, `"turb"`, `"ammonia"`,
   ...) means the same code runs on any of the 12 files unmodified.
2. **Sampling interval inferred from the data itself** (mode of the actual
   timestamp gaps) instead of trusting the "5 seconds" spec sheet value,
   since real transmission intervals drift with WiFi conditions.
3. **Physical-plausibility rules per sensor**: pH outside 0–14, dissolved
   oxygen above 20 mg/L (fresh water physically cannot hold more), or
   negative turbidity/ammonia/nitrate are impossible for the hardware, not
   just "unusual" — discarded before any smoothing.
4. **Rolling median + MAD outlier detection**, tuned with a deliberately
   permissive threshold. A genuine ammonia spike right after a feeding
   event is a *real* signal, not sensor noise — too aggressive an outlier
   filter would erase a biologically meaningful event. This is a judgment
   call worth narrating explicitly in the demo video.
5. **Resample (bin) onto a regular time grid, not reindex.** This is the
   one technical detail that actually matters here: jittered, irregularly
   spaced real timestamps will almost never exactly match a synthetic
   regular grid, so naively reindexing onto one would silently discard
   nearly all real readings. Binning (`resample().mean()`) groups whatever
   readings actually fall in each interval and correctly leaves genuinely
   empty intervals as NaN — that NaN is the real missing-timestamp signal.
6. **Gap-aware interpolation** for the six sensors: short gaps are linearly
   time-interpolated; gaps too long to safely interpolate are dropped
   rather than fabricated.
7. **Fish length/weight are handled completely differently** — they are
   only measured by hand every ~2 weeks, so most rows are legitimately
   "missing" by design, not a sensor fault. Forward-filling the last known
   measurement is appropriate since growth is slow and monotonic; treating
   them with the same gap-interpolation or spike-detection logic as the
   continuous sensors would be meaningless, so they're excluded from
   modeling features entirely (see Section 6).

## 5. Feature engineering — rationale

- **Lag features** (t-1, t-2, t-3, t-6, t-12) for all six sensors: water
  chemistry is strongly autocorrelated minute-to-minute.
- **Rolling mean/std (1h, 3h windows)** for all six sensors: a DO level
  that has been steadily falling for an hour is a meaningfully different
  situation from one that just dipped once.
- **Ammonia × Temperature interaction**: the toxic, un-ionized fraction of
  ammonia rises sharply with both pH and temperature, so this product is a
  more biologically meaningful stress indicator than either raw sensor
  alone — a direct nod to domain knowledge rather than a generic feature.
- **Cyclical hour-of-day (sin/cos)**: DO and temperature both follow a
  diurnal cycle (DO drops overnight from algal respiration without
  photosynthesis); sin/cos avoids a 23:50→00:00 discontinuity.
- **Target built by shifting the target column backward**
  (`shift(-HORIZON_STEPS)`) rather than shifting features forward, so every
  feature column only ever uses information available at prediction time.

## 6. Model architecture & overfitting guards

- **Naive persistence baseline** — predict "no change"; the floor any real
  model must beat.
- **Random Forest Regressor** — primary trend-regression model: handles
  mixed-unit features (°C, mg/L, pH, ppm) without scaling, robust to
  residual sensor noise, gives interpretable feature importances.
- **MLPRegressor** — small feed-forward neural net, included since the
  assessment explicitly allows "sequential neural networks." A lightweight
  stand-in for a recurrent model where TensorFlow/PyTorch isn't available;
  swap in a Keras/PyTorch LSTM on the same windowed features as a drop-in
  upgrade if you have either installed.
- **Fish length/weight are excluded from the feature set.** They update
  roughly every two weeks, far slower than the 30-minute forecast horizon,
  so they carry no relevant signal there and including a column that's
  constant across ~95% of consecutive rows risks the model latching onto
  it as a spurious "time period" fingerprint rather than real water
  chemistry.

**Overfitting guards (time-series specific):**
- **Chronological train(70%)/val(15%)/test(15%) split — never shuffled.**
  A random split would let the model see near-future water conditions via
  interpolated neighbors sitting right next to training rows.
- **Capacity caps on the Random Forest** (`max_depth=10`,
  `min_samples_leaf=5`) so individual trees can't memorize single noisy
  readings.
- **Early stopping on the neural net**, monitored on a held-out slice of
  the *training* period, not test data.
- **Train/val/test metrics reported side by side** — the train→val RMSE
  gap is the signal to dial back capacity, not a number to hide.

## 7. Evaluation results (synthetic test run — replace with real-pond numbers)

| Model | Test RMSE (mg/L) | Test MAE (mg/L) |
|---|---|---|
| Naive persistence baseline | 0.249 | 0.202 |
| Linear Regression | 0.152 | 0.120 |
| **Random Forest** | **0.157** | **0.124** |
| Neural Net (MLP) | 0.243 | 0.159 |

Both Random Forest and Linear Regression clearly beat the persistence
baseline (DO is harder to "no-change" guess than wind power, since it has a
strong but smooth diurnal trend that lag features capture well). The
train→val RMSE gap for the Random Forest (~0.06 mg/L on a roughly 5–8 mg/L
range) is modest given the capacity constraints. `outputs/model_metrics.json`
also reports a **low-DO risk-event recall** — the fraction of genuinely
risky (<3 mg/L) future moments the model correctly flagged in advance —
which is the most decision-relevant metric for a fish-pond early-warning
system; this becomes meaningful once run on a real pond file, since the
synthetic test window here happens not to dip below the threshold.
<img width="2096" height="726" alt="image" src="https://github.com/user-attachments/assets/bd17fb8f-6878-43f0-8f35-620dd8ad7726" />
<img width="2036" height="604" alt="image" src="https://github.com/user-attachments/assets/d974750c-e448-426b-bc89-fef74c8e5216" />
<img width="1768" height="1202" alt="image" src="https://github.com/user-attachments/assets/a5f05774-cb62-4685-a29a-177482c7e163" />




## 8. The demo video
https://drive.google.com/file/d/11dT_oyvKJe_vz_rCafgXmBTKY_qpTxIk/view?usp=sharing
