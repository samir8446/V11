# Self-updating digital twin for Li-ion diagnostics — NASA Ames 18650 (LiCoO₂/graphite)

Engine `battwin` 1.0.0 (pure Python, no Streamlit) + Streamlit front end. Python 3.12.

## Run
```bash
pip install -r requirements.txt
pytest                                   # synthetic ground-truth suite
streamlit run app.py                     # put the parquet files in data/ or upload them
python -m battwin.cli study --master data/battery_master_data.parquet \
       --imp data/impedance_ground_truth.parquet --out results/study     # offline cohort study
python -m battwin.cli study --synthetic --out results/synthetic          # no data needed
```
Streamlit Community Cloud: main file `app.py`. Without data files the app opens on a synthetic demo fleet.
Paths can also be set with `BATTWIN_MASTER` / `BATTWIN_IMP`.

## Layout
```
battwin/
  data/         registry (34 cells, groups), io (dtype normalisation), quality (Hampel, robust BOL,
                crash flag, k-consecutive EOL), soh (rate normalisation), cycles (CycleTable), synthetic
  diagnostics/  indicators (monotonicity/trendability/prognosability/LOCO, one-or-several), pca,
                stress (regression, VIF, confounding, power), icadva, halfcell (LLI/LAM_PE/LAM_NE)
  observers/    ECM dual-time-scale EKF, mechanistic PF, power-law PF (fleet prior), adaptive trend KF,
                hierarchical Bayes, ensemble weights, online conformal bands, Stream (cycle-by-cycle replay)
  prognostics/  metrics (unclipped R²), RUL, evaluation, update frequency, measurement informativeness
  ml/           features (no capacity inputs for SOH; ΔQ(V) for fade), 4 models, nested grouped CV, backtests
  operations/   economics, plant calibration, DP (current × replacement), scenarios/grid study
  study/        cohort run, fair comparison (common cells, per-cell Wilcoxon, Holm), coverage, answers, export
ui/             theme (stable colours/markers), charts (Plotly), blocks (hero, chart+table), cache, views/
tests/          pytest suite on synthetic ground truth (knee, cold, hot, mixed, pulsed, crash)
```
The app refuses to run against a different engine major.minor, and every cache key contains the
engine version and the data content hash.

## Key definitions
* **SOH** = rate-normalised capacity / the cell's own robust BOL. Per-level current/ambient offsets are fitted jointly
  with a flexible ageing trend (Huber IRLS). Constant-condition cells are unchanged.
* **EOL** is a single definition: rate-normalised capacity < `EOL_AH` (default 1.4 Ah) for 3 consecutive
  non-outlier cycles. Cells that never get there (e.g. B0033–B0040 stop at 1.6 Ah) are reported as
  "EOL not reachable in data". The replacement SOH in Mission 3 is a separate *decision*.
* **Groups** (edit `battwin/data/registry.py`):
  * Reference: B0005–07, B0018, B0036
  * High current: B0033/34
  * Hot: B0029–32
  * Mixed conditions: B0038–40
  * Cold: B0041–48, B0053–56
  * Pulsed load: B0025–28 — voltage-curve methods and the ECM twin are disabled
  * Corrupted logging: B0049–52

## What is and is not verified
* Verified: the engine against synthetic electrode-level ground truth. That covers rate-dependent capacity, knees, cold plating, mixed loads, pulsed loads and crashed logging.
* **Not verified here:** runs on the real NASA files. Treat the first real run as the validation.
* Known limits:
  * Half-cell OCVs are literature fits, and LAM_NE is weakly identifiable at 1C.
  * Mechanism shares are model-based attributions.
  * The economics defaults are illustrative, scaled to accelerated lab ageing.
  * With ~30 cells, the stress regression and the per-group Wilcoxon tests are low-power. Power labels are shown.
