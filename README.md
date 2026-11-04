# Questions before tests — analysis code

Code and derived result tables for a study of how much assessment is needed to predict
two-year difficulty with basic activities of daily living in independent older adults.

Six information strategies are compared by refitting every pair of models on identical
participants in three population cohorts — CHARLS (China), ELSA (England) and HRS (USA):

| Code | Information |
|---|---|
| R | Background: age, sex, education, hypertension, diabetes, heart disease, stroke |
| Q | R plus self-rated health, difficulty climbing a flight of stairs, depressive symptoms |
| E1 | Q plus grip strength |
| E2 | Q plus word recall |
| E3 | Q plus both tests |
| L | E3 plus the same measurements one wave earlier |

Models are then applied unchanged to another cohort (leave-one-source-out) and to a later
wave pair of the same survey, with discrimination, calibration and local updating
evaluated separately. The question is how much prognostic information each layer adds,
not how to build a screening rule.

## What is in this repository

| Path | Content |
|---|---|
| `scripts/` | Data preparation, statistical analysis and figure code |
| `results/` | Aggregate result tables, one version per quantity (no participant-level data) |
| `docs/` | Statistical analysis plan, causal diagram, numerical traceability |
| `data/` | Empty by design — see below |
| `pixi.toml`, `pixi.lock` | Pinned environment |
| `versions.json` | Software versions and source table row counts |

## Data are not included

The four surveys are third-party data released under their own data-use agreements and
cannot be redistributed:

| Survey | Country | Access |
|---|---|---|
| CHARLS | China | http://charls.pku.edu.cn |
| ELSA | England | https://www.elsa-project.ac.uk |
| HRS | USA | https://hrs.isr.umich.edu |
| SHARE | Europe | http://www.share-project.org |

Registered researchers obtain each survey from its distributor, or through a harmonised
gateway such as the Gateway to Global Aging Data, and place the analysis cohorts in `data/`:

```
data/cohort_CHARLS.parquet   data/cohort_ELSA.parquet   data/cohort_HRS.parquet
data/cohort_SHARE.parquet    data/cohort2_*.parquet     data/cohort_*_unknown.parquet
data/cohort_*_allages.parquet
```

Each file holds one row per participant with the harmonised variables used here: `pid`,
`base_wave`, `age`, `male`, `educ`, `htn`, `dia`, `hrt`, `str`, `shlt`, `cesd`, `cesd_z`,
`climb`, `walk`, `grip`, `memory`, `p_grip`, `p_memory`, `fall`, `wt`, `psu`, `t_adl`,
`t_ym`, `dead`, `months`, `status`, `has_grip`, `has_mem`, `has_prior`, `t_adl6`.
Harmonisation rules are described in `docs/SAP.md`.

## Environment

Python 3.12.15 with pandas 2.3.3, numpy 2.5.4, scipy 1.18.1, statsmodels 0.15.0,
scikit-learn 1.9.1 and matplotlib 3.11.2. Exact pins are in `pixi.toml` / `pixi.lock`:

```bash
pixi install
pixi run python scripts/01_main_analysis.py
```

## Analysis pipeline

Run in order. Each script reads only `data/` and writes only `results/`.

| Script | Purpose | Output |
|---|---|---|
| `01_main_analysis.py` | Paired comparison of the six strategies; apparent AUC, Brier, calibration | `main_performance.csv`, `paired_delta.csv` |
| `02_transport.py` | Leave-one-source-out transport with four updating levels | `transport.csv` |
| `03_sensitivity.py` | Prespecified sensitivity analyses | `sensitivity.csv` |
| `04_calibration_subgroups.py` | Subgroup discrimination by age, sex and education | `subgroup.csv` |
| `05_transport_calibration.py` | Decile calibration under no updating vs recalibration | `transport_calibration_deciles.csv` |
| `06_share_multicountry.py` | Transport across 13 SHARE countries | `share_multicountry.csv` |
| `07_selection_standardisation.py` | Measurement-availability diagnostic; standardised risk | `selection_diagnostic.csv`, `transport_ci.csv` |
| `08_transport_deciles.py` | Decile calibration grid for all strategies | `transport_deciles_*.csv` |
| `09_unknown_outcome_bounds.py` | Full-denominator bounds for unknown outcomes | `unknown_bounds*.csv` |
| `10_transport_samewave.py` | Two years apart within one survey, identical instruments | `transport_samewave.csv` |
| `11_layer_comparison.py` | Three transport layers on identical metrics | `transport_layers.csv` |
| `12_cross_validation.py` | Five-fold cross-validation of the primary contrast | `cv_performance.csv`, `cv_paired.csv` |
| `13_paired_bootstrap.py` | Paired differences at B = 1,000 | `paired_delta_b1000.csv` |
| `14_item_level.py` | Item-level decomposition of the questionnaire tier | `item_level_exploratory.csv` |
| `15_weighted_sensitivity.py` | Survey-weighted sensitivity, cluster bootstrap for CHARLS | `weighted_sensitivity.csv` |
| `16_concentration.py` | Where new cases fall in the predicted-risk distribution | `concentration.csv` |
| `17_calibration_and_updating.py` | Calibration under transport; paired local updating | `transport_q.csv`, `transport_q_ci.csv`, `local_updates.csv` |

Figures: `fig1_flow.py`, `fig2_increment.py`, `fig3_and_supp.py` (Figure 3 and
supplementary S2, S6, S8), `figS_panels.py`, `figS5_layers.py`, `figS7_bounds.py`.
`style_main.py` and `style_supp.py` hold the type and colour conventions and are
self-contained.

Run them from the repository root, for example `python scripts/fig1_flow.py`; each script
changes to the repository root itself and reads `results/`. The style modules pin the font
chain inside the script, because matplotlib's global settings otherwise depend on which
module was imported first in the session and the same script would render in a different
typeface.

## Calibration-in-the-large

Calibration-in-the-large is the intercept of a calibration model with the slope fixed at 1,
that is the value *a* solving `mean(expit(a + logit(p))) = mean(y)`, obtained by
one-dimensional root finding. The summary `logit(mean(y)) - mean(logit(p))` is not that
quantity and can misreport a cohort whose mean predicted risk is close to the observed
risk, so the root find is used throughout.

## Results

`results/` holds the aggregate outputs that every reported number traces to;
`docs/NUMERICAL_TRACEABILITY.csv` maps each reported value to its file and row. No
participant-level data is included.

## Documentation

| File | Content |
|---|---|
| `docs/SAP.md` | Statistical analysis plan, written before the analyses |
| `docs/DAG.md` | Causal diagram for the covariate set |
| `docs/NUMERICAL_TRACEABILITY.csv` | Each reported value: location, source file, row |
