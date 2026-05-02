# Preterm Birth Prediction: Clinical & Environmental Features

End-to-end statistical analysis and machine learning pipeline for predicting preterm birth (gestational age \< 37 weeks) in a Georgia birth cohort, combining clinical/demographic variables with AlphaEarth satellite-derived environmental features.

------------------------------------------------------------------------

## Background

Preterm birth is a leading cause of neonatal morbidity and mortality. Most clinical risk-prediction models rely on maternal characteristics and pregnancy complications, ignoring the role of the **physical environment** in which a mother lives. This project asks whether high-dimensional satellite embeddings (AlphaEarth, 64 features extracted at a 1000 m buffer around each delivery location) can add predictive value on top of standard clinical predictors.

We compare three modeling strategies:

1.  **Logistic regression (GLM)** — interpretable baseline, with and without environmental features.
2.  **Random Forest (RF)** — non-linear benchmark on full / clinical-only / clinical+best-environmental feature sets.
3.  **DeLong & likelihood-ratio tests** — to evaluate whether environmental information contributes information beyond clinical variables.

------------------------------------------------------------------------

## Data

| Source | Description |
|--------------------------------|----------------------------------------|
| `data/EPIC_GA_status.csv` | Encounter-level metadata (date, ZIP, lat/long, score, status) |
| `data/pull.v1.csv` | Clinical/demographic variables and pregnancy outcomes |
| `outputs/Capstone_AlphaEarth_2023_1000m_extracted.csv` | 64 AlphaEarth embeddings (`A00`–`A63`) at 1000 m buffers |

**Outcome**

- `preterm` — binary, 1 if gestational age at delivery \< 37 weeks.

**Clinical predictors**

- *Demographics*: `race` (ref = White), `age`, `age_cat`, `hisp`, `hospital`, `nulliparous`
- *Risk factors*: `obesity`, `htn`, `gestht`, `diab`, `gestdiab`, `preecl_n`, `preecl_s`, `prom`, `fgr`

**Cohort size after cleaning:** 14,507 deliveries (preterm rate ≈ 11%).

> ⚠️ The raw clinical files (`EPIC_GA_status.csv`, `pull.v1.csv`) and the AlphaEarth extraction contain restricted patient information and are **not** committed to this repository. Place them under `data/` and `outputs/` respectively before running the pipeline.

------------------------------------------------------------------------

## Project structure

```         
project/
├── codes/
│   ├── 01_data_cleaning.R       # cleans + merges raw tables
│   ├── 02_exploration.R         # Table 1, plots, GA county map
│   ├── 03_glm_model.R           # logistic regression + LRTs + Table 2
│   ├── 03_RF_model.R            # ranger RF models + DeLong tests
│   └── full_analysis.Rmd        # consolidated end-to-end report
├── data/                        # raw inputs (not tracked)
├── outputs/                     # cleaned data, fitted models, CSV results
├── figures/                     # exported PNGs from EDA
├── tables/                      # gt-rendered HTML tables (Table 1 & 2)
└── README.md
```

The repo uses [`here`](https://here.r-lib.org/) for path resolution. Every script anchors itself with `here::i_am("codes/<filename>")` so the working directory does not need to be set manually.

------------------------------------------------------------------------

## How to reproduce

### 1. Install R & dependencies

R ≥ 4.5 is recommended. Install required packages:

``` r
install.packages(c(
  # data wrangling & I/O
  "tidyverse", "janitor", "skimr", "here",
  # tables & summaries
  "gtsummary", "gt",
  # plots
  "ggcorrplot", "maps",
  # modeling
  "ranger", "rsample", "pROC", "ResourceSelection", "lmtest",
  # rmarkdown
  "rmarkdown", "knitr"
))
```

### 2. Drop in the data files

```         
data/EPIC_GA_status.csv
data/pull.v1.csv
outputs/Capstone_AlphaEarth_2023_1000m_extracted.csv
```

### 3. Run the pipeline

**Option A — single consolidated report (recommended):**

Open `codes/full_analysis.Rmd` in RStudio and click **Knit**. The first knit takes a few minutes (the RF feature search loops over 64 candidate features at 500 trees each) but subsequent knits are fast thanks to chunk caching.

**Option B — run each script individually, in order:**

``` bash
Rscript codes/01_data_cleaning.R
Rscript codes/02_exploration.R
Rscript codes/03_glm_model.R
Rscript codes/03_RF_model.R
```

All scripts read from / write to the project sub-directories shown above; no output is written outside the project root.

------------------------------------------------------------------------

## Methods

| Step | Script | Output |
|--------------|--------------|--------------------------------------------|
| 1 | `01_data_cleaning.R` | `01_dataclean.Rds`, `01_dataclean.csv` |
| 2 | `02_exploration.R` | Table 1 (HTML), 4 figures (PNG) |
| 3a | `03_glm_model.R` | Three GLMs, LRTs, HL goodness-of-fit, Table 2 (HTML) |
| 3b | `03_RF_model.R` | Three RF models (`.Rds`), per-feature DeLong board (`.csv`) |

**Train/test split.** 80/20 stratified on `preterm`, `set.seed(2026)`.

**Feature search.** For each of 64 AlphaEarth features (`A00`–`A63`) we train RF on `clinical + that one feature`, compute test-set ROC, and run a DeLong test against the clinical-only baseline ROC. The feature with the highest AUC is then refit as the "selected" model.

------------------------------------------------------------------------

## Key results

### Logistic regression

- The `htn × obesity` interaction is **not** significant (LRT p = 0.57).
- Adding the top spatial feature `A59` to the clinical model **does** significantly improve fit (LRT χ² = 7.20 on 1 df, **p = 0.0073**).
- Hosmer–Lemeshow test on the spatial model: χ² = 3.35, df = 8, p = 0.91 → no evidence of poor fit.

**Table 2 highlights** (odds ratios from the spatial GLM):

| Predictor              |   OR | 95 % CI     | p-value   |
|------------------------|-----:|-------------|-----------|
| Severe preeclampsia    | 7.35 | 6.18 – 8.73 | \< 0.001  |
| Fetal growth restr.    | 4.58 | 3.87 – 5.41 | \< 0.001  |
| Prem. rupture of memb. | 4.49 | 3.91 – 5.14 | \< 0.001  |
| Pre-gestational diab.  | 3.18 | 2.42 – 4.16 | \< 0.001  |
| Chronic hypertension   | 2.94 | 2.45 – 3.52 | \< 0.001  |
| Preeclampsia (mild)    | 1.80 | 1.50 – 2.16 | \< 0.001  |
| Gestational diabetes   | 1.34 | 1.11 – 1.61 | 0.003     |
| Nulliparous            | 0.80 | 0.70 – 0.90 | \< 0.001  |
| **A59 (AlphaEarth)**   | 0.04 | 0.00 – 0.40 | **0.007** |

### Random Forest (test-set AUC)

| Model                               | \# vars |   AUC |
|-------------------------------------|--------:|------:|
| Full (clinical + all 64 AlphaEarth) |      78 | 0.760 |
| Clinical only                       |      14 | 0.771 |
| Clinical + top AlphaEarth (`A59`)   |      15 | 0.779 |

DeLong tests — no pairwise difference reaches significance:

- Clinical vs Full: Z = 1.08, **p = 0.28**
- Clinical vs Clinical + A59: Z = −1.47, **p = 0.14**

### Take-aways

1.  Severe preeclampsia, FGR, PROM, pre-gestational diabetes and chronic HTN remain the strongest clinical drivers of preterm birth, consistent with the obstetric literature.
2.  A single AlphaEarth dimension (`A59`) is a **statistically significant** addition to the logistic model, suggesting the local environment carries information not captured by clinical variables.
3.  Random Forest does not show a corresponding AUC boost — environmental signal is small in absolute terms and the tree ensemble may already be exploiting it indirectly via correlated clinical features.
4.  The clinical-only RF (AUC ≈ 0.77) is a strong, parsimonious baseline; adding all 64 environmental features hurts performance slightly, consistent with over-fitting given the low event rate.

------------------------------------------------------------------------

## Outputs you can inspect

After a full run the following deliverables will be available:

- `tables/02_table1.html` — descriptive Table 1 stratified by race
- `tables/03_table2_model.html` — final spatial GLM odds-ratio table
- `figures/02_preterm_rate_plot.png` — preterm rate by race
- `figures/02_birth_weight_dist_plot.png` — birth weight by preterm status
- `figures/02_clinical_rf_heatmap.png` — risk-factor correlation heatmap
- `figures/02_spatial_dis_map.png` — Georgia county-level spatial map
- `outputs/03_All_Features_DeLong_Results.csv` — per-feature AUC + DeLong p-value board
- `outputs/03_rf_*_model_500trees.Rds` — fitted RF objects (full / clinical / selected)

------------------------------------------------------------------------

## Reproducibility

- All randomness seeded at `set.seed(2026)` (data split + each `ranger` call).
- Session info captured at the bottom of `full_analysis.Rmd`.
- Pipeline tested under R 4.5.2 on macOS (aarch64).

------------------------------------------------------------------------

## Authors

- Hanyu Song

## Acknowledgments

- AlphaEarth satellite embeddings (Google) for environmental features.
- The clinical cohort was constructed from EPIC EHR extracts; we thank the data stewards for facilitating access under the appropriate IRB protocol.

## License

*Specify a license here (e.g. MIT) or note that the project is for academic use only and that the underlying clinical data are not redistributable.*
