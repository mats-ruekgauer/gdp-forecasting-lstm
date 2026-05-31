# Forecasting German GDP Growth with LSTM Neural Networks

**Bachelor's Thesis | University of Mannheim | Chair of Empirical Economic Research | July 2026**

## Overview

This thesis investigates whether Long Short-Term Memory (LSTM) neural networks can outperform standard econometric benchmarks and the ifo Institute's forecast for quarterly German GDP growth. Trained on a panel of 25 macroeconomic time series from 1991 Q1 onward, the LSTM is evaluated under recursive six-step-ahead forecasts across 32 rolling origins. The LSTM matches the mean and AR(1) benchmarks across all horizons and is outperformed only by the ifo forecast at the one-step-ahead horizon, an advantage that vanishes once the COVID year 2020 is excluded.

## Repository Structure

```
code/
  models/
    lstm.py              # Model definition, training, and forecasting pipeline
    benchmarks.py        # Benchmark models (mean, AR(1))
  config.py              # Project paths and global settings
  01-load data.ipynb
  02-preprocess data.ipynb
  03-hyperparameter tuning.ipynb
  04-evaluation.ipynb
data/
  raw/                   # Raw input series and raw ifo forecast
  preprocessed/          # Cleaned series and merged dataset
  metadata_series.csv
  metadata_features.csv
results/                 # Forecasts, tuning results, cleaned ifo forecast
requirements.txt
README.md
```

## Replication Scope

Full replication covers:

- Dataset construction from raw series
- Hyperparameter tuning
- Model fitting and evaluation
- Recreation of all empirical results

> **Note:** Replication of raw data loading is omitted. Original data sources with direct download links are documented in `TBD`.

## Usage

### 0. Environment Setup and Configuration

```bash
git clone https://github.com/mats-ruekgauer/gdp-forecasting-lstm.git
cd gdp-forecasting-lstm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the project root in `config.py`:

```python
DIR = "/your/local/path"
```

### 1. Load Raw Data

```bash
jupyter nbconvert --to notebook --execute "code/01-load data.ipynb"
```

- Loads raw series (including metadata) from `data/raw/`
- Exports cleaned series to `data/preprocessed/`
- Loads the raw ifo forecast from `data/raw/`
- Exports the cleaned ifo forecast to `results/`

### 2. Construct Dataset

```bash
jupyter nbconvert --to notebook --execute "code/02-preprocess data.ipynb"
```

- Loads individual series from `data/preprocessed/`
- Loads `metadata_series.csv` from `data/`
- Preprocesses series based on metadata and merges them into a single dataset
- Exports `data/preprocessed/dataset.csv`
- Creates `data/metadata_features.csv` (metadata at feature level rather than series level)
- Exports the GDP series used to generate Figure 4.1 in the thesis

### 3. Tune Hyperparameters

```bash
jupyter nbconvert --to notebook --execute "code/03-hyperparameter tuning.ipynb"
```

- Loads the LSTM from `code/models/lstm.py`, which defines the full model, parameter settings, and training/forecasting pipeline, and loads data automatically from `dataset.csv` via `torch.Dataset` / `torch.DataLoader`, applying publication lags (see thesis)
- Loads `dataset.csv`
- Fits the model across parameter combinations and origins, evaluating each forecast against the dataset
- Exports per-fit MSE to `results/tuning_result_raw.csv`
- Aggregates results over origins and exports `results/tuning_result_aggregated.csv`
- The tuned hyperparameters become the defaults in `code/models/lstm.py`

### 4. Evaluate Models

```bash
jupyter nbconvert --to notebook --execute code/04-evaluation.ipynb
```

- Loads the LSTM from `code/models/lstm.py` and defines its variants (full, without ensembling, without early stopping)
- Loads the benchmark models from `code/models/benchmarks.py`
- Loads `dataset.csv`
- Trains each model on each origin, generating a six-step-ahead forecast (see thesis)
- Aggregates results per model and exports `<model_name>_forecasts.csv` (also per seed when ensembling is used)
- Loads the ifo forecast from `results/ifo_forecast.csv`
- Aggregates results per model and horizon, excluding the first 5 observations for a consistent sample size, plus a robustness version excluding 2020
- Exports the results used to generate Figures 4.2, 4.3, and 4.4 in the thesis