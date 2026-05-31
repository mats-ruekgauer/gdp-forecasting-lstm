from pathlib import Path
import pandas as pd
import torch

BASE_DIR = Path('/Users/mats/Documents/Uni/Kursdateien/Bachelor Thesis')

# data
DATA_SERIES_RAW_DIR = BASE_DIR / 'data/raw/data series'
DATA_SERIES_PREPROCESSED_DIR = BASE_DIR / 'data/preprocessed/data series'
DATA_PATH = BASE_DIR / 'data/preprocessed/dataset.csv'
METADATA_SERIES_PATH = BASE_DIR / 'data/metadata_series.csv'
METADATA_FEATURES_PATH = BASE_DIR / 'data/metadata_features.csv'
IFO_RAW_PATH = BASE_DIR / 'data/raw/ifo forecast/ifo.csv'

# results
RESULTS_DIR = BASE_DIR / 'results'
TEX_DATA_DIR = BASE_DIR / 'thesis/data'

# forecasting
TARGET = 'GDP (QoQ % change)'
DATA_START, DATA_END = '1991-01-01', '2025-10-01'
TEST_START, TEST_END = '2018-01-01', '2025-10-01'
HORIZONS = list(range(1, 7))
ORIGINS = pd.date_range(TEST_START, TEST_END, freq='3MS') - pd.DateOffset(months=3)

# reproducibility
SEED = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")