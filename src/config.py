"""Configuration constants for the ML infant mortality project."""

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'UNICEF-CME_DF_2021_WQ-1.0-download (1).csv'
DATA_PATH_CAUSE = PROJECT_ROOT / 'data' / 'UNICEF-CME_CAUSE_OF_DEATH-1.0-download (1).csv'
TARGET = 'OBS_VALUE'

DATE_COLUMN = 'REF_DATE'
RANDOM_STATE = 42
TEST_SIZE = 0.2

CONFIG_A_FEATURES = [
    'INDICATOR',
    'SEX',
    'REF_DATE',
    'UNIT_MEASURE',
    'SERIES_NAME'
]

CONFIG_B_FEATURES = [
    'INDICATOR',
    'SEX',
    'REF_DATE',
    'UNIT_MEASURE',
    'SERIES_CATEGORY',
    'SERIES_TYPE',
    'SERIES_METHOD',
    'OBS_STATUS',
    'SERIES_YEAR',
    'INTERVAL',
    'YEAR_EXTRACT',
    'REF_DECADE'
]

CONFIG_C_FEATURES = CONFIG_B_FEATURES + [
    'LOWER_BOUND',
    'UPPER_BOUND'
]

COLUMNS_TO_DROP = [
    'WEALTH_QUINTILE',
    'Geographic area',
    'Regional group',
    'COUNTRY_NOTES',
    'CONNECTION',
    'DEATH_CATEGORY',
    'CATEGORY',
    'AGE_GROUP_OF_WOMEN',
    'TIME_SINCE_FIRST_BIRTH',
    'DEFINITION',
    'STATUS',
    'YEAR_TO_ACHIEVE',
    'Model Used'
]

NUMERIC_COLS = {
    'REF_DATE',
    'SERIES_YEAR',
    'LOWER_BOUND',
    'UPPER_BOUND',
    'INTERVAL',
    'YEAR_EXTRACT',
    'REF_DECADE'
}

MODELS = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=RANDOM_STATE),
    'Random Forest': RandomForestRegressor(max_depth=3, min_samples_leaf=20, n_estimators=20, random_state=RANDOM_STATE),
    'Gradient Boosting': GradientBoostingRegressor(max_depth=3, min_samples_leaf=20, n_estimators=20, random_state=RANDOM_STATE)
}