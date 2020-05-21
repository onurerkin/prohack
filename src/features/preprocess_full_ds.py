# region imports
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, cross_val_predict
import lightgbm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from src.features.impute_columns import (impute_categorical_columns, impute_numeric_columns)
from src.contracts.Dataset import Dataset
from src.features.standardize import standardize
from src.features.label_encoder import MultiColumnLabelEncoder
from src.features.impute_columns_full import impute_numeric_columns_full_ds
from src.features.standardize_full_ds import standardize_full_ds
from src.features.create_features import create_features
from src.features.create_features_full import create_features_full



# endregion


def preprocess_full_ds(standardize_or_not=False, impute_or_not=False):
    # Read the data
    train = pd.read_csv('data/processed/train_column_names_fixed.csv')
    X_test = pd.read_csv('data/processed/test_column_names_fixed.csv')
    X_test['y'] = np.nan
    train['is_train'] = 1
    X_test['is_train'] = 0

    full_ds = train.append(X_test)
    full_ds = full_ds.reset_index(drop=True)

    full_ds['galaxy'] = full_ds['galaxy'].astype('category')
    # endregion

    # Create Dataset
    ds = Dataset(full_dataset=full_ds)

    if impute_or_not:
        ds = impute_numeric_columns_full_ds(ds, missing_flag=False)
        # ds.X_train.to_csv('data/interim/df_imputed.csv')

    if standardize_or_not:
        ds = standardize_full_ds(ds)


    ### Create features
    full_ds = ds.full_dataset
    full_ds = full_ds.reset_index()
    full_ds = full_ds.rename(columns={'index':'original_index'})
    full_ds = full_ds.sort_values(by=['galaxy','galactic_year'])

    full_ds = create_features_full(full_ds, 'gender_inequality_index_gii')
    full_ds = create_features_full(full_ds, 'intergalactic_development_index_idi_male_rank')
    full_ds = create_features_full(full_ds, 'intergalactic_development_index_idi_rank')
    full_ds = create_features_full(full_ds, 'intergalactic_development_index_idi_female_rank')
    full_ds = create_features_full(full_ds, 'old_age_dependency_ratio_old_age_65_and_older_per_100_creatures_ages_15-64')
    full_ds = create_features_full(full_ds, 'estimated_gross_galactic_income_per_capita_male')
    full_ds = create_features_full(full_ds, 'estimated_gross_galactic_income_per_capita_female')

    ds.full_dataset=full_ds

    return ds





