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


# endregion


def create_features(full_ds, column_name):
    full_ds = full_ds.join(full_ds.groupby('galaxy')[column_name].mean(), on='galaxy',
                           rsuffix='_galaxy_mean')
    full_ds = full_ds.join(full_ds.groupby('galaxy')[column_name].max(), on='galaxy',
                           rsuffix='_galaxy_max')
    full_ds = full_ds.join(full_ds.groupby('galaxy')[column_name].min(), on='galaxy',
                           rsuffix='_galaxy_min')

    full_ds[column_name + '_lag'] = full_ds.groupby(['galaxy'])[column_name].shift(1)
    full_ds[column_name + '_lag_2'] = full_ds.groupby(['galaxy'])[column_name].shift(2)
    full_ds[column_name + '_lag_3'] = full_ds.groupby(['galaxy'])[column_name].shift(3)
    # full_ds[column_name + '_lag_4'] = full_ds.groupby(['galaxy'])[column_name].shift(4)



    return full_ds