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
from src.features.impute_columns import (impute_categorical_columns,impute_numeric_columns)
from src.contracts.Dataset import Dataset
from src.features.standardize import standardize
from src.features.label_encoder import MultiColumnLabelEncoder
# endregion


def preprocess_(standardize_or_not=False,impute_or_not=False):


    # Read the data
    train = pd.read_csv('data/processed/train_column_names_fixed.csv')
    X_test = pd.read_csv('data/processed/test_column_names_fixed.csv')
    # X_test = X_test.drop(['galactic_year'], axis=1)
    # region cleaning
    y = train['y']
    # train = train.drop(['galactic_year', 'y'], axis=1)
    train = train.drop(['y'], axis=1)
    train['galaxy'] = train['galaxy'].astype('category')
    X_test['galaxy'] = X_test['galaxy'].astype('category')
    # endregion


    ### Feature Engineering ###





    # region Dataset Creation
    X_train, X_val, y_train, y_val = train_test_split(train, y, test_size=0.2, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # Create Dataset
    ds = Dataset(X_train,y_train,X_val,y_val,X_test)



    if impute_or_not:
         ds = impute_numeric_columns(ds,missing_flag=False)
         # ds.X_train.to_csv('data/interim/df_imputed.csv')

    if standardize_or_not:
        ds = standardize(ds)

    return ds





