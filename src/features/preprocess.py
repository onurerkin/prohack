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


def preprocess_(standardize_or_not=False, impute_or_not=False):
    # Read the data
    train = pd.read_csv('data/processed/train_column_names_fixed.csv')
    X_test = pd.read_csv('data/processed/test_column_names_fixed.csv')

    full_dataset_curve_fitted = pd.read_csv(
        '/Users/onurerkinsucu/Dev/prohack/data/interim/full_dataset_curve_fitted.csv')

    train = full_dataset_curve_fitted.loc[full_dataset_curve_fitted['is_train'] == 1].reset_index(drop=True)
    X_test = full_dataset_curve_fitted.loc[full_dataset_curve_fitted['is_train'] == 0].reset_index(drop=True)

    train = train.drop('is_train', axis=1)
    X_test = X_test.drop('is_train', axis=1)

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
    ds = Dataset(X_train, y_train, X_val, y_val, X_test)

    if impute_or_not:
        ds = impute_numeric_columns(ds, missing_flag=False)
        # ds.X_train.to_csv('data/interim/df_imputed.csv')

    if standardize_or_not:
        ds = standardize(ds)

    return ds


def preprocess_train_with_pred_opt(standardize_or_not=False, impute_or_not=False):
    # Read the data
    train_with_pred_opt = pd.read_csv('/Users/onurerkinsucu/Dev/prohack/data/interim/train_with_pred_opt.csv')
    leaking_cols = ['exist_exp_index_binary',
                    'Potential for increase in the Index',
                    'Likely increase in the Index Coefficient']
    train_with_pred_opt = train_with_pred_opt.drop(leaking_cols, axis=1)

    train_with_pred_opt['below_0_7'] = 0
    train_with_pred_opt.loc[train_with_pred_opt['existence_expectancy_index'] < 0.7, 'below_0_7'] = 1
    # y = train_with_pred_opt['y']
    # pred_opt = train_with_pred_opt['pred_opt']
    # train_with_pred_opt = train_with_pred_opt.drop(['y', 'pred_opt'], axis=1)
    # train_with_pred_opt['galaxy'] = train_with_pred_opt['galaxy'].astype('category')

    ### Feature Engineering ###
    # region Dataset Creation
    X_train, X_val = train_test_split(train_with_pred_opt, test_size=0.1)
    X_train, X_test = train_test_split(X_train, test_size=0.1)

    y_train = X_train['y']
    pred_opt_train = X_train['pred_opt']

    y_val = X_val['y']
    pred_opt_val = X_val['pred_opt']

    y_test = X_test['y']
    pred_opt_test = X_test['pred_opt']

    X_train = X_train.drop(['y', 'pred_opt'], axis=1)
    X_train['galaxy'] = X_train['galaxy'].astype('category')

    X_val = X_val.drop(['y', 'pred_opt'], axis=1)
    X_val['galaxy'] = X_val['galaxy'].astype('category')

    X_test = X_test.drop(['y', 'pred_opt'], axis=1)
    X_test['galaxy'] = X_test['galaxy'].astype('category')

    labels_train = (y_train, pred_opt_train)
    labels_val = (y_val, pred_opt_val)
    labels_test = (y_test, pred_opt_test)

    # Create Dataset
    ds = Dataset(X_train, labels_train, X_val, labels_val, X_test, labels_test)

    if impute_or_not:
        ds = impute_numeric_columns(ds, missing_flag=False)
        # ds.X_train.to_csv('data/interim/df_imputed.csv')

    if standardize_or_not:
        ds = standardize(ds)

    return ds


def preprocess_train_with_pred_opt_prod(standardize_or_not=False, impute_or_not=False):
    # Read the data
    train_with_pred_opt = pd.read_csv('/Users/onurerkinsucu/Dev/prohack/data/interim/train_with_pred_opt.csv')
    leaking_cols = ['exist_exp_index_binary',
                    'Potential for increase in the Index',
                    'Likely increase in the Index Coefficient']
    train_with_pred_opt = train_with_pred_opt.drop(leaking_cols, axis=1)
    train_with_pred_opt['below_0_7'] = 0
    train_with_pred_opt.loc[train_with_pred_opt['existence_expectancy_index'] < 0.7, 'below_0_7'] = 1

    X_test = pd.read_csv('data/processed/test_column_names_fixed.csv')
    X_test.loc[X_test['existence_expectancy_index'] < 0.7, 'below_0_7'] = 1

    # y = train_with_pred_opt['y']
    # pred_opt = train_with_pred_opt['pred_opt']
    # train_with_pred_opt = train_with_pred_opt.drop(['y', 'pred_opt'], axis=1)
    # train_with_pred_opt['galaxy'] = train_with_pred_opt['galaxy'].astype('category')

    ### Feature Engineering ###
    # region Dataset Creation

    y_train = train_with_pred_opt['y']
    pred_opt_train = train_with_pred_opt['pred_opt']

    train_with_pred_opt = train_with_pred_opt.drop(['y', 'pred_opt'], axis=1)
    train_with_pred_opt['galaxy'] = train_with_pred_opt['galaxy'].astype('category')
    X_test['galaxy'] = X_test['galaxy'].astype('category')

    labels_train = (y_train, pred_opt_train)

    print(train_with_pred_opt.shape)
    print(X_test.shape)

    # Create Dataset
    ds = Dataset(train_with_pred_opt, labels_train, X_test=X_test)

    if impute_or_not:
        ds = impute_numeric_columns(ds, missing_flag=False)
        # ds.X_train.to_csv('data/interim/df_imputed.csv')

    if standardize_or_not:
        ds = standardize(ds)

    return ds
