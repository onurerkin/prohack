# -*- coding: utf-8 -*-

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
from sklearn.preprocessing import StandardScaler
from src.features.feature_selection import Feature_selection
from src.features.preprocess import preprocess_
from src.features.create_features import create_features
from src.features.preprocess_full_ds import preprocess_full_ds
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import h2o
from h2o.automl import H2OAutoML

# endregion

ds = preprocess_full_ds(standardize_or_not=True, impute_or_not=True)
full_ds = ds.full_dataset

X_train = full_ds[full_ds['is_train'] > 0]
X_test = full_ds[full_ds['is_train'] < 0]
X_test = X_test.sort_values(by=['original_index'])

X_train = X_train.drop(['is_train', 'original_index'], axis=1)
X_test = X_test.drop(['is_train', 'original_index', 'y'], axis=1)

h2o.init()

hf = h2o.H2OFrame(X_train)
hf_test = h2o.H2OFrame(X_test)

x = hf.columns
y = 'y'
x.remove(y)

models_path = '/Users/onurerkinsucu/Dev/prohack/models/h2o_models'
aml = H2OAutoML(max_runtime_secs=1000, stopping_metric='RMSE', sort_metric='RMSE', seed=1,
                export_checkpoints_dir=models_path)
aml.train(x=x, y=y, training_frame=hf)

# View the AutoML Leaderboard
lb = aml.leaderboard
lb.head(rows=lb.nrows)

preds = aml.predict(hf_test)

preds = aml.leader.predict(hf_test)
preds = h2o.as_list(preds)

# X_test_pred = pd.concat([ds.X_test.reset_index(drop=True), y_pred], axis=1)

# X_test_pred.to_csv('/Users/onurerkinsucu/Dev/prohack/data/processed/X_test_pred_19_may.csv', index=False)

# rms = math.sqrt(mean_squared_error(ds.y_test, y_pred))
# print(rms)

# endregion

lb = h2o.automl.get_leaderboard(aml, extra_columns='ALL')

saved_model = h2o.load_model(
    '/Users/onurerkinsucu/Dev/prohack/models/h2o_models/StackedEnsemble_AllModels_AutoML_20200520_194525')
#
# preds=preds.rename(columns={'predict':'y_pred'})
# X_test_pred = pd.concat([ds.X_test.reset_index(drop=True), preds], axis=1)
#
# X_test_pred.to_csv('data/processed/X_test_pred_h2o_20_may.csv', index=False)

pred_2 = h2o.as_list(saved_model.predict(hf_test))
