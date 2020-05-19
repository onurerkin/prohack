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
from src.features.preprocess import preprocess_
# endregion



ds = preprocess_(standardize_or_not=False,impute_or_not=True)


nas = ds.X_train.columns[ds.X_train.isna().sum() / len(ds.X_train) > 0.1]


# region lightgbm
d_train = lightgbm.Dataset(ds.X_train, label=ds.y_train, free_raw_data=False)
d_val = lightgbm.Dataset(ds.X_val, label=ds.y_val, free_raw_data=False)

params = {}
params['learning_rate'] = 0.01
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'rmse'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10

model = lightgbm.train(params, d_train, num_boost_round=5000, valid_sets=d_val, valid_names=None, fobj=None, feval=None,
                       init_model=None, feature_name='auto', categorical_feature=['galaxy'], early_stopping_rounds=20,
                       evals_result=None, verbose_eval=True, learning_rates=None, keep_training_booster=False,
                       callbacks=None)

ds.X_train.columns

y_pred = pd.DataFrame(model.predict(ds.X_test), columns=['y_pred'])
X_test_pred = pd.concat([ds.X_test, y_pred], axis=1)

#X_test_pred.to_csv('/Users/onurerkinsucu/Dev/prohack/data/processed/X_test_pred.csv', index=False)
#from sklearn.metrics import mean_squared_error

#rms = math.sqrt(mean_squared_error(ds.y_test, y_pred))
#print(rms)




# endregion



