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
from sklearn.preprocessing import StandardScaler

from src.features.preprocess import preprocess_
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

# endregion



ds = preprocess_(standardize_or_not=True,impute_or_not=True)



cate_cols=['galaxy']
MultiColumnLabelEncoder(columns = cate_cols).fit(ds.X_train)
ds.X_train = MultiColumnLabelEncoder(columns = cate_cols).transform(ds.X_train)
ds.X_val = MultiColumnLabelEncoder(columns = cate_cols).transform(ds.X_val)
ds.X_test = MultiColumnLabelEncoder(columns = cate_cols).transform(ds.X_test)


sel = RandomForestRegressor(random_state=42, n_estimators = 100, max_depth=30)
sel.fit(ds.X_train, ds.y_train)
sel.score(ds.X_val, ds.y_val)
sel_feature=list(zip(list(ds.X_train.columns), sel.feature_importances_))
sel_feature = sorted(sel_feature, key=lambda x: x[1], reverse=True)

rms = math.sqrt(mean_squared_error(ds.y_val, sel.predict(ds.X_val)))
print(rms)
#
#
# sel_feature = sorted(sel_feature, key=lambda x: x[1], reverse=True)
#
# sel_feature

# column_names = list(ds.X_train.columns)
# # Create X_train_std
# scaler = StandardScaler()
# ds.full_dataset = ds.X_train.reset_index(drop=True).append(ds.X_val).reset_index(drop=True).append(ds.X_test).reset_index(drop=True)
# full_dataset = ds.full_dataset
# scaler = scaler.fit(full_dataset)
# ds.X_train = pd.DataFrame(scaler.transform(ds.X_train),columns=column_names)
# ds.X_val = pd.DataFrame(scaler.transform(ds.X_val),columns=column_names)
# ds.X_test = pd.DataFrame(scaler.transform(ds.X_test),columns=column_names)

# nas = ds.X_train.columns[ds.X_train.isna().sum() / len(ds.X_train) > 0.1]


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
                       init_model=None, feature_name='auto',  early_stopping_rounds=20,
                       evals_result=None, verbose_eval=True, learning_rates=None, keep_training_booster=False,
                       callbacks=None)
#categorical_feature=['galaxy'],
ds.X_train.columns

y_pred = pd.DataFrame(model.predict(ds.X_test), columns=['y_pred'])
X_test_pred = pd.concat([ds.X_test, y_pred], axis=1)

#X_test_pred.to_csv('/Users/onurerkinsucu/Dev/prohack/data/processed/X_test_pred.csv', index=False)

#rms = math.sqrt(mean_squared_error(ds.y_test, y_pred))
#print(rms)




# endregion



