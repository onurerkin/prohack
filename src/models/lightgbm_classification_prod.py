from src.features.load_model_ready_data import load_model_ready_data
import math
import numpy as np
import pandas as pd
import lightgbm as lgb


ds = load_model_ready_data()
cate_cols = ['galaxy']

d_train = lgb.Dataset(ds.X_train, label=ds.y_train, free_raw_data=False)
# d_val = lgb.Dataset(ds.X_val, label=ds.y_val, free_raw_data=False)

params = {}
params['learning_rate'] = 0.02
params['objective'] = 'regression'
params['metric'] = 'rmse'
params['min_data'] = 1
params['min_data_in_bin'] = 1
params['num_leaves'] = 18
params['feature_fraction'] = 0.9242398025602147
params['bagging_fraction'] = 0.5704142185125258
params['bagging_freq'] = 2
params['min_child_samples'] = 61
params['sub_feature'] = 0.5
params['max_depth'] = 20

model = lgb.train(params, d_train, num_boost_round=1090, valid_sets=None, valid_names=None, fobj=None, feval=None,
                       init_model=None, feature_name='auto', early_stopping_rounds=None,
                       categorical_feature=cate_cols,
                       evals_result=None, verbose_eval=10, learning_rates=None, keep_training_booster=False,
                       callbacks=None)
