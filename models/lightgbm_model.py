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

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

# endregion


ds = preprocess_(standardize_or_not=True, impute_or_not=True)

ds.X_train['is_train'] = 1
ds.X_val['is_train'] = 1
ds.X_test['is_train'] = 0


full_ds = ds.X_train.append(ds.X_val).reset_index(drop=True).append(ds.X_test).reset_index(drop=True)


# important_cols =['intergalactic_development_index_idi_male_rank',
#  'intergalactic_development_index_idi_rank',
#  'gender_inequality_index_gii',
#  'intergalactic_development_index_idi_male',
#  'intergalactic_development_index_idi_female',
#  'old_age_dependency_ratio_old_age_65_and_older_per_100_creatures_ages_15-64',
#  'estimated_gross_galactic_income_per_capita_male',
#  'intergalactic_development_index_idi_female_rank',
#  'intergalactic_development_index_idi',
#  'life_expectancy_at_birth_male_galactic_years',
#  'life_expectancy_at_birth_female_galactic_years',
#  'expected_years_of_education_male_galactic_years',
#  'domestic_credit_provided_by_financial_sector_percentage_of_ggp'
#                  ]
#
#
# for col in important_cols:
#     full_ds = create_features(full_ds, col)

full_ds = create_features(full_ds,'gender_inequality_index_gii')
full_ds = create_features(full_ds,'intergalactic_development_index_idi_male_rank')
full_ds = create_features(full_ds,'intergalactic_development_index_idi_rank')
full_ds = create_features(full_ds,'intergalactic_development_index_idi_female_rank')
full_ds = create_features(full_ds,'old_age_dependency_ratio_old_age_65_and_older_per_100_creatures_ages_15-64')
full_ds = create_features(full_ds,'estimated_gross_galactic_income_per_capita_male')
full_ds = create_features(full_ds,'estimated_gross_galactic_income_per_capita_female')
# full_ds = create_features(full_ds,'domestic_credit_provided_by_financial_sector_percentage_of_ggp')


# full_ds['intergalactic_development_index_idi_rank'] = np.log(full_ds['intergalactic_development_index_idi_rank']+2)
# full_ds['intergalactic_development_index_idi_male_rank'] = np.log(full_ds['intergalactic_development_index_idi_male_rank']+2)
# full_ds['estimated_gross_galactic_income_per_capita_male'] = np.log(full_ds['estimated_gross_galactic_income_per_capita_male']+2)
# full_ds['estimated_gross_galactic_income_per_capita_female'] = np.log(full_ds['estimated_gross_galactic_income_per_capita_female']+2)

print(sum(full_ds.isna().sum()))

# full_ds = full_ds.join(full_ds.groupby('galaxy')['gender_inequality_index_gii'].mean(), on='galaxy', rsuffix='_galaxy_mean')
# full_ds = full_ds.join(full_ds.groupby('galaxy')['gender_inequality_index_gii'].max(), on='galaxy', rsuffix='_galaxy_max')
# full_ds = full_ds.join(full_ds.groupby('galaxy')['gender_inequality_index_gii'].min(), on='galaxy', rsuffix='_galaxy_min')
#
#
# full_ds =full_ds.join(full_ds.groupby('galaxy')['intergalactic_development_index_idi_male_rank'].mean(), on='galaxy',
#              rsuffix='_galaxy_mean')
# full_ds =full_ds.join(full_ds.groupby('galaxy')['intergalactic_development_index_idi_rank'].mean(), on='galaxy',
#              rsuffix='_galaxy_mean')
# full_ds =full_ds.join(full_ds.groupby('galaxy')['intergalactic_development_index_idi_female_rank'].mean(), on='galaxy',
#              rsuffix='_galaxy_mean')
# full_ds =full_ds.join(
#     full_ds.groupby('galaxy')['old_age_dependency_ratio_old_age_65_and_older_per_100_creatures_ages_15-64'].mean(),
#     on='galaxy', rsuffix='_galaxy_mean')
#
#
# full_ds =full_ds.join(
#     full_ds.groupby('galaxy')['estimated_gross_galactic_income_per_capita_male'].mean(),
#     on='galaxy', rsuffix='_galaxy_mean')
#
# full_ds =full_ds.join(
#     full_ds.groupby('galaxy')['estimated_gross_galactic_income_per_capita_female'].mean(),
#     on='galaxy', rsuffix='_galaxy_mean')



full_ds['galaxy'] = full_ds['galaxy'].astype('category')

X_train = full_ds[full_ds['is_train'] == 1]
X_test = full_ds[full_ds['is_train'] == 0]

X_train = X_train.drop('is_train', axis=1)
X_test = X_test.drop('is_train', axis=1)

y_train = ds.y_train.append(ds.y_val).reset_index(drop=True)

# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=43)



ds.X_train = X_train
ds.y_train = y_train
# ds.X_val = X_val
# ds.y_val = y_val
ds.X_test = X_test

cate_cols = ['galaxy']

#
# ds.X_train = MultiColumnLabelEncoder(columns = cate_cols).transform(ds.X_train)
# ds.X_val = MultiColumnLabelEncoder(columns = cate_cols).transform(ds.X_val)
# ds.X_test = MultiColumnLabelEncoder(columns = cate_cols).transform(ds.X_test)
# selected_features=Feature_selection(ds)






# ds.X_train = ds.X_train[selected_features]
# ds.X_val = ds.X_val[selected_features]
# ds.X_test = ds.X_test[selected_features]

# MultiColumnLabelEncoder(columns = cate_cols).fit(ds.X_train)
# ds.X_train = MultiColumnLabelEncoder(columns = cate_cols).transform(ds.X_train)
# ds.X_val = MultiColumnLabelEncoder(columns = cate_cols).transform(ds.X_val)
# ds.X_test = MultiColumnLabelEncoder(columns = cate_cols).transform(ds.X_test)


# region lightgbm

d_train = lightgbm.Dataset(ds.X_train, label=ds.y_train, free_raw_data=False)
# d_val = lightgbm.Dataset(ds.X_val, label=ds.y_val, free_raw_data=False)

params = {}
params['learning_rate'] = 0.01
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'rmse'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10

model = lightgbm.train(params, d_train, num_boost_round=1000, valid_sets=None, valid_names=None, fobj=None, feval=None,
                       init_model=None, feature_name='auto', early_stopping_rounds=None,
                       categorical_feature=cate_cols,
                       evals_result=None, verbose_eval=10, learning_rates=None, keep_training_booster=False,
                       callbacks=None)



# model = lightgbm.cv(params, d_train, num_boost_round=5000, folds=None, nfold=20, stratified=False, shuffle=True, metrics=None, fobj=None, feval=None, init_model=None, feature_name='auto', categorical_feature=cate_cols, early_stopping_rounds=None, fpreproc=None, verbose_eval=10, show_stdv=False, seed=0, callbacks=None, eval_train_metric=False)

# rms = math.sqrt(mean_squared_error(ds.y_test, y_pred))

# categorical_feature=['galaxy'],
ds.X_train.columns

y_pred = pd.DataFrame(model.predict(ds.X_test), columns=['y_pred'])
X_test_pred = pd.concat([ds.X_test.reset_index(drop=True), y_pred], axis=1)

X_test_pred.to_csv('/Users/onurerkinsucu/Dev/prohack/data/processed/X_test_pred_19_may.csv', index=False)

# rms = math.sqrt(mean_squared_error(ds.y_test, y_pred))
# print(rms)


# endregion
