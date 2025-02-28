#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 17:54:37 2020

@author: busebalci
"""
from sklearn.metrics import mean_squared_error
from math import sqrt
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
from src.features.log_transform import create_log_transform

from src.models.dnn_regressor_copy import train, predict


ds = preprocess_(standardize_or_not=True, impute_or_not=True)

ds.X_train['is_train'] = 1
ds.X_val['is_train'] = 2
ds.X_test['is_train'] = 0

X_train = ds.X_train
X_val = ds.X_val
X_test = ds.X_test

X_train['y'] = ds.y_train
X_val['y'] = ds.y_val
X_test['y'] = np.nan

full_ds = ds.X_train.append(ds.X_val).reset_index(drop=True).append(ds.X_test).reset_index(drop=True)
full_ds = full_ds.reset_index()
full_ds = full_ds.sort_values(['galaxy', 'galactic_year'])

# import nltk
#
# nltk.download('stopwords')
# from nltk.corpus import stopwords
#
# names = full_ds['galaxy'].str.cat(sep=', ')
# stop_words = stopwords.words('english')
# punctuations = '''!()-[]{};:'’"\,<>./?@#$%^&*_~'''
# from nltk.tokenize import word_tokenize
#
# nltk.download('punkt')
# word_tokens = word_tokenize(names)
#
# filtered_names = [w for w in word_tokens if not w in stop_words and w not in punctuations]
# filtered_names = set(filtered_names)
#
# for name in filtered_names:
#     full_ds['a' + name + '_is_in_galaxy_name'] = 0
#     for i in range(len(full_ds)):
#         if name in full_ds['galaxy'].iloc[i]:
#             full_ds['a' + name + '_is_in_galaxy_name'].iloc[i] = 1

# features = ['intergalactic_development_index_idi_male_rank',
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
#  'domestic_credit_provided_by_financial_sector_percentage_of_ggp']


features = ['existence_expectancy_index',
            'existence_expectancy_at_birth', 'gross_income_per_capita',
            'income_index', 'expected_years_of_education_galactic_years',
            'mean_years_of_education_galactic_years',
            'intergalactic_development_index_idi', 'education_index',
            'intergalactic_development_index_idi_rank',
            'population_using_at_least_basic_drinking-water_services_percentage',
            'population_using_at_least_basic_sanitation_services_percentage',
            'gross_capital_formation_percentage_of_ggp',
            'population_total_millions', 'population_urban_percentage',
            'mortality_rate_under-five_per_1000_live_births',
            'mortality_rate_infant_per_1000_live_births',
            'old_age_dependency_ratio_old_age_65_and_older_per_100_creatures_ages_15-64',
            'population_ages_15_64_millions',
            'population_ages_65_and_older_millions',
            'life_expectancy_at_birth_male_galactic_years',
            'life_expectancy_at_birth_female_galactic_years',
            'population_under_age_5_millions',
            'young_age_0-14_dependency_ratio_per_100_creatures_ages_15-64',
            'adolescent_birth_rate_births_per_1000_female_creatures_ages_15-19',
            'total_unemployment_rate_female_to_male_ratio',
            'vulnerable_employment_percentage_of_total_employment',
            'unemployment_total_percentage_of_labour_force',
            'employment_in_agriculture_percentage_of_total_employment',
            'labour_force_participation_rate_percentage_ages_15_and_older',
            'labour_force_participation_rate_percentage_ages_15_and_older_female',
            'employment_in_services_percentage_of_total_employment',
            'labour_force_participation_rate_percentage_ages_15_and_older_male',
            'employment_to_population_ratio_percentage_ages_15_and_older',
            'jungle_area_percentage_of_total_land_area',
            'share_of_employment_in_nonagriculture_female_percentage_of_total_employment_in_nonagriculture',
            'youth_unemployment_rate_female_to_male_ratio',
            'unemployment_youth_percentage_ages_15_24',
            'mortality_rate_female_grown_up_per_1000_people',
            'mortality_rate_male_grown_up_per_1000_people',
            'infants_lacking_immunization_red_hot_disease_percentage_of_one-galactic_year-olds',
            'infants_lacking_immunization_combination_vaccine_percentage_of_one-galactic_year-olds',
            'gross_galactic_product_ggp_per_capita',
            'gross_galactic_product_ggp_total',
            'outer_galaxies_direct_investment_net_inflows_percentage_of_ggp',
            'exports_and_imports_percentage_of_ggp',
            'share_of_seats_in_senate_percentage_held_by_female',
            'natural_resource_depletion',
            'mean_years_of_education_female_galactic_years',
            'mean_years_of_education_male_galactic_years',
            'expected_years_of_education_female_galactic_years',
            'expected_years_of_education_male_galactic_years',
            'maternal_mortality_ratio_deaths_per_100000_live_births',
            'renewable_energy_consumption_percentage_of_total_final_energy_consumption',
            'estimated_gross_galactic_income_per_capita_male',
            'estimated_gross_galactic_income_per_capita_female',
            'rural_population_with_access_to_electricity_percentage',
            'domestic_credit_provided_by_financial_sector_percentage_of_ggp',
            'population_with_at_least_some_secondary_education_female_percentage_ages_25_and_older',
            'population_with_at_least_some_secondary_education_male_percentage_ages_25_and_older',
            'gross_fixed_capital_formation_percentage_of_ggp',
            'remittances_inflows_percentage_of_ggp',
            'population_with_at_least_some_secondary_education_percentage_ages_25_and_older',
            'intergalactic_inbound_tourists_thousands',
            'gross_enrolment_ratio_primary_percentage_of_primary_under-age_population',
            'respiratory_disease_incidence_per_100000_people',
            'interstellar_phone_subscriptions_per_100_people',
            'interstellar_data_net_users_total_percentage_of_population',
            'current_health_expenditure_percentage_of_ggp',
            'intergalactic_development_index_idi_female',
            'intergalactic_development_index_idi_male',
            'gender_development_index_gdi',
            'intergalactic_development_index_idi_female_rank',
            'intergalactic_development_index_idi_male_rank', 'adjusted_net_savings',
            'creature_immunodeficiency_disease_prevalence_adult_percentage_ages_15-49_total',
            'private_galaxy_capital_flows_percentage_of_ggp',
            'gender_inequality_index_gii']

log_features = ['gross_income_per_capita', 'mortality_rate_under-five_per_1000_live_births',
                'infants_lacking_immunization_red_hot_disease_percentage_of_one-galactic_year-olds',
                'infants_lacking_immunization_combination_vaccine_percentage_of_one-galactic_year-olds',
                'gross_galactic_product_ggp_per_capita',
                'gross_galactic_product_ggp_total',
                'share_of_seats_in_senate_percentage_held_by_female',
                'estimated_gross_galactic_income_per_capita_male',
                'estimated_gross_galactic_income_per_capita_female',
                'estimated_gross_galactic_income_per_capita_male',
                'estimated_gross_galactic_income_per_capita_female',
                'domestic_credit_provided_by_financial_sector_percentage_of_ggp']

for feature in features:
    full_ds = create_features(full_ds, feature)

for feature in log_features:
    full_ds = create_log_transform(full_ds, feature)

full_ds = full_ds.sort_values('index')
full_ds = full_ds.drop('index', axis=1)

full_ds = full_ds.drop(['y'], axis=1)

cate_cols = ['galaxy']

MultiColumnLabelEncoder(columns = cate_cols).fit(full_ds)
full_ds = MultiColumnLabelEncoder(columns = cate_cols).transform(full_ds)


all_cols = list(full_ds.columns)
# all_cols.remove('galaxy')

full_ds[all_cols] = full_ds[all_cols].astype(np.float32)


X_train = full_ds[full_ds['is_train'] == 1]
X_test = full_ds[full_ds['is_train'] == 0]
X_val = full_ds[full_ds['is_train'] == 2]

X_train = X_train.drop('is_train', axis=1)
X_test = X_test.drop('is_train', axis=1)
X_val = X_val.drop('is_train', axis=1)


# y_train = ds.y_train.append(ds.y_val).reset_index(drop=True)
# y_train = ds.y_train.reset_index(drop=True)

ds.X_train = X_train
ds.X_val = X_val
ds.X_test = X_test
# ds.y_train = y_train

ds = impute_numeric_columns(ds,missing_flag=False)

ds.X_train = ds.X_train.astype(np.float32)
ds.X_val = ds.X_val.astype(np.float32)
ds.X_test = ds.X_test.astype(np.float32)

(history,model) = train(X_train=ds.X_train, y_train=np.array(ds.y_train), X_val=ds.X_val, y_val=np.array(ds.y_val), layers=[16, 32,16], num_classes=1,
              cate_cols=cate_cols,learning_rate=1e-4, epochs=1000,batch_size=32,dropout_rate=0.1)



####RESULTSSSS

full_data=ds.X_val.copy()

#Get y_pred from dnn
cate_cols = ['galaxy']
y_pred=predict(model, X_test=full_data, cate_cols=cate_cols)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])
y_pred=np.array(y_pred)
#Define y_true

y_true=ds.y_val
y_true=np.array(y_true)

#Get y_opt_true and y_opt_pred
full_data['y_pred']=y_true 
y_opt_true=optimize_mckinsey(full_data)
full_data['y_pred']=y_pred
y_opt_pred=optimize_mckinsey(full_data)


loss1=0.8*sqrt(mean_squared_error(y_true,y_pred))
loss2=0.2*0.01*sqrt(mean_squared_error(y_opt_true,y_opt_pred))                                

    
total_loss=loss1+loss2

y_pred = pd.DataFrame(y_pred,columns=['y_pred'])
y_true=pd.DataFrame(y_true,columns=['y_true'])
y_opt_true=pd.DataFrame(y_opt_true,columns=['y_opt_true'])
y_opt_pred=pd.DataFrame(y_opt_pred,columns=['y_opt_pred'])

frames=[ y_true,y_pred, y_opt_true, y_opt_pred]
result = pd.concat(frames, axis=1)

num_mistakes=len(result[result['y_opt_true']!=result['y_opt_pred']])


