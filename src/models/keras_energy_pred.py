from typing import Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    Input,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from src.features.label_encoder import MultiColumnLabelEncoder
from src.features.preprocess import preprocess_train_with_pred_opt
from src.features.create_features import create_features
from src.features.impute_columns import (impute_categorical_columns, impute_numeric_columns)
from src.models.dnn_classifier import train, predict
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import accuracy_score

ds = preprocess_train_with_pred_opt(standardize_or_not=True, impute_or_not=True)

ds.y_train = ds.y_train[1]
ds.y_val = ds.y_val[1]
ds.y_test = ds.y_test[1]

ds.y_train[ds.y_train == 100] = 1
ds.y_train[ds.y_train != 1] = 0

ds.y_val[ds.y_val == 100] = 1
ds.y_val[ds.y_val != 1] = 0

ds.y_test[ds.y_test == 100] = 1
ds.y_test[ds.y_test != 1] = 0

# region feature engineering

ds.X_train['is_train'] = 1
ds.X_val['is_train'] = 2
ds.X_test['is_train'] = 0

X_train = ds.X_train
X_val = ds.X_val
X_test = ds.X_test

full_ds = ds.X_train.append(ds.X_val).append(ds.X_test)
full_ds = full_ds.reset_index(drop=True)
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
            'gender_inequality_index_gii', 'y_curve_fitted']

for feature in features:
    full_ds = create_features(full_ds, feature)

full_ds = full_ds.sort_values('index')
full_ds = full_ds.drop('index', axis=1)

cate_cols = ['galaxy']

all_cols = list(full_ds.columns)
# all_cols.remove('galaxy')


X_train = full_ds[full_ds['is_train'] == 1]
X_test = full_ds[full_ds['is_train'] == 0]
X_val = full_ds[full_ds['is_train'] == 2]

X_train = X_train.drop('is_train', axis=1)
X_test = X_test.drop('is_train', axis=1)
X_val = X_val.drop('is_train', axis=1)
# endregion

cate_cols = ['galaxy']
MultiColumnLabelEncoder(columns=cate_cols).fit(X_train)

X_train = MultiColumnLabelEncoder(columns=cate_cols).transform(X_train)
X_val = MultiColumnLabelEncoder(columns=cate_cols).transform(X_val)
X_test = MultiColumnLabelEncoder(columns=cate_cols).transform(X_test)

ds.X_train = X_train.astype(np.float32)
ds.X_val = X_val.astype(np.float32)
ds.X_test = X_test.astype(np.float32)

ds = impute_numeric_columns(ds, missing_flag=False)
train_with_pred_opt = pd.read_csv('/Users/onurerkinsucu/Dev/prohack/data/interim/train_with_pred_opt.csv')

model = train(X_train=ds.X_train, y_train=ds.y_train, X_val=ds.X_val, y_val=ds.y_val,
              layers=[200, 200, 100], num_classes=2,
              cate_cols=cate_cols, learning_rate=1e-5, epochs=150, batch_size=32, dropout_rate=0.25)

test_pred = predict(model=model[1], X_test=ds.X_test, cate_cols=cate_cols)


def calculate_score(test_pred, y_true, X):
    pred_df = pd.DataFrame(np.round(test_pred), columns=['prob_0', 'class'])
    pred_df['class'] = pred_df['class'] * 100
    pred_df['y_test'] = y_true.reset_index(drop=True) * 100

    prob_df = pd.DataFrame(test_pred, columns=['prob_0', 'prob_1'])
    prob_df['y_test'] = y_true.reset_index(drop=True)

    rms = sqrt(mean_squared_error(y_true * 100, prob_df['prob_1'] * 100
                                  ))
    print("20% of the RMSE: {}".format(rms * 0.2 / 100))
    print("Accuracy: {}".format(accuracy_score(pred_df['y_test'], pred_df['class'])))

    X['energy'] = np.array(prob_df['prob_1'] * 100)
    print("Total Energy We Have: {}".format(50000 * len(X) / 890))
    print("Total Enery We Used: {}".format(sum(X['energy'])))

    y_with_index = y_true.reset_index()
    X['index'] = np.array(y_with_index['index'])
    real_indexes = list(X['index'])
    X['real_existence_expectancy_index'] = np.array(
        train_with_pred_opt.iloc[real_indexes]['existence_expectancy_index'])

    belows = X[X['real_existence_expectancy_index'] < 0.7]
    print("Total Energy has to be given to below 0.7: {}".format(5000 * len(X) / 890))
    print("Total Enery We Gave to below 0.7: {}".format(sum(belows['energy'])))


calculate_score(test_pred, ds.y_test, ds.X_test)

val_pred = predict(model=model[1], X_test=ds.X_val, cate_cols=cate_cols)

coef = 21741.573033707864 / 22770.2234909878
val_pred = val_pred * coef
calculate_score(val_pred, ds.y_val, ds.X_val)



pred_df = pd.DataFrame(np.round(test_pred), columns=['prob_0', 'class'])
pred_df['class'] = pred_df['class'] * 100
pred_df['y_test'] = ds.y_test.reset_index(drop=True) * 100

prob_df = pd.DataFrame(test_pred, columns=['prob_0', 'prob_1'])
prob_df['y_test'] = ds.y_test.reset_index(drop=True)

prob_df = prob_df.sort_values(by=['prob_1'])

rms = sqrt(mean_squared_error(ds.y_test * 100, prob_df['prob_1'] * 100
                              ))
print(rms * 0.2 / 100)


prob_df['energy'] = prob_df['prob_1'] * 100
prob_df.loc[prob_df['prob_1'] > 0.50, 'energy'] = 100
prob_df.loc[prob_df['prob_1'] < 0.50, 'energy'] = 0
rms = sqrt(mean_squared_error(ds.y_test * 100, prob_df['energy']
                              ))
print(rms * 0.2 / 100)

len(ds.y_test)
print(rms * 0.2 / 100)
print(accuracy_score(pred_df['y_test'], pred_df['class']))

ds.X_test['energy'] = np.array(prob_df['prob_1'] * 100)
sum(ds.X_test['energy'])
50000 * len(ds.X_test) / 890

y_test_with_index = ds.y_test.reset_index()
ds.X_test['index'] = np.array(y_test_with_index['index'])
real_indexes_test_set = list(ds.X_test['index'])
ds.X_test['real_existence_expectancy_index'] = np.array(
    train_with_pred_opt.iloc[real_indexes_test_set]['existence_expectancy_index'])

belows = ds.X_test[ds.X_test['real_existence_expectancy_index'] < 0.7]
sum(belows['energy'])

rms = sqrt(mean_squared_error(test_labels[1], test_pred[1]
                              )) / 100

print(rms)

test_to_optimize = ds.X_test

test_to_optimize['y_pred'] = test_pred[0]
# region optimization_function
from scipy.optimize import linprog
import pandas as pd
import numpy as np
from cvxopt import matrix
from cvxopt import glpk


def optimize_mckinsey(train):
    # Create exist_exp_index_binary(i) column
    train['exist_exp_index_binary'] = 0
    train['exist_exp_index_binary'][train['existence_expectancy_index'] < 0.7] = 1
    train['exist_exp_index_binary'] = -train['exist_exp_index_binary']

    # Create potential for increase in the index column
    train['Potential for increase in the Index'] = 0
    train['Likely increase in the Index Coefficient'] = 0
    for i in range(len(train)):
        train['Potential for increase in the Index'].iloc[i] = -np.log(train['y_pred'].iloc[i] + 0.01) + 3
        train['Likely increase in the Index Coefficient'].iloc[i] = -(
                (train['Potential for increase in the Index'].iloc[i]) ** 2 / 1000)

        # train['Likely increase in the Index Coefficient']=train['Likely increase in the Index Coefficient'].fillna(0)

    # OPTIMIZATION

    # Decision Variables
    # x(i): amount of energy to allocate for each galaxy
    # X matrix
    var_list = list(train['galaxy'])

    # Objective function
    # max the total likely increase in index
    # Objective function coefficients
    c = list(train['Likely increase in the Index Coefficient'])  # construct a cost function

    # Constraints
    # 1: sum(x(i))<=50000
    # 2: for each i x(i)<=100
    # 3: for each i x(i)>=5000*exist_exp_index_binary(i)

    # Inequality equations with upper bound, LHS-Constraints 1 and 3
    A_ineq = [np.ones(len(train)).tolist(), list(train['exist_exp_index_binary'])]

    # Inequality equations with upper bound, RHS-Constraints 1 and 3
    B_ineq = [50000 * len(train) / 890, -5000 * len(train) / 890]

    # Define bounds-Constraint #2
    bounds = list(((0, 100),) * len(train))

    # Run optimization problem
    # pass these matrices to linprog, use the method 'interior-point'. '_ub' implies the upper-bound or
    # inequality matrices and '_eq' imply the equality matrices
    res_no_bounds = linprog(c, A_ub=A_ineq, b_ub=B_ineq, bounds=bounds, method='revised simplex')
    print(res_no_bounds)

    # Create "pred_opt"
    pred_opt = res_no_bounds['x']

    # Calculate the
    increase = sum((pred_opt * train['Potential for increase in the Index'] ** 2) / 1000)
    print('Optimal increase: .{}'.format(increase))

    # Compare with the base case

    # Create a naive solution

    train_for_base = train.copy()
    ss = pd.DataFrame({
        'Index': train_for_base.index,
        'pred': train_for_base['y_pred'],
        'opt_pred': 0,
        'eei': train_for_base['existence_expectancy_index']  # So we can split into low and high EEI galaxies
    })
    # Fix opt_pred
    n_low_eei = ss.loc[ss.eei < 0.7].shape[0]
    n_high_eei = ss.loc[ss.eei >= 0.7].shape[0]
    ss.loc[ss.eei < 0.7, 'opt_pred'] = 99  # 66*99 = 6534 - >10%, <100 each
    ss.loc[ss.eei >= 0.7, 'opt_pred'] = (50000 - 99 * len(
        ss.loc[ss.eei < 0.7, 'opt_pred'])) / n_high_eei  # The rest to high eei gs
    # Leaving 5k zillion whatsits to the admin
    ss = ss.drop('eei', axis=1)

    train_for_base['potential_increase'] = -np.log(train_for_base['y_pred'] + 0.01) + 3
    increase_base = sum((ss['opt_pred'] * train_for_base['potential_increase'] ** 2) / 1000)
    print('Naive increase: .{}'.format(increase_base))

    return pred_opt


# endregion

test_energy_optimized = optimize_mckinsey(test_to_optimize)

rms = sqrt(mean_squared_error(test_labels[1], test_energy_optimized
                              )) / 100

print(rms)
