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
    train = pd.read_csv('/Users/onurerkinsucu/Dev/prohack/data/processed/train_column_names_fixed.csv')
    X_test = pd.read_csv('/Users/onurerkinsucu/Dev/prohack/data/processed/test_column_names_fixed.csv')
    X_test = X_test.drop(['galactic_year'], axis=1)
    # region cleaning
    y = train['y']
    train = train.drop(['y', 'galactic_year'], axis=1)
    train['galaxy'] = train['galaxy'].astype('category')
    X_test['galaxy'] = X_test['galaxy'].astype('category')
    # endregion

    # na_50_percent = ['gross_capital_formation_percentage_of_ggp',
    #                  'population_total_millions', 'population_urban_percentage',
    #                  'mortality_rate_under_five_per_1000_live_births',
    #                  'mortality_rate_infant_per_1000_live_births',
    #                  'old_age_dependency_ratio_old_age_65_and_older_per_100_creatures_ages_15_64',
    #                  'population_ages_15–64_millions',
    #                  'population_ages_65_and_older_millions',
    #                  'life_expectancy_at_birth_male_galactic_years',
    #                  'life_expectancy_at_birth_female_galactic_years',
    #                  'population_under_age_5_millions',
    #                  'young_age_0_14_dependency_ratio_per_100_creatures_ages_15_64',
    #                  'adolescent_birth_rate_births_per_1000_female_creatures_ages_15_19',
    #                  'total_unemployment_rate_female_to_male_ratio',
    #                  'vulnerable_employment_percentage_of_total_employment',
    #                  'unemployment_total_percentage_of_labour_force',
    #                  'employment_in_agriculture_percentage_of_total_employment',
    #                  'labour_force_participation_rate_percentage_ages_15_and_older',
    #                  'labour_force_participation_rate_percentage_ages_15_and_older_female',
    #                  'employment_in_services_percentage_of_total_employment',
    #                  'labour_force_participation_rate_percentage_ages_15_and_older_male',
    #                  'employment_to_population_ratio_percentage_ages_15_and_older',
    #                  'jungle_area_percentage_of_total_land_area',
    #                  'share_of_employment_in_nonagriculture_female_percentage_of_total_employment_in_nonagriculture',
    #                  'youth_unemployment_rate_female_to_male_ratio',
    #                  'unemployment_youth_percentage_ages_15–24',
    #                  'mortality_rate_female_grown_up_per_1000_people',
    #                  'mortality_rate_male_grown_up_per_1000_people',
    #                  'infants_lacking_immunization_red_hot_disease_percentage_of_one_galactic_year_olds',
    #                  'infants_lacking_immunization_combination_vaccine_percentage_of_one_galactic_year_olds',
    #                  'gross_galactic_product_ggp_per_capita',
    #                  'gross_galactic_product_ggp_total',
    #                  'outer_galaxies_direct_investment_net_inflows_percentage_of_ggp',
    #                  'exports_and_imports_percentage_of_ggp',
    #                  'share_of_seats_in_senate_percentage_held_by_female',
    #                  'natural_resource_depletion',
    #                  'mean_years_of_education_female_galactic_years',
    #                  'mean_years_of_education_male_galactic_years',
    #                  'expected_years_of_education_female_galactic_years',
    #                  'expected_years_of_education_male_galactic_years',
    #                  'maternal_mortality_ratio_deaths_per_100000_live_births',
    #                  'renewable_energy_consumption_percentage_of_total_final_energy_consumption',
    #                  'estimated_gross_galactic_income_per_capita_male',
    #                  'estimated_gross_galactic_income_per_capita_female',
    #                  'rural_population_with_access_to_electricity_percentage',
    #                  'domestic_credit_provided_by_financial_sector_percentage_of_ggp',
    #                  'population_with_at_least_some_secondary_education_female_percentage_ages_25_and_older',
    #                  'population_with_at_least_some_secondary_education_male_percentage_ages_25_and_older',
    #                  'gross_fixed_capital_formation_percentage_of_ggp',
    #                  'remittances_inflows_percentage_of_ggp',
    #                  'population_with_at_least_some_secondary_education_percentage_ages_25_and_older',
    #                  'intergalactic_inbound_tourists_thousands',
    #                  'gross_enrolment_ratio_primary_percentage_of_primary_under_age_population',
    #                  'respiratory_disease_incidence_per_100000_people',
    #                  'interstellar_phone_subscriptions_per_100_people',
    #                  'interstellar_data_net_users_total_percentage_of_population',
    #                  'current_health_expenditure_percentage_of_ggp',
    #                  'intergalactic_development_index_idi_female',
    #                  'intergalactic_development_index_idi_male',
    #                  'gender_development_index_gdi',
    #                  'intergalactic_development_index_idi_female_rank',
    #                  'intergalactic_development_index_idi_male_rank', 'adjusted_net_savings',
    #                  'creature_immunodeficiency_disease_prevalence_adult_percentage_ages_15_49_total',
    #                  'private_galaxy_capital_flows_percentage_of_ggp',
    #                  'gender_inequality_index_gii']
    #
    # train = train.drop(na_50_percent, axis=1)

    # region Dataset Creation
    X_train, X_val, y_train, y_val = train_test_split(train, y, test_size=0.1, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # Create Dataset
    ds = Dataset(X_train,y_train,X_val,y_val,X_test)



    if impute_or_not:

    # region impute
         ds = impute_numeric_columns(ds)
    
    
    if standardize_or_not:
        ds = standardize(ds)

    return ds