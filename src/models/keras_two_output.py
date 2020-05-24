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


# region helper functions
def _to_input_list(df: pd.DataFrame, cate_cols: list) -> list:
    """
    Coverts pandas dataframe to list of inputs for keras

    Args:
        df: pandas dataframe to be converted into list
        cate_cols: list, categorical columns list

    Returns: List, list format of the pandas dataframe

    """
    # TODO: defensive programming
    # dataframe
    df_list = []

    for c in cate_cols:
        df_list.append(df[c].values)

    # cont cols
    other_cols = [c for c in df.columns if c not in cate_cols]
    df_list.append(df[other_cols].values)

    return df_list


def _create_input_layer(
        X_train: pd.DataFrame, hidden_layers: list, cate_cols: list
) -> Tuple[list, tf.Tensor]:
    # TODO: Create the case with zero categorical columns

    """
    Creates input layer which contains categorical and numeric inputs
    Args:
        X_train: pandas DataFrame, Train features
        cate_cols: categorical columns list

    Returns: Input layer and output of input layer

    """

    input_models = []
    output_embeddings = []

    # region categorical inputs
    for categorical_var in cate_cols:
        # Name of the categorical variable that will be used in the Keras Embedding layer
        cat_emb_name = categorical_var.replace(" ", "") + "_Embedding"

        # Define the embedding_size
        no_of_unique_cat = X_train[categorical_var].nunique() + 50
        embedding_size = int(min(np.ceil((no_of_unique_cat) / 2), 50))
        # embedding_size = 200
        # One Embedding Layer for each categorical variable
        input_model = Input(shape=(1,))
        output_model = Embedding(no_of_unique_cat, embedding_size, name=cat_emb_name)(
            input_model
        )
        output_model = Reshape(target_shape=(embedding_size,))(output_model)

        # Appending all the categorical inputs
        input_models.append(input_model)

        # Appending all the embeddings
        output_embeddings.append(output_model)
    # endregion

    # region numeric inputs
    cont_cols = [item for item in list(X_train.columns) if item not in cate_cols]

    input_numeric = Input(shape=(len(cont_cols),))
    embedding_numeric = Dense(hidden_layers[0])(input_numeric)
    input_models.append(input_numeric)
    # endregion

    output_embeddings.append(embedding_numeric)
    output_of_input_layer = Concatenate()(output_embeddings)

    return input_models, output_of_input_layer


def _create_hidden_layers(
        hidden_layers: list, output_of_input_layer: tf.Tensor, dropout_rate: int
) -> tf.Tensor:
    """
    Creates hidden layers
    Args:
        hidden_layers: List of nodes in hidden layers
        output_of_input_layer: Output of input layer created by create_input_layer()
        dropout_rate: Float, dropout rate

    Returns: Output of hidden layers

    """
    # region first hidden layer

    # First hidden layer which takes output_of_input_layer as previous layer
    output = Dense(hidden_layers[0], activation="elu", kernel_initializer="he_normal")(
        output_of_input_layer
    )
    output = BatchNormalization()(output)
    output = Dropout(dropout_rate)(output)

    # Remove the first hidden layer from the list
    hidden_layers.pop(0)
    # endregion

    # region other hidden layers
    # Add remaining hidden layers
    if len(hidden_layers) > 0:
        for num_nodes in hidden_layers:
            output = Dense(num_nodes, activation="elu", kernel_initializer="he_normal")(
                output
            )
            output = BatchNormalization()(output)
            output = Dropout(dropout_rate)(output)
    # endregion

    return output


def _create_output_layer(
        num_classes: int, output_of_hidden_layers: tf.Tensor
) -> tf.Tensor:
    # TODO: Create the output for regression
    """
    Creates output layer
    Args:
        num_classes: Number of classes in target variable
        output_of_hidden_layers: output of the hidden layers which is created by create_hidden_layers()

    Returns: output layer of the model

    """

    def y_sigmoid(x):
        return (K.sigmoid(x) * 0.7)

    def energy_sigmoid(x):
        return (K.sigmoid(x) * 100)

    output_1 = Dense(32, activation='relu')(output_of_hidden_layers)
    output_1 = Dense(32, activation='relu')(output_1)
    output_1 = Dense(1, activation=y_sigmoid)(output_1)
    output_2 = Dense(32, activation='relu')(output_of_hidden_layers)
    output_2 = Dense(32, activation='relu')(output_2)
    output_2 = Dense(1, activation=energy_sigmoid)(output_2)
    return [output_1, output_2]


def _create_keras_model(
        X_train: pd.DataFrame,
        layers: list,
        num_classes: int,
        dropout_rate: float,
        cate_cols: list,
) -> tf.keras.Model:
    """
    Creates the model
    Args:
        X_train: pandas DataFrame, Train features
        layers:  List of nodes in hidden layers
        num_classes: Number of classes in target variable
        dropout_rate: dropout rate
        cate_cols: categorical columns list

    Returns: keras model

    """

    input_models, output_of_input_layer = _create_input_layer(
        X_train=X_train, hidden_layers=layers, cate_cols=cate_cols
    )
    output = _create_hidden_layers(
        hidden_layers=layers,
        output_of_input_layer=output_of_input_layer,
        dropout_rate=dropout_rate,
    )

    output = _create_output_layer(num_classes, output_of_hidden_layers=output)

    model = Model(inputs=input_models, outputs=output)

    return model


def _compile_model(model: tf.keras.Model, num_classes: int, learning_rate: float):
    """
    Compiles the model
    Args:
        model: keras model created by create_keras_model()
        num_classes: Number of classes in target variable
        learning_rate: Learning rate

    """

    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_true - y_pred), axis=-1))

    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[rmse],
    )


def _fit_model(
        model: tf.keras.Model,
        X_train_list: list,
        y_train,
        X_val_list: list,
        y_val,
        epochs: int,
        batch_size: int,
) -> tf.keras.callbacks.History:
    """
    Trains the model
    Args:
        model: keras model created by create_keras_model()
        X_train_list: list version of X_train created by to_input_list()
        y_train: train labels
        X_val_list: list version of X_val created by to_input_list()
        y_val: val labels
        epochs: number of epochs
        batch_size: batch size

    Returns: history of training

    """
    history = model.fit(
        X_train_list,
        y_train,
        validation_data=(X_val_list, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    return history


# endregion
# region main_functions
def predict(model: tf.keras.Model, X_test: pd.DataFrame, cate_cols: list) -> np.array:
    """
    predict function
    Args:
        model: keras model fit by fit_model
        X_test: Test features
        cate_cols: categorical columns list

    Returns: y_pred

    """
    X_test_list = _to_input_list(df=X_test, cate_cols=cate_cols)
    y_pred = model.predict(X_test_list)
    return y_pred


def train(
        X_train: pd.DataFrame,
        y_train: Union[pd.Series, np.array],
        X_val: pd.DataFrame,
        y_val: Union[pd.Series, np.array],
        layers: list,
        num_classes: int,
        cate_cols: list,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        dropout_rate: float = 0.3,
) -> Tuple[tf.keras.callbacks.History, tf.keras.Model]:
    """
    Training main function that takes dataset and parameters as input and returns the trained model with history
    Args:
        X_train: Train features
        y_train: train labels
        X_val: Validation labels
        y_val: validation labels
        layers: List of nodes in hidden layers
        num_classes: Number of classes in target variable
        cate_cols: categorical columns list
        learning_rate: learning rate
        epochs: number of epochs
        batch_size: batch size
        dropout_rate: dropout rate

    Returns: history of training, trained model

    """

    X_train_list = _to_input_list(df=X_train, cate_cols=cate_cols)
    X_val_list = _to_input_list(df=X_val, cate_cols=cate_cols)

    # if len(y_train.shape) == 1:
    #     y_train_categorical = tf.keras.utils.to_categorical(
    #         y_train, num_classes=num_classes, dtype="float32"
    #     )
    #
    #     y_val_categorical = tf.keras.utils.to_categorical(
    #         y_val, num_classes=num_classes, dtype="float32"
    #     )

    model = _create_keras_model(
        X_train=X_train,
        layers=layers,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        cate_cols=cate_cols,
    )

    _compile_model(model=model, num_classes=num_classes, learning_rate=learning_rate)
    history = _fit_model(
        model=model,
        X_train_list=X_train_list,
        y_train=y_train,
        X_val_list=X_val_list,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
    )

    return history, model


# endregion

ds = preprocess_train_with_pred_opt(standardize_or_not=True, impute_or_not=True)

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
            'gender_inequality_index_gii']

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

ds = impute_numeric_columns(ds,missing_flag=False)


train_labels = [np.array(ds.y_train[0]), np.array(ds.y_train[1])]
val_labels = [np.array(ds.y_val[0]), np.array(ds.y_val[1])]
test_labels = [np.array(ds.y_test[0]), np.array(ds.y_test[1])]

model = train(X_train=ds.X_train, y_train=train_labels, X_val=ds.X_val, y_val=val_labels,
              layers=[16, 32, 16], num_classes=1,
              cate_cols=cate_cols, learning_rate=1e-2, epochs=1000, batch_size=32, dropout_rate=0.1)

test_pred = predict(model=model[1], X_test=ds.X_test, cate_cols=cate_cols)

from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(test_labels[0], test_pred[0]
                              ))

print(rms)

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
