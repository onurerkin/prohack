#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:21:21 2020

@author: busebalci
"""


# TODO: regression
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
from functools import partial


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
    output = Dense(num_classes, activation="sigmoid")(output_of_hidden_layers)
    return output


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
    
    # def custom_loss(y_true, y_pred, train_data):
        
    #     train_data['y_pred']=y_true
    #     y_opt_true=optimize_mckinsey(train_data)
    #     train_data['y_pred']=y_pred
    #     y_opt_pred=optimize_mckinsey(train_data)
    #     loss=0.8*K.sqrt(K.mean(K.square(y_true-y_pred), axis=-1))
    #     +0.2*0.01*K.sqrt(K.mean(K.square(y_opt_true-y_opt_pred), axis=-1))                                
            
    #     return loss
    
    #loss_with_optimize = partial(custom_loss, train_data=train_data)

    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[rmse],
    )



def _fit_model(
    model: tf.keras.Model,
    X_train_list: list,
    y_train: Union[pd.Series, np.array],
    X_val_list: list,
    y_val: Union[pd.Series, np.array],
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
