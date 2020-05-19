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
from src.features.impute_columns import (impute_categorical_columns,impute_numeric_columns)
from src.contracts.Dataset import Dataset
from src.features.standardize import standardize
from src.features.label_encoder import MultiColumnLabelEncoder
from src.features.preprocess import preprocess_
# endregion

# Import `Sequential` from `keras.models`
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from keras_lr_finder import LRFinder


#Preprocess data
ds = preprocess_(standardize_or_not=False,impute_or_not=True)

cate_cols=['galaxy']
MultiColumnLabelEncoder(columns = cate_cols).fit(ds.X_train)
ds.X_train = MultiColumnLabelEncoder(columns = cate_cols).transform(ds.X_train)
ds.X_val = MultiColumnLabelEncoder(columns = cate_cols).transform(ds.X_val)
ds.X_test = MultiColumnLabelEncoder(columns = cate_cols).transform(ds.X_test)



def modified_sigmoid(x): 
  return tf.nn.sigmoid(x) * 0.7


for num_nodes in range(10,1000,20):
    #Initialize the constructor
    model = Sequential()
    model.add(Dense(200, activation='relu', input_shape=(ds.X_train.shape[1],)))
    #model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(1, activation=modified_sigmoid))
    
    model.summary()
    
    #kernel_regularizer=l2(0.1),bias_regularizer=l2(0.1)
    
    #Learning Rate Finder
    
    #lr_finder = LRFinder(model)
    #lr_finder.find(ds.X_train, ds.y_train, start_lr=0.0001, end_lr=100, batch_size=32, epochs=5)
    #lr_finder.plot_loss(n_skip_beginning=1, n_skip_end=1)
    
    
    #Compile model
    model.compile(loss='mse', optimizer=Adam(lr=1e-5))
    
    
    callbacks = []
    history = model.fit(np.array(ds.X_train).astype(np.float32), np.array(ds.y_train).astype(np.float32),validation_data=(np.array(ds.X_val).astype(np.float32), np.array(ds.y_val).astype(np.float32)), batch_size=128, epochs=10, callbacks=callbacks, verbose=1)
    
    mse = model.evaluate(np.array(ds.X_val).astype(np.float32), np.array(ds.y_val).astype(np.float32))
    print("MSE: %.4f" % mse)
    print("Number of Nodes: %.4f" % num_nodes)


