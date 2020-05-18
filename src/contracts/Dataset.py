from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd


"""
This dataclass will be used to bucket all train, validation and test dataset for your model.
As a primary dataclass, it will be use in other contracts.
So, because we will use this class for many different techniques (imputation, modeling, testing)
We can set val and test as optional.
"""


@dataclass
class Dataset:
    X_train: Optional[Union[pd.DataFrame, np.array]] = None
    y_train: Optional[Union[pd.DataFrame, np.array]] = None
    X_val: Optional[Union[pd.DataFrame, np.array]] = None
    y_val: Optional[Union[pd.DataFrame, np.array]] = None
    X_test: Optional[Union[pd.DataFrame, np.array]] = None
    y_test: Optional[Union[pd.DataFrame, np.array]] = None
    full_dataset: Optional[Union[pd.DataFrame, np.array]] = None
