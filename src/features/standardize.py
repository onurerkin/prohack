import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.contracts.Dataset import Dataset


def standardize(dataset: Dataset):
    """
    Standardizer function. Takes X_train, X_val(optional), X_test(optional) and standardizes them.


    Args:
        dataset: this is the dataclass Dataset where at least X_train is not None

    Returns: a Dataset dataclass where the dataset are standardized (X_train, X_val and X_test) if X_val and X_test are
    not None.

    """

    # If no X_train is provided raise Error
    if dataset.X_train is None:
        raise ValueError("please provide X_train")

    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    X_train_numeric = dataset.X_train.select_dtypes(include=numerics)
    column_names = list(X_train_numeric.columns)
    # Create X_train_std
    X_train_np = np.array(X_train_numeric)
    scaler = StandardScaler()
    X_train_std = pd.DataFrame(scaler.fit_transform(X_train_np), columns=column_names)
    dataset.X_train[column_names] = X_train_std

    if dataset.X_val is not None:
        X_val_numeric = dataset.X_val[column_names]
        X_val_np = np.array(X_val_numeric)
        X_val_std = pd.DataFrame(scaler.transform(X_val_np), columns=column_names)
        dataset.X_val[column_names] = X_val_std

    if dataset.X_test is not None:
        X_test_numeric = dataset.X_test[column_names]
        X_test_np = np.array(X_test_numeric)
        X_test_std = pd.DataFrame(scaler.transform(X_test_np), columns=column_names)
        dataset.X_test[column_names] = X_test_std

    return dataset
