import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.contracts.Dataset import Dataset


def standardize_full_ds(dataset: Dataset):
    """
    Standardizer function. Takes X_train, X_val(optional), X_test(optional) and standardizes them.


    Args:
        dataset: this is the dataclass Dataset where at least X_train is not None

    Returns: a Dataset dataclass where the dataset are standardized (X_train, X_val and X_test) if X_val and X_test are
    not None.

    """

    # If no X_train is provided raise Error
    if dataset.full_dataset is None:
        raise ValueError("please provide full_dataset")

    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    full_dataset_numeric = dataset.full_dataset.select_dtypes(include=numerics)
    column_names = list(full_dataset_numeric.columns)
    # Create full_dataset_std
    full_dataset_np = np.array(full_dataset_numeric)
    scaler = StandardScaler()
    full_dataset_std = scaler.fit_transform(full_dataset_np)
    dataset.full_dataset[column_names] = full_dataset_std

    return dataset
