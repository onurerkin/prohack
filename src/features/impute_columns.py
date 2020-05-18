import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn_pandas import CategoricalImputer

from src.contracts.Dataset import Dataset

enable_iterative_imputer


def impute_numeric_columns(dataset: Dataset) -> Dataset:
    """
    Imputes numerical columns for train, validation, and test sets.
    It uses IterativeImputer and it fit_transform the train set but
    fit validation and test sets
    Args:
        dataset: it's the Dataset dataclass where X_train should be set

    Returns:
        A Tuple of imputed DataFrames for training, validation, and test sets

    """

    if dataset.X_train is None:
        raise ValueError(
            "X_train in the Dataset dataclass is empty, you should at least provide X_train"
        )

    num_cols = list(
        dataset.X_train.columns[
            (dataset.X_train.dtypes == "float") | (dataset.X_train.dtypes == "bool")
        ]
    )
    print("Imputing Training Set")

    imputer = IterativeImputer(
        n_nearest_features=5, skip_complete=True, random_state=42
    )

    imputed_train_cols = imputer.fit_transform(dataset.X_train.loc[:, num_cols])

    train_inds = dataset.X_train.index

    df_imputed_train_cols = pd.DataFrame(
        data=imputed_train_cols, index=train_inds, columns=num_cols
    )
    X_train_imputed = dataset.X_train.copy()
    X_train_imputed.loc[train_inds, num_cols] = df_imputed_train_cols

    nas = dataset.X_train[num_cols].isna().sum()
    nas = list(nas[nas != 0].index)

    for col in nas:
        X_train_imputed[col + "_isna"] = dataset.X_train[col].isna()

    if dataset.X_val is not None:
        # Impute val
        print("Imputing Val Set")
        val_inds = dataset.X_val.index
        imputed_val_cols = imputer.transform(dataset.X_val.loc[:, num_cols])
        df_imputed_val_cols = pd.DataFrame(
            data=imputed_val_cols, index=val_inds, columns=num_cols
        )
        X_val_imputed = dataset.X_val.copy()
        X_val_imputed.loc[val_inds, num_cols] = df_imputed_val_cols

        for col in nas:
            X_val_imputed[col + "_isna"] = dataset.X_val[col].isna()

        dataset.X_val = X_val_imputed

    if dataset.X_test is not None:
        # Impute test
        print("Imputing Test Set")
        test_inds = dataset.X_test.index
        imputed_test_cols = imputer.transform(dataset.X_test.loc[:, num_cols])
        df_imputed_test_cols = pd.DataFrame(
            data=imputed_test_cols, index=test_inds, columns=num_cols
        )
        X_test_imputed = dataset.X_test.copy()
        X_test_imputed.loc[test_inds, num_cols] = df_imputed_test_cols

        for col in nas:
            X_test_imputed[col + "_isna"] = dataset.X_test[col].isna()

        dataset.X_test = X_test_imputed

    dataset.X_train = X_train_imputed
    return dataset


def impute_categorical_columns(dataset: Dataset) -> Dataset:
    """
    Imputes categorical columns for train, validation, and test sets.
    It uses IterativeImputer and it fit_transform the train set but
    fit validation and test sets
    Args:
        dataset: it's the Dataset dataclass where X_train should be set

    Returns:
        A Dataset Dataclass containing the
    """

    imputer = CategoricalImputer()

    # Create complete df
    df = dataset.X_train

    df = df.append([dataset.X_val, dataset.X_test]).reset_index(drop=True)
    cate_cols = list(df.columns[df.dtypes == "category"])
    nas = df[cate_cols].isna().sum()

    X_train_copy = dataset.X_train.copy()
    X_val_copy = dataset.X_val.copy()
    X_test_copy = dataset.X_test.copy()

    # Iterate over the columns and impute
    for col in pd.DataFrame(nas[nas != 0]).index:
        X_train_copy[col + "_isna"] = X_train_copy[col].isna()
        print("Imputing :" + str(col))
        X_train_copy[col] = imputer.fit_transform(X_train_copy[col])

        if X_val_copy is not None:
            X_val_copy[col + "_isna"] = X_val_copy[col].isna()
            X_val_copy[col] = imputer.transform(X_val_copy[col])
            dataset.X_val = X_val_copy

        if X_test_copy is not None:
            X_test_copy[col + "_isna"] = X_test_copy[col].isna()
            X_test_copy[col] = imputer.transform(X_test_copy[col])
            dataset.X_test = X_test_copy

    dataset.X_train = X_train_copy

    return dataset
