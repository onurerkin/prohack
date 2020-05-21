import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn_pandas import CategoricalImputer

from src.contracts.Dataset import Dataset

enable_iterative_imputer


def impute_numeric_columns_full_ds(dataset: Dataset, missing_flag=True) -> Dataset:
    """
    Imputes numerical columns for train, validation, and test sets.
    It uses IterativeImputer and it fit_transform the train set but
    fit validation and test sets
    Args:
        dataset: it's the Dataset dataclass where X_train should be set

    Returns:
        A Tuple of imputed DataFrames for training, validation, and test sets

    """

    if dataset.full_dataset is None:
        raise ValueError(
            "full_dataset in the Dataset dataclass is empty, you should at least provide full_dataset"
        )

    num_cols = list(
        dataset.full_dataset.columns[
            (dataset.full_dataset.dtypes == "float") | (dataset.full_dataset.dtypes == "bool") | (dataset.full_dataset.dtypes == "int")
        ]
    )
    print("Imputing")

    imputer = IterativeImputer(
        n_nearest_features=10, skip_complete=True, random_state=42
    )

    imputed_train_cols = imputer.fit_transform(dataset.full_dataset.loc[:, num_cols])

    train_inds = dataset.full_dataset.index

    df_imputed_train_cols = pd.DataFrame(
        data=imputed_train_cols, index=train_inds, columns=num_cols
    )
    full_dataset_imputed = dataset.full_dataset.copy()
    full_dataset_imputed.loc[train_inds, num_cols] = df_imputed_train_cols
    if missing_flag:
        nas = dataset.full_dataset_imputed[num_cols].isna().sum()
        nas = list(nas[nas != 0].index)

        for col in nas:
            missing_flag[col + "_isna"] = dataset.missing_flag[col].isna()

    dataset.full_dataset = full_dataset_imputed

    return dataset
