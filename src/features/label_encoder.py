from sklearn.preprocessing import LabelEncoder
import pandas as pd

class MultiColumnLabelEncoder:
    """
    MultiColumnLabelEncoder can be used to label encode multiple columns in a DataFrame
    Examples:
        MultiColumnLabelEncoder(columns = cate_cols).fit(df)
        X_train = MultiColumnLabelEncoder(columns = cate_cols).transform(X_train)
    """

    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def fit(self, X, y=None):
        return self  # not relevant here

    def transform(self, X):
        """
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        """
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


