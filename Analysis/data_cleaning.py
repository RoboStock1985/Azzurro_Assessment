import numpy as np
import pandas as pd


def remove_cross_correlated_columns(df: pd.DataFrame,
                                    tolerance=0.90) -> pd.DataFrame:

    """Checks to see which columns have a high degree of cross-correlation
    and removes one of them."""

    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                              k=1).astype(bool))

    # Find features with correlation greater than tolerance
    to_drop = [column for column in upper.columns
               if any(upper[column] > tolerance)]

    # Drop features
    print(f"Dropping columns: {to_drop}")
    df.drop(to_drop, axis=1, inplace=True)

    return df


def remove_outliers(df: pd. DataFrame) -> pd.DataFrame:

    """Searchs for extreme outlier values in a provided dataframe.
    Removes records which contain those outliers."""

    print(len(df))
    for col in list(df.columns):

        col_type = df[col].dtype
        if col_type in ['float64']:

            print(col)
            q_low = df[col].quantile(0.01)
            q_hi = df[col].quantile(0.99)

            df = df[(df[col] <= q_hi) & (df[col] >= q_low)]
            print(len(df))

        else:
            print(f"Column {col} is not numeric.")

    return df


def drop_cols_which_have_only_one_value(df: pd.DataFrame) -> pd.DataFrame:

    """Drops columns which only have one value - as useless for ML"""

    print(df.nunique())
    res = df
    for col in df.columns:
        if len(df[col].unique()) == 1:
            res = res.drop(col, axis=1)

    return res
