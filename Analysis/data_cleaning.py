import numpy as np
import pandas as pd


def remove_cross_correlated_columns(df: pd.DataFrame,
                                    target: str,
                                    tolerance=0.85, ) -> pd.DataFrame:

    """Checks to see which columns have a high degree of cross-correlation
    and removes one of them."""

    corr_matrix = df.drop(target, axis=1).corr().abs()

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
            print(f"Dropping column : {col} as it has only one value.")
            res = res.drop(col, axis=1)

    return res


def convert_binary_feats_to_numeric(df: pd.DataFrame,
                                    col: str) -> pd.DataFrame:

    """"""

    unique_vals = df[col].unique()
    # print(f"Column : {col}, Unique Values : {unique_vals}")
    if len(unique_vals) == 2:

        if set(unique_vals) == set(["N", "Y"]):
            df[col].replace('N', 0, inplace=True)
            df[col].replace('Y', 1, inplace=True)
        else:
            # could be other columns with binary options
            unique_vals.sort()
            df[col].replace(unique_vals[0], 0, inplace=True)
            df[col].replace(unique_vals[1], 1, inplace=True)

        df[col] = pd.to_numeric(df[col])

    return df


def cast_columns_to_correct_types(df):

    """"""

    date_cols = ["QUOTE_DATE", "COVER_START", "P1_DOB", "MTA_DATE"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format="mixed")

    numeric_cols = ["RISK_RATED_AREA_B", "RISK_RATED_AREA_C",
                    "NCD_GRANTED_YEARS_C", "SPEC_SUM_INSURED",
                    "SPEC_ITEM_PREM", "UNSPEC_HRP_PREM",
                    "BEDROOMS", "LISTED", "YEARBUILT", "MTA_FAP",
                    "MTA_APRP", "LAST_ANN_PREM_GROSS", "POLICY_STATUS"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    # ensure that remaining columns are object
    non_obj_cols = date_cols + numeric_cols
    obj_cols = [x for x in df.columns if x not in non_obj_cols]
    for col in obj_cols:
        # df[col] = df[col].astype('category')
        df[col] = df[col].astype(object)

    return df


def fill_missing_values(df: pd.DataFrame,
                        fill_df: pd.DataFrame) -> pd.DataFrame:

    """"""

    df["RISK_RATED_AREA_B"].fillna(fill_df["RISK_RATED_AREA_B"].mode()[0],
                                   inplace=True)
    df["RISK_RATED_AREA_C"].fillna(fill_df["RISK_RATED_AREA_C"].mode()[0],
                                   inplace=True)

    df["MTA_FAP"].fillna(df["MTA_FAP"].mean(), inplace=True)
    df["MTA_APRP"].fillna(df["MTA_APRP"].mean(), inplace=True)

    return df


def create_new_features(df: pd.DataFrame) -> pd.DataFrame:

    """"""

    # create new features that could be useful based on date
    df["DaysBetweenQuoteAndCover"] = \
        (df["COVER_START"] - df["QUOTE_DATE"]).dt.days

    df["DaysBetweenQuoteAndCover"].fillna(df["DaysBetweenQuoteAndCover"]
                                          .mode()[0], inplace=True)

    date_cols = df.select_dtypes(include=['datetime64[ns]']).columns
    df.drop(columns=date_cols, inplace=True)

    return df
