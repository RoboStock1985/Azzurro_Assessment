import datetime
import pandas as pd


def get_month_name_from_num(month_num: int) -> str:

    """Determines the short name of a month from an int and returns."""

    datetime_object = datetime.datetime.strptime(str(month_num), "%m")
    month_name = datetime_object.strftime("%b")

    return month_name


def create_month_dummies_from_date(df: pd.DataFrame,
                                   col_name: str,
                                   new_col_name: str) -> pd.DataFrame:

    """Converts a string date column to Datetime.
    Extracts the month of that date.
    Creates Dummy Variables using the month column."""

    df[new_col_name] = pd.to_datetime(df[col_name])

    # create categorical columns which represent the month for these
    # date is not much use in an ML model
    
    df[new_col_name] = df[new_col_name].dt.month
    df[new_col_name] = df[new_col_name].apply(get_month_name_from_num)

    dummy_cols = [new_col_name]
    df = pd.get_dummies(df, prefix=dummy_cols, columns=dummy_cols)

    if col_name != new_col_name:
        df.drop(columns=[col_name], inplace=True)

    return df
