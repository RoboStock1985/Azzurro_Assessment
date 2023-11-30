import pandas as pd


def create_dt_columns(df: pd.DataFrame) -> pd.DataFrame:

    """Uses date columns to add dt features for month,
    weekday and day in month. Fills NaNs."""

    date_cols = df.select_dtypes(include=['datetime64[ns]']).columns

    for col in date_cols:
        df[col + '_MONTH'] = df[col].dt.month
        df[col + '_WEEKDAY'] = df[col].dt.weekday
        df[col + '_DAY_IN_MONTH'] = df[col].dt.day

        df[col + '_MONTH'].fillna(df[col + '_MONTH'].mode()[0],
                                  inplace=True)
        df[col + '_WEEKDAY'].fillna(df[col + '_WEEKDAY'].mode()[0],
                                    inplace=True)
        df[col + '_DAY_IN_MONTH'].fillna(df[col + '_DAY_IN_MONTH'].mode()[0],
                                         inplace=True)

    return df
