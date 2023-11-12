import pandas as pd
from imblearn.over_sampling import SMOTE


def oversample_balance_data(df: pd.DataFrame, target) -> pd.DataFrame:

    """Creates synthetic data rows using SMOTE"""

    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(df[df.columns.drop(target)], df[target])
    df = X.join(y)

    return df
