import pandas as pd
from imblearn.over_sampling import SMOTE


def oversample_balance_data(df: pd.DataFrame, target) -> pd.DataFrame:

    """Creates synthetic data rows using SMOTE"""

    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(df[df.columns.drop(target)], df[target])
    df = X.join(y)

    print("Balanced Data Using Oversampling.")

    target_occurence_rate = df[target].value_counts()[1] /\
        (df[target].value_counts()[0] + df[target].value_counts()[1])
    print(f"{target} occurence rate is : {target_occurence_rate:.2%}")

    return df
