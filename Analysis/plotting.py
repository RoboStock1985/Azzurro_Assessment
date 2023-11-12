import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def create_correlation_matrix(data: pd.DataFrame, target=None):

    """Draws a correlation matrix using supplied Dataframe.
    If target variable is supplied will draw only that row."""

    plt.figure(figsize=(9, 9))
    corrmat = data.corr()

    if target:
        corrmat = corrmat.loc[[target]]
        corrmat.drop(columns=[target], inplace=True)
        sns.heatmap(corrmat, xticklabels=corrmat.columns, cmap="Spectral_r")
    else:
        sns.heatmap(corrmat, cbar=True, annot=False, square=True, fmt='.2f',
                    annot_kws={'size': 8}, yticklabels=data.columns,
                    xticklabels=data.columns, cmap="Spectral_r")

    plt.tight_layout()
    plt.show()


def create_boxplot(data: pd.DataFrame, feature: str, target: str):

    """Creates a basic 2D boxplot using a supplied feature and target."""

    plt.show()
    sns.catplot(x=target, y=feature, data=data, kind="box", aspect=1.5)
    plt.title(f"Boxplot for {feature}.")
    plt.show()
