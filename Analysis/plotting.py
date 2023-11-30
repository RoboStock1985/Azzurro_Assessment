import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rcParams

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import time


def create_correlation_matrix(data: pd.DataFrame, target=None):

    """Draws a correlation matrix using supplied Dataframe.
    If target variable is supplied will draw only that row."""

    plt.figure(figsize=(9, 9))
    corrmat = data.corr(method="pearson")

    if target:
        corrmat = corrmat.loc[[target]]
        corrmat.drop(columns=[target], inplace=True)
        sns.heatmap(corrmat, xticklabels=corrmat.columns, cmap="Spectral_r")
    else:
        sns.heatmap(corrmat, cbar=True, annot=False, square=True, fmt='.2f',
                    annot_kws={'size': 8}, yticklabels=data.columns,
                    xticklabels=data.columns, cmap="Spectral_r")

    plt.tight_layout()
    plt.savefig(f'./Data/CorrelationPlots/correlation_matrix_vs_{target}.png')
    plt.close()


def create_boxplot(data: pd.DataFrame, feature: str, target: str):

    """Creates a basic 2D boxplot using a supplied feature and target."""

    plt.show()
    sns.catplot(x=target, y=feature, data=data, kind="box", aspect=1.5)
    plt.title(f"Boxplot for {feature}.")
    plt.tight_layout()
    plt.savefig(f'./Data/BoxPlots/{feature}_boxplot_vs_target.png')
    plt.close()


def AUC_PLOT(X_test, y_test, model):

    rcParams['figure.figsize'] = (8, 8)

    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_test))]

    # predict probabilities
    lr_probs = model.predict_proba(X_test)

    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]

    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)

    # summarize scores
    NSAUC = 'AUC=%.3f' % (ns_auc)
    ROAUC = 'AUC=%.3f' % (lr_auc)

    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label=('No Skill:'+NSAUC))
    model_name = str(model)
    plt.plot(lr_fpr, lr_tpr, marker='.', label=(model_name+':'+ROAUC))

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend()

    ts = time.time()
    plt.savefig(f'./Data/AUC_Plots/AUC_Plot_{ts}.png')
    plt.close()
