import sklearn

import pandas as pd

from lazypredict import Supervised
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score

# these classifiers can sometimes hang and halt the training process - remove
Supervised.removed_classifiers.append("NuSVC")
Supervised.CLASSIFIERS.remove(("NuSVC", sklearn.svm.NuSVC))
Supervised.removed_classifiers.append("SVC")
Supervised.CLASSIFIERS.remove(("SVC", sklearn.svm.SVC))
LazyClassifier = Supervised.LazyClassifier


def lazy_predict(train_data: pd.DataFrame, test_data: pd.DataFrame,
                 target_col: str) -> pd.DataFrame:

    """Uses the lazy predict library to run multiple ML models
    :param: pd.DataFrame train_data: training data
    :param: pd.DataFrame test_data: testing data
    :param: str target_col: target dependent variable
    :return: pd.DataFrame: results for all models tested
    """

    train_columns = list(train_data.columns)
    train_columns.remove(target_col)

    X_train = train_data[train_columns]
    X_test = test_data[train_columns]
    y_train = train_data[[target_col]]
    y_test = test_data[[target_col]]

    # fit all models
    clf = LazyClassifier(predictions=True, verbose=1)
    _, predictions = clf.fit(X_train, X_test, y_train, y_test)

    results = []
    for classifier in list(predictions.columns):

        precision = precision_score(y_test, predictions[classifier])
        recall = recall_score(y_test, predictions[classifier])
        f1 = f1_score(y_test, predictions[classifier])
        accuracy = accuracy_score(y_test, predictions[classifier])

        results_dict = {'Model': classifier, 'F1 Score': f1,
                        'Precision': precision, 'Recall': recall,
                        'Accuracy': accuracy, 'Params': 'Default'}
        results.append(results_dict)

    results_df = pd.DataFrame(results, columns=['Model', 'F1 Score',
                                                'Precision', 'Recall',
                                                'Accuracy'])

    return results_df
