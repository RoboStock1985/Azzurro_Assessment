import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score

from typing import Tuple

from ml_parameters import model_to_params_dict

MODELS = [DecisionTreeClassifier(random_state=42)]


def hypertune_model(model, X_train: pd.DataFrame, y_train: pd.DataFrame,
                    explore_params=False) -> Tuple[str]:

    """Accepts a basic Model. Retrieves parameter grid for that model.
    Creates training pipeline. Tunes model based on desired metric.
    Returns best model."""

    verbosity_level = 3

    # number of concurrent jobs to run
    num_jobs = 5

    if explore_params:
        num_cross_validations = 5
    else:
        num_cross_validations = 2

    pipe, new_params_grid = establish_parameters_for_pipeline(model,
                                                              explore_params)

    # set up RandomisedSearchCV Object
    optimal_params = RandomizedSearchCV(estimator=pipe,
                                        param_distributions=new_params_grid,
                                        scoring="f1",
                                        verbose=verbosity_level,
                                        n_jobs=num_jobs,
                                        cv=num_cross_validations)

    optimal_params.fit(X_train, y_train)

    best_params = optimal_params.best_params_
    # best_score = optimal_params.best_score_
    best_estimator = optimal_params.best_estimator_

    return best_estimator, best_params


def establish_parameters_for_pipeline(model, explore_params: bool) -> dict:

    """Sets up parameters for model by adding prefix.
    Adds parameters for preprocessing steps, including toggle on/off."""

    model_name = type(model).__name__

    # get model current parameters
    current_param_grid = {key: [value] for key,
                          value in model.get_params().items()}

    # get parameter grid if required and available
    parameters_grid = current_param_grid
    if explore_params:
        try:
            parameters_grid = model_to_params_dict[model_name]
        except KeyError:
            print(f"No parameter grid found for model : {model_name}.")
            parameters_grid = current_param_grid

    # create model training pipeline - includes preprocessing
    pipe = Pipeline([('scale', StandardScaler()),
                    ('pca', PCA()),
                    ('model', model)], verbose=3)

    # associate parameters with each pipeline step
    new_params_grid = {}
    for key in parameters_grid.keys():
        new_params_grid[f"model__{key}"] = parameters_grid[key]

    if explore_params:
        # add preprocessing params
        n_components = [5, 10]
        # new_params_grid["pca__n_components"] = n_components
        new_params_grid["pca"] = [None]
        for n_component in n_components:
            new_params_grid["pca"].append(PCA(n_component))
        # add in param to toggle scaling
        new_params_grid["scale"] = [None, StandardScaler()]
    else:
        new_params_grid["scale"] = [None]
        new_params_grid["pca"] = [None]

    return pipe, new_params_grid


def basic_metrics(y_test, y_pred,
                  verbose=False) -> Tuple[float]:

    """Calculates commonly used metrics for ML Classification."""

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    if verbose:
        print("-------------------------------------------------")
        print('Precision: %.3f' % precision)
        print('Recall: %.3f' % recall)
        print('F1: %.3f' % f1)
        print('Accuracy: %.3f' % accuracy(y_test, y_pred))
        print("-------------------------------------------------")

    return f1, recall, precision, accuracy


def run_basic_ML(df_train: pd.DataFrame, df_test: pd.DataFrame, target: str,
                 tune=False) -> pd.DataFrame:

    """Splits the incoming data into appropriate partitions
    and submits these to  hypertune_model to obtain model performance.
    """

    train_columns = list(df_train.columns)
    train_columns.remove(target)

    X_train = df_train[train_columns]
    y_train = df_train[[target]]

    x_test = df_test[train_columns]
    y_test = df_test[target]

    # changing shape of y columns to y arrays - might only be required for SGD
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    results = []
    for model in MODELS:

        # incremental feature adding/subtraction should probably go here
        # rather than in main

        model_name = type(model).__name__
        print(f"Model: {model_name}")

        fitted_model, best_params = hypertune_model(model, X_train, y_train,
                                                    explore_params=tune)

        y_pred = fitted_model.predict(x_test)
        f1, recall, precision, accuracy = basic_metrics(y_test, y_pred)
        results_dict = {'Model': model_name, 'F1 Score': f1,
                        'Precision': precision, 'Recall': recall,
                        'Accuracy': accuracy, 'Params': str(best_params)}
        results.append(results_dict)

        # TODO - save model and example input data

    results_columns = results_dict.keys()
    results_df = pd.DataFrame(results, columns=results_columns)

    return results_df
