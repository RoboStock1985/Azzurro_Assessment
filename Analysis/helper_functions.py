import pandas as pd
from pathlib import Path

from lazy_predict import lazy_predict
from classic_ml import run_basic_ML
from neural_networks import run_basic_NN

from typing import Iterable


def get_lazy_results(df_train: pd.DataFrame, df_test: pd.DataFrame,
                     TARGET: str) -> pd.DataFrame:

    """Helper function to run lazy predict and save results."""

    lazy_df_results = lazy_predict(df_train, df_test, TARGET)
    lazy_df_results["Type"] = "Lazy"

    lazy_results_path = Path("./Data/ML_Lazy_Results.csv")
    lazy_df_results.to_csv(lazy_results_path, index=False)

    return lazy_df_results


def get_classic_ml_results(df_train: pd.DataFrame, df_test: pd.DataFrame,
                           TARGET: str) -> pd.DataFrame:

    """Helper function to run classic ML and save results."""

    classic_ml_results = run_basic_ML(df_train, df_test, TARGET)
    classic_ml_results["Type"] = "Classic"

    classic_ml_results_path = Path("./Data/ML_Classic_Results.csv")
    classic_ml_results.to_csv(classic_ml_results_path, index=False)

    return classic_ml_results


def get_tuned_ml_results(df_train: pd.DataFrame, df_test: pd.DataFrame,
                         TARGET: str) -> pd.DataFrame:

    """Helper function to run Tuned ML and save results."""

    tuned_ml_results = run_basic_ML(df_train, df_test, TARGET, tune=True)
    tuned_ml_results["Type"] = "Tuned"

    tuned_ml_results_path = Path("./Data/Tuned_Results.csv")
    tuned_ml_results.to_csv(tuned_ml_results_path, index=False)

    return tuned_ml_results


def get_basic_NN_results(df_train: pd.DataFrame, df_test: pd.DataFrame,
                         TARGET: str) -> pd.DataFrame:

    """Helper function to run Basic NN and save results."""

    basic_NN_results = run_basic_NN(df_train, df_test, TARGET)
    basic_NN_results["Type"] = "Neural Network"

    basic_NN_results_path = Path("./Data/Basic_NN_Results.csv")
    basic_NN_results.to_csv(basic_NN_results_path, index=False)

    return basic_NN_results


def run_ML(df_train: pd.DataFrame, df_test: pd.DataFrame, TARGET: str,
           run_flags: Iterable[bool]) -> pd.DataFrame:

    """Helper function to run one or more of the ML Model
    processing functions."""

    lazy_predict_flag, run_classic_ml_flag, run_tuned_ml_flag, \
        run_basic_NN_flag = run_flags

    # use lazy predict to get basic overview of ML performance
    results_to_combine = []
    if lazy_predict_flag:
        results_to_combine.append(get_lazy_results(df_train, df_test,
                                                   TARGET))
    if run_classic_ml_flag:
        results_to_combine.append(get_classic_ml_results(df_train, df_test,
                                                         TARGET))
    if run_tuned_ml_flag:
        results_to_combine.append(get_tuned_ml_results(df_train, df_test,
                                                       TARGET))
    if run_basic_NN_flag:
        results_to_combine.append(get_basic_NN_results(df_train, df_test,
                                                       TARGET))
    combined_ml_results = pd.concat(results_to_combine)

    return combined_ml_results
