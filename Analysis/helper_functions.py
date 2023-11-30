import pandas as pd
from pathlib import Path

from lazy_predict import lazy_predict
from classic_ml import run_basic_ML
from neural_networks import run_basic_NN

from typing import Iterable
from matplotlib import pyplot as plt


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


def feature_selection(df_train: pd.DataFrame, df_test: pd.DataFrame,
                      target: str, run_flags: Iterable,
                      max_features=None) -> pd.DataFrame:

    """Uses Forward Feature Selection to iteratively investigate performance.
    Records numerous metrics."""

    # really should use a validation set for this
    # and a test set for final assessment
    # ok as will use OOT set for final testing
    features = list(df_train.columns)
    features.remove(target)
    feats_being_used = [target]
    best_auc = 0
    print(len(features))

    # shuffle dataframe to illustrate
    # that order of features does not impact
    # random.shuffle(features)

    # organise so feat order is the same every time
    features.sort()

    if max_features:
        features = features[:max_features]

    model_performance_results = []
    for feat_num, feature in enumerate(features):

        print(f"Now adding feature {feat_num} of {len(features)}.")

        feats_being_used.append(feature)

        df_train_ML = df_train[feats_being_used]
        df_test_ML = df_test[feats_being_used]

        combined_results = run_ML(df_train_ML, df_test_ML,
                                  target, run_flags)

        # get performance - if it has improved then keep the feature
        # otherwise delete the feature from the used list

        # record metrics
        latest_f1 = combined_results['F1 Score'].mean()
        latest_precision = combined_results['Precision'].mean()
        latest_recall = combined_results['Recall'].mean()
        latest_accuracy = combined_results['Accuracy'].mean()
        latest_auc = combined_results['AUC Score'].mean()

        model_performance = {"NFeatures": feat_num,
                             "F1 Score": latest_f1,
                             "Recall": latest_recall,
                             "Precision": latest_precision,
                             "Accuracy": latest_accuracy,
                             "AUC Score": latest_auc}

        model_performance_results.append(model_performance)

        print(f'AUC Score is : {latest_auc}')
        print(f'Best AUC Score is : {best_auc}')

        if latest_auc > best_auc:
            best_auc = latest_auc
        else:
            feats_being_used.remove(feature)

    feat_sel_res_df = pd.DataFrame(model_performance_results,
                                   columns=["NFeatures",
                                            "F1 Score",
                                            "Recall",
                                            "Precision",
                                            "Accuracy",
                                            "AUC Score"])
    print(feat_sel_res_df)
    feat_sel_res_df.to_csv("./Data/FeatureSelectionResults.csv",
                           index=False)

    for metric in ["F1 Score", "AUC Score"]:
        feat_sel_res_df.plot.scatter(x="NFeatures", y=metric)
        plt.savefig(f'./Data/FeatureSelectionPlot_{metric}.png')

    feats_being_used.remove(target)
    combined_results["Best Features"] = str(feats_being_used)
    combined_results["NFeatures"] = len(feats_being_used) - 1

    return combined_results
