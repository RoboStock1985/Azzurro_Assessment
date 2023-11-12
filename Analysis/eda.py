import pandas as pd

from pathlib import Path
from typing import Iterable

from sklearn.model_selection import train_test_split

from helper_functions import run_ML
from balance_data import oversample_balance_data
from date_utilities import create_month_dummies_from_date
from data_cleaning import remove_cross_correlated_columns, remove_outliers
from data_cleaning import drop_cols_which_have_only_one_value
from plotting import create_correlation_matrix, create_boxplot

import itertools

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

DATA_PATH = Path('./Data/ASSESSMENT_BREAKAGE_DATASET.csv')
TEST_DATA_PATH = Path('./Data/TEST_DATA.csv')

TARGET = 'BREAK_NEXT_MONTH'


def examine_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:

    """Performs the entire Exploratory Data Analysis for this dataset
     and returns a cleaned & prepped dataset which can be used for ML models

     :param pd.DataFrame df: The dataframe to be processed
     :return pd.DataFrame: the cleaned dataframe
     """

    print(df.head())

    # drop rows where data is completely duplicated as early as possible
    # to reduce processing time
    num_duplicates = df.duplicated().sum()
    print(f"Found {num_duplicates} duplicates, removing...")
    df.drop_duplicates(inplace=True)

    # check number of columns, rows and data types
    print(df.shape)
    # (837977, 99)
    print(df.info())

    # Noticed 99 columns - only 42 listed in DD Document

    # majority of columns are numeric data type, which is desirable
    # 3 columns - INDIVIDUAL_ID, Determination_Date, Transaction_Month
    # are object (string)

    # leave INDIVIDUAL_ID as it is - as will not be using it for ML Model

    df = create_month_dummies_from_date(df, 'Determination_Date',
                                        'Determination_Month')
    df = create_month_dummies_from_date(df, 'Transaction_Month',
                                        'Transaction_Month')

    print(df.info())

    # look at unique values in each column &
    # drop all columns which have only one value
    df = drop_cols_which_have_only_one_value(df)

    # drop INDIVIDUAL_ID - as far too many unique values
    # would cause overfitting in model
    df.drop(columns=['INDIVIDUAL_ID'], inplace=True)

    # check for columns with NULLS
    print(df.isnull().sum())

    # lots of nulls - no NULLS in dependent variable, which is encouraging

    # Note : Dependent variable is int type, has two unique values and no nulls

    # many bureau variables are quite sparse - this is due to many
    # customers never having had credit lines
    # can assume that real-life use case we will have lots of missing values
    # for these columns also - therefore cannot just drop them - also large num

    # for certain models like LightGBM and XGBBoost -
    # it might be best to leave these as NULL

    # investigate distributions of variables to determine imputation values
    print(df.describe())

    # for feature in list(df.columns):
    #     sns.displot(df[feature])

    # NOTE: Some variables extremely skewed - E.g. Start_Of_Month_Balance
    # Remove outliers later

    # NOTE: Disparity in number of 1s and 0s for dependent variable
    print("Target Variable Counts")
    print(df[TARGET].value_counts())
    target_occurence_rate = (df[TARGET].value_counts()[1]
                             / df[TARGET].value_counts()[0])
    print(f"{TARGET} occurence rate is : {target_occurence_rate:.2%}")

    # may need to address class imbalance - 0.90%

    # for feature in list(df.columns):
    #     create_boxplot(df, feature, TARGET)

    # lots of extreme values falling outside of normal range

    # for feature in list(df.columns):
    #     feat_val_counts = df[feature].value_counts()
    #     print(f"Values for {feature}: {feat_val_counts}")

    # overhwelmingly seems like the NULLS come from the bureau variables
    # wouldn't make sense to impute using mean
    # use most common variable - 0
    # this will technically lose some information
    # as the model will not differentiate between:
    # no longer has credit and has never had credit
    # could create a separate dummy variable for this

    df.fillna(0, inplace=True)
    print(df.isnull().sum())

    # try to address by using scalers

    # create corr mat for only variables of interest
    # create_correlation_matrix(df, target=TARGET)

    # # can create full matrix but it is very busy
    # create_correlation_matrix(df)

    print('Finished Cleaning & Prepping Data.')

    return df


def run_all_steps(steps_to_run: Iterable[bool]) -> pd.DataFrame:

    """Runs all of the steps for EDA, Data Cleaning & ML Model Fitting."""

    remove_outliers_flag, remove_cross_corr_feat_flag, \
        balance_data_flag = steps_to_run

    # load the data using the "|" separator, as this is what the file uses
    df = pd.read_csv(filepath_or_buffer=DATA_PATH, sep='|')
    # df = pd.read_csv(filepath_or_buffer=TEST_DATA_PATH, sep=',')

    # save a chopped down version to investigate -
    # as using entire dataset is very slow
    # create temp 1% dataset
    # test_df = df.sample(8000)
    # test_df.to_csv(Path('./Data/TEST_DATA.csv'), index=False)

    # save a chopped down version to investigate -
    # as using entire dataset is very slow
    # create temp 20% dataset
    # test_df = df.sample(170000)
    # test_df.to_csv(Path('./Data/TEST_DATA.csv'), index=False)

    X, y = df[df.columns.drop(TARGET)], df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42,
                                                        stratify=y)
    # df_train = pd.concat([x_train, y_train], axis=1)
    # df_test = pd.concat([x_test, y_test], axis=1)

    df_train = x_train.join(y_train)
    df_test = x_test.join(y_test)

    df_train = examine_and_clean_data(df_train)
    df_test = examine_and_clean_data(df_test)

    # remove outliers
    # turning on seems to decrease performance
    if remove_outliers_flag:
        df_train = remove_outliers(df_train)

    # tried to remove most extreme values using 1% window
    # due to large number of variables and high skewness of data
    # this resulted in more than 50% of data being dropped

    # find any cross-correlated independent variables and reduce
    # E.g. Start_Of_Month_Balance and Determination_Balance
    # Amount Paid So Far could be a useful variable

    # can alter tolerance level if it is too strict - would depend
    # on improving model performance
    if remove_cross_corr_feat_flag:
        df_train = remove_cross_correlated_columns(df_train, TARGET)

    # create corr mat for only variables of interest
    # create_correlation_matrix(df_train, target=TARGET)
    # can create full matrix but it is very busy
    # create_correlation_matrix(df_train)

    # address class imbalance - only do for training data
    # could also try undersampling - which would use much less data but is
    # less synthetic so could be effective
    if balance_data_flag:
        df_train = oversample_balance_data(df_train, TARGET)

    # add categorical columns to test - to match train

    # ensure that df_train and df_test have the same features
    df_train_cols = list(df_train.columns)
    df_test_cols = list(df_test.columns)

    cols_in_test_but_not_in_train = [x for x in df_test_cols
                                     if x not in df_train_cols]
    # remove these cols from test
    df_test.drop(columns=cols_in_test_but_not_in_train, axis=1, inplace=True)
    cols_in_train_but_not_in_test = [x for x in df_train_cols
                                     if x not in df_test_cols]
    # add these cols to test
    for feature in cols_in_train_but_not_in_test:
        df[feature] = 0

    # lazy_predict, classic_ml, tuned_ml, basic_NN
    run_flags = [False, False, False, True]
    combined_results = run_ML(df_train, df_test, TARGET, run_flags)

    combined_results["Removed Outliers"] = remove_outliers_flag
    combined_results["Removed Cross Corr"] = remove_cross_corr_feat_flag
    combined_results["Balanced Data"] = balance_data_flag

    return combined_results


if __name__ == '__main__':

    all_combinations_of_steps_to_run = list(itertools.product([True, False],
                                                              repeat=3))

    # apparent best configuration for decision tree from lazy predict
    # all_combinations_of_steps_to_run = [[False, False, False]]

    all_results = []
    for steps in all_combinations_of_steps_to_run:
        results = run_all_steps(steps)
        all_results.append(results)

    # save results to file
    results_path = Path("./Data/ML_Results.csv")
    all_results = pd.concat(all_results)
    all_results.to_csv(results_path, index=False)

    print("Done.")
