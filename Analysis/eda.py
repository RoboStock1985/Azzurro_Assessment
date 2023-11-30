import pandas as pd
import numpy as np

from pathlib import Path
from typing import Iterable

from sklearn.model_selection import train_test_split

from helper_functions import run_ML, feature_selection
from balance_data import oversample_balance_data
from date_utilities import create_dt_columns
from data_cleaning import remove_cross_correlated_columns, remove_outliers
from data_cleaning import drop_cols_which_have_only_one_value
from data_cleaning import convert_binary_feats_to_numeric, create_new_features
from data_cleaning import cast_columns_to_correct_types, fill_missing_values
from plotting import create_correlation_matrix, create_boxplot

import itertools
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rcParams

# plt.style.use("ggplot")
rcParams['figure.figsize'] = (8, 12)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# DATA_PATH = Path('./Data/small_home_insurance.csv')
DATA_PATH = Path('./Data/home_insurance.csv')

TARGET = 'POLICY_STATUS'


def process_test_data(test_df: pd.DataFrame,
                      train_df: pd.DataFrame) -> pd.DataFrame:

    """Replicate relevant parts of data preparation for Test.
    Results in a dataframe which will match the features of
    the train df."""

    # inpute missing values using train
    df_test = fill_missing_values(test_df, train_df)

    df = cast_columns_to_correct_types(df_test)
    df = create_dt_columns(df)

    df = create_new_features(df)

    for col in df.columns:
        convert_binary_feats_to_numeric(df, col)
    df_test = pd.get_dummies(df)

    # ensure that df_train and df_test have the same features
    df_train_cols = list(train_df.columns)
    df_test_cols = list(df_test.columns)

    cols_in_test_but_not_in_train = [x for x in df_test_cols
                                     if x not in df_train_cols]
    # remove these cols from test
    df_test.drop(columns=cols_in_test_but_not_in_train, axis=1,
                 inplace=True)
    cols_in_train_but_not_in_test = [x for x in df_train_cols
                                     if x not in df_test_cols]
    # add these cols to test
    for feature in cols_in_train_but_not_in_test:
        df_test[feature] = 0

    # confirm that dfs have the same columns
    assert set(df_test.columns) == set(train_df.columns)

    # re-order test_df
    df_test = df_test[train_df.columns]

    # ensure that columns have same dtypes
    df_test = df_test.astype(train_df.dtypes.to_dict())

    # for x in train_df.columns:
    #     df_test[x]=df_test[x].astype(train_df[x].dtypes.name)

    return df_test


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
    print(df.info())

    # Data columns (total 67 columns):
    #  #   Column                  Non-Null Count  Dtype
    # ---  ------                  --------------  -----
    #  0   QUOTE_DATE              4494 non-null   datetime64[ns]
    #  1   COVER_START             12142 non-null  object - datetime64[ns]
    #  2   CLAIM3YEARS             12142 non-null  object - Cat/Binary
    #  3   P1_EMP_STATUS           12142 non-null  object - Cat
    #  4   P1_PT_EMP_STATUS        104 non-null    object - Cat
    #  5   BUS_USE                 12142 non-null  object - Cat/Binary
    #  6   CLERICAL                170 non-null    object - Cat/Binary
    #  7   AD_BUILDINGS            12142 non-null  object - Cat/Binary
    #  8   RISK_RATED_AREA_B       8947 non-null   float64 - Ordinal
    #  9   SUM_INSURED_BUILDINGS   12142 non-null  float64 - Cat/Binary
    #  10  NCD_GRANTED_YEARS_B     12142 non-null  float64 - Cat
    #  11  AD_CONTENTS             12142 non-null  object - Cat/Binary
    #  12  RISK_RATED_AREA_C       11621 non-null  float64 - Ordinal
    #  13  SUM_INSURED_CONTENTS    12142 non-null  float64 - Cat
    #  14  NCD_GRANTED_YEARS_C     12142 non-null  float64 - Ordinal
    #  15  CONTENTS_COVER          12142 non-null  object - Cat/Binary
    #  16  BUILDINGS_COVER         12142 non-null  object - Cat/Binary
    #  17  SPEC_SUM_INSURED        12142 non-null  float64 - Continuous
    #  18  SPEC_ITEM_PREM          12142 non-null  float64 - Continuous
    #  19  UNSPEC_HRP_PREM         12142 non-null  float64 - Continuous
    #  20  P1_DOB                  12142 non-null  object - datetime64[ns]
    #  21  P1_MAR_STATUS           12142 non-null  object - Cat
    #  22  P1_POLICY_REFUSED       12142 non-null  object - Cat/Binary
    #  23  P1_SEX                  12142 non-null  object - Cat
    #  24  APPR_ALARM              12142 non-null  object - Cat/Binary
    #  25  APPR_LOCKS              12142 non-null  object - Cat/Binary
    #  26  BEDROOMS                12142 non-null  float64 - Ordinal
    #  27  ROOF_CONSTRUCTION       12142 non-null  float64 - Cat
    #  28  WALL_CONSTRUCTION       12142 non-null  float64 - Cat
    #  29  FLOODING                12142 non-null  object - Cat/Binary
    #  30  LISTED                  12142 non-null  float64 - Ordinal
    #  31  MAX_DAYS_UNOCC          12142 non-null  float64 - Cat
    #  32  NEIGH_WATCH             12142 non-null  object - Cat/Binary
    #  33  OCC_STATUS              12142 non-null  object - Cat
    #  34  OWNERSHIP_TYPE          12142 non-null  float64 - Cat
    #  35  PAYING_GUESTS           12142 non-null  float64 - Cat/Binary
    #  36  PROP_TYPE               12142 non-null  float64 - Cat
    #  37  SAFE_INSTALLED          12142 non-null  object - Cat/Binary
    #  38  SEC_DISC_REQ            12142 non-null  object - Cat/Binary
    #  39  SUBSIDENCE              12142 non-null  object - Cat/Binary
    #  40  YEARBUILT               12142 non-null  float64 - Continuous
    #  41  CAMPAIGN_DESC           0 non-null      float64 - Empty
    #  42  PAYMENT_METHOD          12142 non-null  object - Cat
    #  43  PAYMENT_FREQUENCY       5596 non-null   float64 - Cat
    #  44  LEGAL_ADDON_PRE_REN     12142 non-null  object - Cat/Binary
    #  45  LEGAL_ADDON_POST_REN    12142 non-null  object - Cat/Binary
    #  46  HOME_EM_ADDON_PRE_REN   12142 non-null  object - Cat/Binary
    #  47  HOME_EM_ADDON_POST_REN  12142 non-null  object - Cat/Binary
    #  48  GARDEN_ADDON_PRE_REN    12142 non-null  object - Cat/Binary
    #  49  GARDEN_ADDON_POST_REN   12142 non-null  object - Cat/Binary
    #  50  KEYCARE_ADDON_PRE_REN   12142 non-null  object - Cat/Binary
    #  51  KEYCARE_ADDON_POST_REN  12142 non-null  object - Cat/Binary
    #  52  HP1_ADDON_PRE_REN       12142 non-null  object - Cat/Binary
    #  53  HP1_ADDON_POST_REN      12142 non-null  object - Cat/Binary
    #  54  HP2_ADDON_PRE_REN       12142 non-null  object - Cat/Binary
    #  55  HP2_ADDON_POST_REN      12142 non-null  object - Cat/Binary
    #  56  HP3_ADDON_PRE_REN       12142 non-null  object - Cat/Binary
    #  57  HP3_ADDON_POST_REN      12142 non-null  object - Cat/Binary
    #  58  MTA_FLAG                12142 non-null  object - Cat/Binary
    #  59  MTA_FAP                 3682 non-null   float64 - Continuous
    #  60  MTA_APRP                3682 non-null   float64 - Continuous
    #  61  MTA_DATE                1809 non-null   object - datetime64[ns]
    #  62  LAST_ANN_PREM_GROSS     12142 non-null  float64 - Continuous
    #  63  i                       12142 non-null  int64 - Cat - Unique
    #  64  Police                  12142 non-null  object - Cat - Unique
    #  65  Quote_year              4494 non-null   float64 - Continuous
    #  66  POLICY_STATUS           12142 non-null  int32 - Binary/Target
    # dtypes: datetime64[ns](1), float64(24), int32(1), int64(1), object(40)
    # memory usage: 6.5+ MB

    print(df.info())

    for column in df.columns:
        print("Variable Counts:")
        print(df[column].value_counts())
        print("------------------------")

    # drop useless columns
    df.drop(columns=['CAMPAIGN_DESC', 'PAYMENT_FREQUENCY', 'i', 'Police'],
            inplace=True)

    print(df.shape)

    # ensure that all columns are the correct type
    # cast all columns as object for categorical
    # excluding numeric and date columns

    df = cast_columns_to_correct_types(df)

    print(df.info())
    print(df.shape)

    # look at unique values in each column &
    # drop all columns which have only one value
    df = drop_cols_which_have_only_one_value(df)

    # check for columns with NULLS
    print(df.isnull().sum())

    # QUOTE_DATE                 7648
    # COVER_START                   0
    # CLAIM3YEARS                   0
    # P1_EMP_STATUS                 0
    # P1_PT_EMP_STATUS          12038
    # BUS_USE                       0
    # CLERICAL                  11972
    # AD_BUILDINGS                  0
    # RISK_RATED_AREA_B          3195
    # SUM_INSURED_BUILDINGS         0
    # NCD_GRANTED_YEARS_B           0
    # AD_CONTENTS                   0
    # RISK_RATED_AREA_C           521
    # SUM_INSURED_CONTENTS          0
    # NCD_GRANTED_YEARS_C           0
    # CONTENTS_COVER                0
    # BUILDINGS_COVER               0
    # SPEC_SUM_INSURED              0
    # SPEC_ITEM_PREM                0
    # UNSPEC_HRP_PREM               0
    # P1_DOB                        0
    # P1_MAR_STATUS                 0
    # P1_POLICY_REFUSED             0
    # P1_SEX                        0
    # APPR_ALARM                    0
    # APPR_LOCKS                    0
    # BEDROOMS                      0
    # ROOF_CONSTRUCTION             0
    # WALL_CONSTRUCTION             0
    # FLOODING                      0
    # LISTED                        0
    # MAX_DAYS_UNOCC                0
    # NEIGH_WATCH                   0
    # OCC_STATUS                    0
    # OWNERSHIP_TYPE                0
    # PAYING_GUESTS                 0
    # PROP_TYPE                     0
    # SAFE_INSTALLED                0
    # SEC_DISC_REQ                  0
    # SUBSIDENCE                    0
    # YEARBUILT                     0
    # PAYMENT_METHOD                0
    # LEGAL_ADDON_PRE_REN           0
    # LEGAL_ADDON_POST_REN          0
    # HOME_EM_ADDON_PRE_REN         0
    # HOME_EM_ADDON_POST_REN        0
    # GARDEN_ADDON_PRE_REN          0
    # GARDEN_ADDON_POST_REN         0
    # KEYCARE_ADDON_PRE_REN         0
    # KEYCARE_ADDON_POST_REN        0
    # HP1_ADDON_PRE_REN             0
    # HP1_ADDON_POST_REN            0
    # HP2_ADDON_PRE_REN             0
    # HP2_ADDON_POST_REN            0
    # HP3_ADDON_POST_REN            0
    # MTA_FLAG                      0
    # MTA_FAP                    8460
    # MTA_APRP                   8460
    # MTA_DATE                  10333
    # LAST_ANN_PREM_GROSS           0
    # POLICY_STATUS                 0
    # dtype: int64

    # drop columns with more than 75% NULL
    allowed_null_fraction = 0.75
    keep_cols = []
    for col in df.columns:
        null_rate = df[col].isnull().sum()/len(df)
        if null_rate < allowed_null_fraction:
            keep_cols.append(col)
        else:
            print(f"Dropping column : {col} due to high rate of NULLs.")

    df = df[keep_cols]
    print(df.isnull().sum())

    # now look for rows with high number of nulls - as this looks like an issue
    print(df.shape)
    df['null_fraction'] = df.isnull().sum(axis=1)/len(df.columns)
    df = df[df['null_fraction'] < allowed_null_fraction]
    df.drop(columns=['null_fraction'], inplace=True)
    print(df.shape)

    df = df[keep_cols]
    print(df.isnull().sum())

    # still NULLS in the following columns
    # QUOTE_DATE                7648 - Date
    # RISK_RATED_AREA_B         3195 - Ordinal
    # RISK_RATED_AREA_C          521 - Ordinal
    # MTA_FAP                   8460 - Continuous
    # MTA_APRP                  8460 - Continuous

    df = fill_missing_values(df, df)

    # for the date columns - we should now etxract potentially useful info 
    # then drop the date columns - and fill the new columns nulls
    df = create_dt_columns(df)

    df = create_new_features(df)

    print(df.isnull().sum())
    print(df.describe())

    # use flags to easily turn on/off
    create_dist_plots = False
    if create_dist_plots:
        feats_to_plot = list(df.columns)
        for feature in feats_to_plot:
            print(f"Plotting {feature}...")
            sns.displot(df[feature])
            plt.savefig(f'./Data/DistPlots/{feature}_distribution.png')
            plt.close()
            print(f"Created 1D Distribution Plot For {feature}.")

    # create boxplots to see outliers and compairson to TARGET
    create_box_plots = False
    if create_box_plots:
        for feature in feats_to_plot:
            create_boxplot(df, feature, TARGET)
            print(f"Created Box Plot For {feature}.")

    # a few variables are overwhelmingly skewed to one option - remove
    skewed_cols = ["HP1_ADDON_PRE_REN", "HP2_ADDON_PRE_REN",
                   "HP3_ADDON_POST_REN", "LISTED", "OCC_STATUS",
                   "P1_POLICY_REFUSED", "PAYING_GUESTS", "ROOF_CONSTRUCTION",
                   "SUBSIDENCE", "WALL_CONSTRUCTION"]

    df.drop(columns=skewed_cols, inplace=True)

    # noticed negative values for LAST_ANN_PREM_GROSS
    # is this possible?
    # this may have caused spike just above zero where NaNs have been filled
    # domain knowledge would be useful here

    # MAX_DAYS_UNNOC seems to have two primary values - drop others

    # shouldn't use P1_SEX as determinative feature
    # - as it is a protected attribute
    # Ideally should use as a variable to conduct
    # post-model training fairness analysis
    # I.e. Ensure that Selection Rate & Model
    # Performance is consistent across protected attributes
    protected_atts = ["P1_SEX"]
    df.drop(columns=protected_atts, inplace=True)

    # QUOTE_DATE_DAY_IN_MONTH is massively skewed to one date
    # this is likely due to filling NaNs with mode
    # consider filling NaNs with matching distribution or dropping

    # Some continuous variables quite skewed or have long tails
    # Remove outliers later
    # SPEC_ITEM_PREM, LISTED, MTA_FAP, LAST_ANN_PREM_GROSS

    print(df.shape)

    # convert binary categorical variables to 1 and 0
    for col in df.columns:
        convert_binary_feats_to_numeric(df, col)

    df.info()

    # create target rate plots for remaining cat variables
    cat_vars = df.select_dtypes(include=[object]).columns
    plot_cat_vars = False
    if plot_cat_vars:
        for cat_var in cat_vars:
            corr_with_target = df.groupby(cat_var)[TARGET].mean()
            std_with_target = list(df.groupby(cat_var)[TARGET].std())
            print(corr_with_target)
            corr_with_target.plot(kind='bar', yerr=std_with_target)
            plt.title(f"Mean Target Rate W.R.T {cat_var} Categories")
            plt.ylabel("Mean Target Rate")
            plt.savefig(f'./Data/CatRatePlots/{cat_var}_Category_Vs_Target.png')
            plt.close()

    print("Target Variable Counts")
    print(df[TARGET].value_counts())
    target_occurence_rate = df[TARGET].value_counts()[1] /\
        (df[TARGET].value_counts()[0] + df[TARGET].value_counts()[1])
    print(f"{TARGET} occurence rate is : {target_occurence_rate:.2%}")

    # Variable Counts:
    # POLICY_STATUS
    # 1    8675
    # 0    3467
    # Name: count, dtype: int64
    # Live is actually much more likely than Lapsed, Cancelled & Unknown
    # Randomly guessing should be correct 71.45% of the time.
    # Not very imbalanced. Could try SMOTE but may not be necessary or helpful.

    # can create full matrix but it is very busy
    # do not use categorical cols for corr mat

    create_corr_mats_flag = False
    if create_corr_mats_flag:
        corr_mat_cols = [x for x in df.columns if x not in cat_vars]
        create_correlation_matrix(df[corr_mat_cols])
        # create corr mat for only variables of interest
        create_correlation_matrix(df[corr_mat_cols], target=TARGET)

    # create dummy variables for categories
    dummy_vars_df = pd.get_dummies(df)

    print('Finished Cleaning & Prepping Data.')

    return dummy_vars_df


def run_all_steps(steps_to_run: Iterable[bool]) -> pd.DataFrame:

    """Runs all of the steps for EDA, Data Cleaning & ML Model Fitting."""

    remove_outliers_flag, remove_cross_corr_feat_flag, \
        balance_data_flag = steps_to_run

    df = pd.read_csv(filepath_or_buffer=DATA_PATH)

    # small_df = df.sample(25600)
    # small_df.to_csv("small_home_insurance.csv", index=False)

    base_target_col = "POL_STATUS"
    print(df.shape)
    print(df[base_target_col].value_counts())
    # target_nans = df[base_target_col].isnull().sum()
    df.dropna(subset=[base_target_col], inplace=True)
    print(df.shape)

    # create target variable column
    # Live, Lapsed, Cancelled, Unknown
    # Assume we are only interested in Live
    df[TARGET] = np.where(df[base_target_col] == "Live", 1, 0)
    print(df[TARGET].head(100))
    df.drop(columns=[base_target_col], inplace=True)

    # organise by quote date - this will allow use to select OOT data for final testing
    df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"])
    df["Quote_year"] = df["QUOTE_DATE"].dt.year

    print(df["Quote_year"].value_counts())

    # be aware : no NaNs in OOT data
    # could artificially add in by slsecting random sample
    oot_filter = ((df["Quote_year"] == 2011) | (df["Quote_year"] == 2012))
    oot_data  = df[oot_filter]
    df = df[~oot_filter]

    # drop "Quote_year" - as will nt be used in model
    oot_data.drop(columns=["Quote_year"], inplace=True)
    df.drop(columns=["Quote_year"], inplace=True)

    # now split into train and validation
    X, y = df[df.columns.drop(TARGET)], df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42,
                                                        stratify=y)
    df_train = x_train.join(y_train)
    df_test = x_test.join(y_test)

    df_train = examine_and_clean_data(df_train)

    # remove outliers
    # turning on seems to decrease performance
    if remove_outliers_flag:
        df_train = remove_outliers(df_train)

    # can alter tolerance level if it is too strict - would depend
    # on improving model performance
    if remove_cross_corr_feat_flag:
        df_train = remove_cross_correlated_columns(df_train, TARGET)

    # address class imbalance - only do for training data
    # could also try undersampling - which would use much less data but is
    # less synthetic so could be effective
    if balance_data_flag:
        df_train = oversample_balance_data(df_train, TARGET)

    # process test to match structure of train
    df_test = process_test_data(df_test, df_train)
    df_oot = process_test_data(oot_data, df_train)

    # save train and test data
    df_train.to_csv("./Data/Datasets/TrainData.csv", index=False)
    df_test.to_csv("./Data/Datasets/TestData.csv", index=False)
    df_oot.to_csv("./Data/Datasets/OOTTestData.csv", index=False)

    # lazy_predict, classic_ml, tuned_ml, basic_NN
    run_flags = [False, False, True, False]

    # use to confirm that classifier will get 100% if given target
    # df_train["Cheat"] = df[TARGET]
    # df_test["Cheat"] = df[TARGET]

    # Forward Feature Selection
    # Could try Backward Feature Elimination - takes too long
    feature_removal_flag = False
    if feature_removal_flag:
        combined_results = feature_selection(df_train, df_test,
                                             TARGET, run_flags)
    else:
        # lazy_predict, classic_ml, tuned_ml, basic_NN
        combined_results = run_ML(df_train, df_test, TARGET, run_flags)

    combined_results["Removed Outliers"] = remove_outliers_flag
    combined_results["Removed Cross Corr"] = remove_cross_corr_feat_flag
    combined_results["Balanced Data"] = balance_data_flag

    return combined_results


if __name__ == '__main__':

    # constructs list of lists of combinations of run parameters for
    # remove_outliers_flag, remove_cross_corr_feat_flag, balance_data_flag
    all_combinations_of_steps_to_run = list(itertools.product([True, False],
                                                              repeat=3))

    # apparent best configuration for Random Forest Classifier
    all_combinations_of_steps_to_run = [[False, False, True]]

    # for neural network keep all and don't balance
    # all_combinations_of_steps_to_run = [[False, False, False]]

    all_results = []
    for steps in all_combinations_of_steps_to_run:
        results = run_all_steps(steps)
        all_results.append(results)

    # save all results to file
    results_path = Path("./Data/ML_Results.csv")
    all_results = pd.concat(all_results)

    all_results["Balanced Score"] = \
        all_results[['F1 Score', 'AUC Score']].mean(axis=1)

    all_results.to_csv(results_path, index=False)

    print("Done.")
