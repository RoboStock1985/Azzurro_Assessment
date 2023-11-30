import pickle
import shap

import pandas as pd

from matplotlib import pyplot as plt

from classic_ml import basic_metrics


def load_model():

    """"""

    # load the model from disk
    filename = "./Data/Models/finalized_model.pkl"
    loaded_model = pickle.load(open(filename, 'rb'))

    return loaded_model


def test_model(model, df_test: pd.DataFrame, target: str):

    """"""

    train_columns = list(df_test.columns)
    train_columns.remove(target)

    x_test = df_test[train_columns]
    y_test = df_test[target]

    # changing shape of y columns to y arrays - might only be required for SGD
    y_test = y_test.values.ravel()

    y_pred = model.predict(x_test)
    f1, recall, precision, accuracy, roc_auc = basic_metrics(y_test, y_pred,
                                                             verbose=True)


def get_shapley_values_and_plot(df_train: pd.DataFrame, pipeline, target: str):

    """"""

    train_columns = list(df_train.columns)
    train_columns.remove(target)
    X = df_train[train_columns]
    X = X.sample(1000)

    X_array = pipeline.named_steps['scale'].fit_transform(X)
    # X = pipeline.named_steps['pca'].fit_transform(X)
    # explainer = shap.KernelExplainer(pipeline.named_steps['model'].predict_proba, X_array)
    # explainer = shap.Explainer(pipeline.named_steps['model'].predict_proba, X_array)
    explainer = shap.TreeExplainer(pipeline.named_steps['model'])
    print("Created Shap Explainer.")

    shap_values = explainer.shap_values(X_array)
    print("Calculated Shapley Values..")

    # i = 4
    # shap.force_plot(explainer.expected_value, shap_values[i],
    #                 features=X.iloc[i], feature_names=X.columns)

    shap.summary_plot(shap_values, features=X, feature_names=X.columns)
    plt.savefig('./Data/ShapPlots/ShapSummaryPlot.png')

    # shap.summary_plot(shap_values, features=X, feature_names=X.columns,
    #                   plot_type='bar')
    # plt.savefig('./Data/ShapPlots/ShapSummaryBarPlot.png')


if __name__ == "__main__":

    TARGET = "POLICY_STATUS"

    train_data = pd.read_csv("./Data/Datasets/TrainData.csv")
    test_data = pd.read_csv("./Data/Datasets/TestData.csv")
    oot_test_data = pd.read_csv("./Data/Datasets/OOTTestData.csv")

    model = load_model()
    test_model(model, test_data, TARGET)
    get_shapley_values_and_plot(train_data, model, TARGET)
