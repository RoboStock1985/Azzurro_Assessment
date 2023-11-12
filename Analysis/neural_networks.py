import pandas as pd
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from classic_ml import basic_metrics


def fit_keras_model(X_train: pd.DataFrame, y_train: pd.DataFrame,
                    X_eval: pd.DataFrame, y_eval: pd.DataFrame):

    """Uses Tensorflow - Keras to fit a basic NN model."""

    print("Constructing Keras Model...")

    n_input_features = len(list(X_train.columns))
    model = Sequential()

    # first layer & one hidden
    model.add(Dense(12, input_shape=(n_input_features,), activation='relu'))
    # hidden layers
    model.add(Dense(8, activation='relu'))
    # final layer
    model.add(Dense(1, activation='sigmoid'))

    print("Compiling Keras Model...")

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=tf.keras.metrics.Precision())

    print("Fitting Keras Model...")

    X_train = np.asarray(X_train).astype('float32')
    y_train = np.asarray(y_train).astype('float32')
    X_eval = np.asarray(X_eval).astype('float32')
    y_eval = np.asarray(y_eval).astype('float32')

    n_epochs = 50
    batch_size = 10
    verbosity_level = 1
    model.fit(X_train, y_train, epochs=n_epochs,
              batch_size=batch_size,
              verbose=verbosity_level,
              validation_data=(X_eval, y_eval))

    return model


def run_basic_NN(df_train: pd.DataFrame, df_test: pd.DataFrame, target: str):

    """Runs the Neural Network Function"""

    train_columns = list(df_train.columns)
    train_columns.remove(target)

    X_train = df_train[train_columns]
    y_train = df_train[[target]]

    x_test = df_test[train_columns]
    y_test = df_test[[target]]

    # split test into test & eval as Keras can use this
    X_eval, X_test, y_eval, y_test = train_test_split(x_test, y_test,
                                                      test_size=0.5,
                                                      random_state=42,
                                                      stratify=y_test)

    prob_limit = 0.5
    y_pred = fit_keras_model(X_train, y_train, X_eval,
                             y_eval).predict((X_test) > prob_limit).astype(int)

    f1, recall, precision, accuracy = basic_metrics(y_test, y_pred)
    results = []
    results_dict = {'Model': "Keras", 'F1 Score': f1,
                    'Precision': precision, 'Recall': recall,
                    'Accuracy': accuracy, 'Params': 'Default'}
    results.append(results_dict)

    results_columns = results_dict.keys()
    results_df = pd.DataFrame(results, columns=results_columns)

    return results_df
