xgb_parameters_grid = {
    "reg_alpha": [0.01, 0.1, 1],
    "reg_lambda": [4.5],
    "subsample": [0.5, 0.75, 1],
    "n_estimators": [20],
    "max_depth": [3, 8, 10],
    "colsample_bytree": [0.2, 0.5, 0.8],
    "gamma": [0.1, 0.3, 0.8],
    "learning_rate": [0.01, 0.1, 1],
    "scale_pos_weight": [0.001, 0.1, 1.0],
    "early_stopping_rounds": [20],
    "eval_metric": ["aucpr"]
}

rf_param_grid = {'n_estimators': [100, 1000],
                 'max_features': ['sqrt', 'auto'],
                 'max_depth': [5, 50],
                 'criterion': ['gini', 'entropy'],
                 'random_state': [42],
                 'bootstrap': [True, False],
                 'min_samples_leaf': [2, 4],
                 'min_samples_split': [5, 10]}

extra_trees_param_grid = {'n_estimators': [100, 200, 500],
                          'min_samples_leaf': [5, 10, 20],
                          'max_features': [2, 3, 4]}

k_neighbours_param_grid = {'n_neighbors': [5, 10, 50],
                           'weights': ['uniform', 'distance'],
                           'algorithm': ['auto', 'ball_tree',
                                         'kd_tree', 'brute']}

log_reg_param_grid = {'penalty': ['l1', 'l2', 'elasticnet', None],
                      'tol': [0.001, 0.0001, 0.00001],
                      'C': [0.1, 1, 2]}

sgd_param_grid = {'loss': ['hinge', 'log_loss', 'modified_huber'],
                  'penalty': ['l2', 'l1', 'elasticnet', None],
                  'tol': [0.001, 0.0001, 0.00001]}

decision_tree_grid = {'criterion': ['gini', 'log_loss'],
                      'max_depth': [2, 5, 10],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 5, 10],
                      'min_weight_fraction_leaf': [0, 0.25, 0.5],
                      'max_features': ['sqrt', 'log2']}

# decision_tree_grid = {'criterion': ['gini'],
#                       'max_depth': [None, 2, 10]}

model_to_params_dict = {'XGBClassifier': xgb_parameters_grid,
                        "RandomForestClassifier": rf_param_grid,
                        "KNeighborsClassifier": k_neighbours_param_grid,
                        "LogisticRegression": log_reg_param_grid,
                        "SGDClassifier": sgd_param_grid,
                        "DecisionTreeClassifier": decision_tree_grid}
