# rf_param_grid = {'n_estimators': [6, 7, 8],
#                  'max_features': ['sqrt', 'auto'],
#                  'max_depth': [55],
#                  'criterion': ['gini', 'entropy'],
#                  'min_samples_leaf': [2, 3, 4],
#                  'min_samples_split': [35, 40, 45]}

rf_param_grid = {'n_estimators': [100, 150, 200],
                 'max_features': ['sqrt'],
                 'max_depth': [55],
                 'criterion': ['gini'],
                 'min_samples_leaf': [3],
                 'min_samples_split': [40]}

# [CV 1/5] END model__bootstrap=True, model__criterion=entropy, model__max_depth=5, model__max_features=sqrt,
# model__min_samples_leaf=2, model__min_samples_split=10, model__n_estimators=100, model__random_state=42,
# pca=PCA(n_components=10), scale=StandardScaler();, score=0.610 total time=  50.9s

model_to_params_dict = {"RandomForestClassifier": rf_param_grid}
