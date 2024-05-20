import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
import os

if __name__ == '__main__':
    PATH_DATA = 'data'
    DATA_FILE_NAME = 'np_from_selected_compressed.npz'

    print("Loading dataset...")
    data = np.load(os.path.join(PATH_DATA, DATA_FILE_NAME))
    X = data['X']
    y = data['y']
    print(X.shape, y.shape)

    print("Training model using cross validation...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    lr = LogisticRegression(max_iter=500)
    param_grid = {"C": [0.01, 0.1, 1, 10, 100, 1000]}

    inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)

    outer_scores = []
    best_params_list = []
    inner_scores = []

    for train_idx, test_idx in outer_cv.split(X_train, y_train):
        X_train_i, X_test_i, = X_train[train_idx], X_train[test_idx]
        y_train_i, y_test_i = y_train[train_idx], y_train[test_idx]

        grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=inner_cv)
        grid_search.fit(X_train_i, y_train_i)

        inner_scores.append(grid_search.cv_results_)
        best_params_list.append(grid_search.best_params_)

        best_model = grid_search.best_estimator_
        outer_scores.append(best_model.score(X_test_i, y_test_i))

    params_scores = []
    fit_times = []

    counter = 0
    params_list = []
    for run in inner_scores:
        if counter == 0:
            params_list = run['params']
        params_scores.append(run['mean_test_score'])
        fit_times.append(run['mean_fit_time'])
        counter += 1

    print("Param scores:")
    params_scores = np.array(params_scores)
    print(params_list)
    np.set_printoptions(precision=3)
    print('Mean score:', params_scores.mean(axis=0))
    print('Std deviation:', params_scores.std(axis=0))

    fit_times = np.array(fit_times)
    print("Fit times:", fit_times.mean(axis=0))

# Outputs:
# Param scores:
# [{'C': 0.01}, {'C': 0.1}, {'C': 1}, {'C': 10}, {'C': 100}, {'C': 1000}]
# Mean score: [0.448 0.815 0.972 0.984 0.992 0.993]
# Std deviation: [0.009 0.006 0.002 0.001 0.001 0.001]
# Fit times: [ 0.941  3.597  8.039 16.33  18.013 17.214]
