import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
import pickle
import os
from typing import Union


def load_data_from_npz(file_path: Union[str, os.PathLike]) -> tuple[NDArray, NDArray]:
    """Loads data from .npz file.
    Returns X - feature matrix and y - target variables (class labels) vector."""
    data = np.load(file_path)
    X = data['X']
    y = data['y']
    print('Dataset size:', X.shape, y.shape)
    return X, y


def nested_cross_validation(X: NDArray, y: NDArray, clf: LogisticRegression, param_grid: dict,
                            n_inner_splits: int, n_outer_splits: int) -> tuple[list, list, list]:
    inner_cv = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=0)
    outer_cv = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=0)

    outer_scores = []
    best_params_list = []
    inner_scores = []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test, = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=inner_cv)
        grid_search.fit(X_train, y_train)

        inner_scores.append(grid_search.cv_results_)
        best_params_list.append(grid_search.best_params_)

        best_model = grid_search.best_estimator_
        outer_scores.append(best_model.score(X_test, y_test))
    return inner_scores, outer_scores, best_params_list


def process_inner_cross_validation_results(inner_scores: list) -> tuple[list, NDArray, NDArray]:
    params_scores = []
    fit_times = []
    params_list = []

    counter = 0
    for run in inner_scores:
        if counter == 0:
            params_list = run['params']
        params_scores.append(run['mean_test_score'])
        fit_times.append(run['mean_fit_time'])
        counter += 1

    params_scores = np.array(params_scores)
    fit_times = np.array(fit_times)
    return params_list, params_scores, fit_times


def print_inner_cross_validation_results(params_list: list, params_scores: NDArray, fit_times: NDArray) -> None:
    print("Param scores - inner:")
    print(params_list)
    np.set_printoptions(precision=3)
    print("Mean score:", params_scores.mean(axis=0))
    print("Fit times:", fit_times.mean(axis=0))


def print_outer_cross_validation_results(best_params_list: list, outer_scores: list):
    outer_scores = np.array(outer_scores)

    print('Best results on test sample:')
    print('Best parameters:')
    print(best_params_list)
    print('Accuracy of the best model')
    print(outer_scores)
    print('Mean accuracy of the best model:')
    print(f'{outer_scores.mean()} +- {outer_scores.std()}')


def simple_cross_validation(X: NDArray, y: NDArray, clf: LogisticRegression, n_splits: int) -> NDArray:
    cv_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = []

    for train_idx, test_idx in cv_fold.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        score = clf.score(X[test_idx], y[test_idx])
        scores.append(score)
    return np.array(scores)


def print_simple_cross_validation_score(scores: NDArray) -> None:
    print(f'Accuracy from cross-validation: {scores.mean():.3f} +- {scores.std():.3f}')


def save_trained_model(clf: LogisticRegression, file_path: Union[str, os.PathLike]) -> None:
    print("Saving the trained classifier...")
    try:
        pickle.dump(clf, open(file_path, 'wb'), protocol=4)
    except Exception as e:
        print(f"Error: {e}")