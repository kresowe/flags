import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import os
import math
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
                            n_inner_splits: int, n_outer_splits: int) -> tuple[list, list]:
    """Makes nested cross validation on dataset X with labels y using classifier clf.
    Parameters of clf given in param_grid are compared using grid search method.
    Returns
    inner_scores - detailed results from inner loop,
    outer_scores - scores of the best classifier in each iteration of outer loop
    best_params_list - best set of grid parameters in each iteration of outer loop """
    inner_cv = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=0)
    outer_cv = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=0)

    best_params_list = []
    inner_scores = []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test, = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=inner_cv)
        grid_search.fit(X_train, y_train)

        inner_scores.append(grid_search.cv_results_)
        best_params_list.append(grid_search.best_params_)
    return inner_scores, best_params_list


def process_cross_validation_results(inner_scores: list) -> tuple[list, NDArray, NDArray]:
    """Extracts important information from detailed outputs from nested cross validation.
    inner_scores should be first value returned by nested_cross_validation.
    Returns
    params_list - list of combinations of parameters checked in grid search,
    params_scores - mean score for each position from params list
    fit_times - mean times of fitting for each position from params list"""
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
    params_scores = params_scores.mean(axis=0)
    fit_times = fit_times.mean(axis=0)
    return params_list, params_scores, fit_times


def print_cross_validation_results(params_list: list, params_scores: NDArray, fit_times: NDArray,
                                   best_params_list: list) -> None:
    """Prints the results of nested cross validation.
    Mean scores and fit times correspond to positions in params_list."""
    print("Param scores - inner:")
    print(params_list)
    np.set_printoptions(precision=3)
    print("Mean score:", params_scores)
    print("Fit times:", fit_times)

    print('Best results on test sample:')
    print('Best parameters:')
    print(best_params_list)


def simple_cross_validation(X: NDArray, y: NDArray, clf: LogisticRegression, n_splits: int) -> NDArray:
    """Makes simple (non-nested) cross validation to estimate the performance of clf.
    It works on dataset X with labels y using classifier clf. """
    cv_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = []

    for train_idx, test_idx in cv_fold.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        score = clf.score(X[test_idx], y[test_idx])
        scores.append(score)
    return np.array(scores)


def print_simple_cross_validation_score(scores: NDArray) -> None:
    """Prints mean accuracy and its standard deviation obtained in simple_cross_validation."""
    print(f'Accuracy from cross-validation: {scores.mean():.3f} +- {scores.std():.3f}')


def save_trained_model(clf: LogisticRegression, file_path: Union[str, os.PathLike]) -> None:
    """Serializes classifier clf in file file_path so that it can be later loaded and used without fitting model again."""
    print("Saving the trained classifier...")
    try:
        pickle.dump(clf, open(file_path, 'wb'), protocol=4)
    except Exception as e:
        print(f"Error: {e}")


def model_performance_per_class(X: NDArray, y: NDArray, clf: LogisticRegression, test_part: float = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_part, random_state=0)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    labels = np.unique(y_test)

    class_accuracies = {}
    for label in labels:
        indices_lab = np.where(y_test == label)
        class_accuracies[label] = np.sum(y_test[indices_lab] == y_pred[indices_lab]) / len(y_test[indices_lab])

    print('Number of all classes:', len(labels))
    print('Labels with 100% accuracy:')
    great_accuracy_labels = []
    for label in class_accuracies:
        if math.isclose(class_accuracies[label], 1.0, abs_tol=1e-3):
            great_accuracy_labels.append(label)
    print('Number: ', len(great_accuracy_labels), '. Labels:', great_accuracy_labels)

    print('Labels with accuracy < 100%:')
    mistake_labels = []
    for label in class_accuracies:
        if class_accuracies[label] < 0.999:
            mistake_labels.append(label)
            print(label, ':', class_accuracies[label])

    for label in mistake_labels:
        print(f'Label {label} was identified as', y_pred[np.where(y_test == label)])

    return class_accuracies
