from sklearn.linear_model import LogisticRegression
import os
from flags.ml_operations import (load_data_from_npz, nested_cross_validation, process_cross_validation_results,
                                 print_cross_validation_results)


if __name__ == '__main__':
    DIRNAME = os.path.dirname(__file__)
    PATH_DATA = os.path.join(DIRNAME, 'data')
    DATA_FILE_NAME = 'np_from_selected_compressed.npz'

    print("Loading dataset...")
    X, y = load_data_from_npz(os.path.join(PATH_DATA, DATA_FILE_NAME))

    print("Training model using cross validation...")
    lr = LogisticRegression(max_iter=500, random_state=0)
    param_grid = {"C": [0.01, 0.1, 1, 10, 100, 1000]}

    inner_scores, best_params_list = nested_cross_validation(X, y, clf=lr, param_grid=param_grid,
                                                             n_inner_splits=4, n_outer_splits=4)

    params_list, params_scores, fit_times = process_cross_validation_results(inner_scores)
    print_cross_validation_results(params_list, params_scores, fit_times, best_params_list)

# Outputs:
# Param scores - inner:
# [{'C': 0.01}, {'C': 0.1}, {'C': 1}, {'C': 10}, {'C': 100}, {'C': 1000}]
# Mean score: [0.621 0.882 0.976 0.988 0.992 0.992]
# Fit times: [ 1.142  5.181  9.517 16.664 18.193 15.951]
# Best results on test sample:
# Best parameters:
# [{'C': 100}, {'C': 100}, {'C': 100}, {'C': 100}]
# Accuracy of the best model
# [0.993 0.994 0.992 0.995]
# Mean accuracy of the best model:
# 0.994 +- 0.001
