from sklearn.linear_model import LogisticRegression
import os
from ml_operations import (load_data_from_npz, nested_cross_validation, process_inner_cross_validation_results,
                           print_inner_cross_validation_results, print_outer_cross_validation_results)


if __name__ == '__main__':
    PATH_DATA = 'data'
    DATA_FILE_NAME = 'np_from_selected_compressed.npz'

    print("Loading dataset...")
    X, y = load_data_from_npz(os.path.join(PATH_DATA, DATA_FILE_NAME))

    print("Training model using cross validation...")
    lr = LogisticRegression(max_iter=500)
    param_grid = {"C": [0.01, 0.1, 1, 10, 100, 1000]}

    inner_scores, outer_scores, best_params_list = nested_cross_validation(X, y, clf=lr, param_grid=param_grid,
                                                                           n_inner_splits=4, n_outer_splits=4)

    params_list, params_scores, fit_times = process_inner_cross_validation_results(inner_scores)
    print_inner_cross_validation_results(params_list, params_scores, fit_times)
    print_outer_cross_validation_results(best_params_list, outer_scores)

# Outputs:
# Param scores - inner:
# [{'C': 0.01}, {'C': 0.1}, {'C': 1}, {'C': 10}, {'C': 100}, {'C': 1000}]
# Mean score: [0.618 0.879 0.975 0.988 0.992 0.992]
# Fit times: [ 1.17   5.005 10.908 18.61  19.399 17.206]
# Best results on test sample:
# Best parameters:
# [{'C': 100}, {'C': 1000}, {'C': 100}, {'C': 100}]
# Accuracy of the best model
# [0.992 0.993 0.989 0.994]
# Mean accuracy of the best model:  # but this result is irrelevant since the best model is not always the same.
# 0.9921515079415064 +- 0.0020032928137539074
