from sklearn.linear_model import LogisticRegression
import os
from flags.ml_operations import (load_data_from_npz, simple_cross_validation, print_simple_cross_validation_score,
                                 save_trained_model, model_performance_per_class)

if __name__ == '__main__':
    DIRNAME = os.path.dirname(__file__)
    PATH_DATA = os.path.join(DIRNAME, 'data')
    DATA_FILE_NAME = 'np_from_selected_compressed.npz'
    PATH_MODEL = os.path.join(DIRNAME, 'models')
    MODEL_FILE_NAME = 'clf.pkl'

    print("Loading dataset...")
    X, y = load_data_from_npz(os.path.join(PATH_DATA, DATA_FILE_NAME))

    lr = LogisticRegression(C=100, max_iter=500, random_state=0)

    print("Evaluating model performance using cross validation...")
    scores = simple_cross_validation(X, y, clf=lr, n_splits=5)
    print_simple_cross_validation_score(scores)
    # Accuracy from cross-validation: 0.993 +- 0.001

    print("Checking model performance for each class...")
    clf = LogisticRegression(C=100, max_iter=500, random_state=0)
    scores_class = model_performance_per_class(X, y, clf, test_part=0.2)

    print("Training best model on the whole available dataset...")
    final_clf = LogisticRegression(C=100, max_iter=500, random_state=0)
    final_clf.fit(X, y)

    save_trained_model(final_clf, os.path.join(PATH_MODEL, MODEL_FILE_NAME))
