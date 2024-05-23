from sklearn.linear_model import LogisticRegression
import os
from ml_operations import (load_data_from_npz, simple_cross_validation, print_simple_cross_validation_score,
                           save_trained_model)

if __name__ == '__main__':
    PATH_DATA = 'data'
    DATA_FILE_NAME = 'np_from_selected_compressed.npz'
    PATH_MODEL = 'models'
    MODEL_FILE_NAME = 'clf.pkl'

    print("Loading dataset...")
    X, y = load_data_from_npz(os.path.join(PATH_DATA, DATA_FILE_NAME))

    lr = LogisticRegression(C=100, max_iter=500)

    print("Evaluating model performance using cross validation...")
    scores = simple_cross_validation(X, y, clf=lr, n_splits=5)
    print_simple_cross_validation_score(scores)
    # Accuracy from cross-validation: 0.993 +- 0.001

    print("Training best model on the whole available dataset...")
    final_clf = LogisticRegression(C=100, max_iter=500)
    final_clf.fit(X, y)

    save_trained_model(final_clf, os.path.join(PATH_MODEL, MODEL_FILE_NAME))


