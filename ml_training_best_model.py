import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

if __name__ == '__main__':
    PATH_DATA = 'data'
    DATA_FILE_NAME = 'np_from_selected_compressed.npz'
    PATH_MODEL = 'models'
    MODEL_FILE_NAME = 'clf.pkl'

    print("Loading dataset...")
    data = np.load(os.path.join(PATH_DATA, DATA_FILE_NAME))
    X = data['X']
    y = data['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    lr = LogisticRegression(C=1, max_iter=500)

    print("Training best model on the whole training set...")
    lr.fit(X_train, y_train)
    print("Testing best model....")
    y_pred_train = lr.predict(X_train)
    y_pred_test = lr.predict(X_test)
    print('Train set accuracy: {:.3f}'.format(accuracy_score(y_train, y_pred_train)))  # Train set accuracy: 0.979
    print('Test set accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred_test)))  # Test set accuracy: 0.976

    print("Training best model on the whole available dataset...")
    final_clf = LogisticRegression(C=1, max_iter=500)
    final_clf.fit(X, y)

    print("Saving the trained classifier...")
    try:
        pickle.dump(final_clf, open(os.path.join(PATH_MODEL, MODEL_FILE_NAME), 'wb'), protocol=4)
    except Exception as e:
        print(f"Error: {e}")
