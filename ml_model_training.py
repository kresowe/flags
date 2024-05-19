import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

if __name__ == '__main__':
    PATH_DATA = 'data'
    DATA_FILE_NAME = 'np_from_selected_compressed.npz'

    data = np.load(os.path.join(PATH_DATA, DATA_FILE_NAME))
    X = data['X']
    y = data['y']
    print(X.shape, y.shape)
    print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print(y_train)

    lr = LogisticRegression(random_state=0)
    print("Training model....")
    lr.fit(X_train, y_train)
    print("Testing model....")
    y_pred = lr.predict(X_test)
    print(y_pred)
    print(y_test)
    print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)))  # Accuracy: 0.976

