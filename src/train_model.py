from sklearn.svm import SVC

def train_svm(X_train, y_train, kernel='linear'):
    clf = SVC(kernel=kernel)
    clf.fit(X_train, y_train)
    return clf
