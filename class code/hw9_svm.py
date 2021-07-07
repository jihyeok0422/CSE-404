import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    raw_data = loadmat('data.mat')
    X = raw_data['X']
    y = raw_data['Y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.444, random_state=0)

    c_list = [.0001, .001, .01, .1, 1, 10, 100, 1000, 10000, 100000]
    print("------------------------------------")
    for c in c_list:
        # Kernel Linear
        svc_linear = svm.SVC(kernel='linear', C=c)
        svc_linear.fit(X_train, y_train.ravel())
        predicted1 = svc_linear.predict(X_test)
        cnf_matrix1 = confusion_matrix(y_test, predicted1)
        print("c (linear): ", c)
        print("Support vectors:", len(svc_linear.support_))
        print("Confusion Matrix (linear): \n", cnf_matrix1)
        print("Accuracy: ", (cnf_matrix1[0][0] + cnf_matrix1[1][1])/cnf_matrix1.sum())
        print("")
        # Kernel Default rbf

        svc_rbf = svm.SVC(kernel='rbf', C=c)
        svc_rbf.fit(X_train, y_train.ravel())
        predicted2 = svc_rbf.predict(X_test)
        cnf_matrix2 = confusion_matrix(y_test, predicted2)
        print("c (rbf): ", c)
        print("Support vectors:", len(svc_rbf.support_))
        print("Confusion Matrix (rbf): \n", cnf_matrix2)
        print("Accuracy: ", (cnf_matrix2[0][0] + cnf_matrix2[1][1])/cnf_matrix2.sum())
        print("")

        # Kernel Poly
        svc_poly = svm.SVC(kernel='poly', degree=100, C=c)
        svc_poly.fit(X_train, y_train.ravel())
        predicted3 = svc_poly.predict(X_test)
        cnf_matrix3 = confusion_matrix(y_test, predicted3)
        print("c (poly): ", c)
        print("Support vectors:", len(svc_poly.support_))
        print("Confusion Matrix (poly): \n", cnf_matrix3)
        print("Accuracy: ", (cnf_matrix3[0][0] + cnf_matrix3[1][1])/cnf_matrix3.sum())
        print("")
        print("------------------------------------")

    print("Finished")



