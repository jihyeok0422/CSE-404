import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd

# Runs the PCA algorithm
def pca(X, num_components):
    x_norm = StandardScaler().fit_transform(X)

    cov_mat = np.cov(x_norm, rowvar=False)

    eig_val, eig_vect = np.linalg.eig(cov_mat)

    sorted_index = np.argsort(eig_val)[::-1]
    sorted_eigenvectors = eig_vect[:, sorted_index]

    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

    x_reduced = np.dot(eigenvector_subset.transpose(), x_norm.transpose()).transpose()

    return x_reduced, eigenvector_subset

# Reconstrcuting Image
def reconstruct(X_reduced, eigenvector_subset):
    recon = X_reduced @ eigenvector_subset.T
    return recon

if __name__ == '__main__':
    data = loadmat('USPS.mat')
    X = pd.DataFrame(data['A'])
    y = pd.DataFrame(data['L'])
    error = []
    list = [10, 50, 100, 200]
    for i in list:
        pca_data, eigVec = pca(X, i)
        print("D:", i)
        print(pca_data)
        Error10 = np.sqrt(mean_squared_error(X.iloc[:, :i], pca_data))
        print("Reconstruction Error when d =", i, ":", Error10)
        print("")
        recon = reconstruct(pca_data, eigVec)
        eachNum = np.reshape(recon[0], (16, 16))
        plt.imshow(eachNum, cmap="gray")
        plt.show()