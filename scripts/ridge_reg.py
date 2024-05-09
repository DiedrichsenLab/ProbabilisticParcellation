import numpy as np
import scipy.optimize as so


def standardize(d, type='zscore'):
    if type == 'z-score':
        d = (d - np.mean(d)) / np.std(d)
    if type == 'norm':
        d = (d / np.sqrt((d**2).sum()))
    if type == 'mean':
        d = (d - np.mean(d))

    return d


def ridgeFit(Y, X, fit_intercept=True, voxel_wise=False, alpha=1.0):
    """Fits a multiple regression loss function 

    Args:
        y (ndarray): Dependent variable 
        X (np.ndarray): regression design matrix 
        b0 (np.ndarray). Initial guess for the parameter vector
        alpha (float): Ridge regression parameter
    Returns:
        R2: Fitted R2 value 
        b: Fitted 
    """

    if fit_intercept:
        # add a column of ones to the design matrix
        X = np.hstack((np.ones((X.shape[0], 1)), X))

    # using the closed form solution for ridge regression
    B = np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T @ Y

    # calculate residuals
    res = Y - X @ B

    # calculate RSS
    RSS = sum(res**2)

    # calculate TSS
    TSS = sum((Y - np.mean(Y))**2)

    # calculate R2
    R2 = 1 - RSS / TSS

    if not voxel_wise:
        RSS = sum(RSS)
        TSS = sum(TSS)
        R2 = 1 - RSS / TSS

    return (R2, B)


if __name__ == "__main__":
    Y = np.random.randn(514, 68)
    X = np.random.randn(514, 169)

    R2, B = ridgeFit(Y, X, fit_intercept=True, voxel_wise=False, alpha=1.0)

    print("A shape: ")
