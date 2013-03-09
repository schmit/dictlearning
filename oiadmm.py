import numpy as np

def shrinkage(A,b):
    '''
    Soft thresholding
    '''
    assert(b >= 0)
    Y = np.maximum(abs(A)-b, np.zeros(A.shape))
    Y = Y * np.sign(A)
    return Y


def OIADMM(D, A, X, L, beta, tau):
    '''
    Updates dictionary D using OIADMM method
    Note that this method works in both online as batch setting.

    D: dictionary
    np.array(dim, atoms)
    X: observation
    '''
    L = np.squeeze(L)
    Gammahat = np.squeeze(X - np.dot(D, A))
    Gamma = shrinkage(Gammahat + L / beta, 1 / beta)
    G = -np.outer((L / beta + Gammahat - Gamma), A.T)
    Dhat = np.maximum(np.zeros(D.shape), D - tau * G)

    L = L + beta * (X - np.dot(Dhat, A) - Gamma)
    return Dhat, L


def ADMMforX(A, P, lamb, gamma, phi, kappa):

    X = np.zeros((A.shape[1], 1))
    E = P
    rho = np.zeros(P.shape)

    max_its = 1000
    its = 0
    error = 1000
    tol = 1e-6
    while(its < max_its and error > tol):
        its = its+1
        AX = np.dot(A, X)
        rhophi = rho / phi
        E = shrinkage(P - AX + rhophi, 1 / phi)

        inner = AX + E - P - rhophi
        G = np.dot(A.transpose(), inner)
        X = np.max(X - kappa*G - lamb * kappa / phi, 0)
        rho = rho + gamma * phi * (P - np.dot(A, X) - E)
        error = np.norm(E)
    return X
