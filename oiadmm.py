import numpy as np

def OIADMM(D, A, X, Delta, beta, tau):
    '''
    Updates dictionary D using IOADMM method
    Note that this method works in both online as batch setting.

    D: dictionary
    np.array(dim, atoms)
    X: observation
    '''
    Gammahat = P - np.dot(A, X)
    Gamma = soft(Ghat + Delta / beta, 1 / beta)

    G = -np.dot((Delta / beta + Gammahat - Gamma),X)
    Ahat = prod(max(0, A - tau * G))

    Delta = Delta + beta * (P - np.dot(Ahat, X) - Gamma)
    return Ahat, Delta