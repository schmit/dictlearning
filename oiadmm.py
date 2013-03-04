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
    Updates dictionary D using IOADMM method
    Note that this method works in both online as batch setting.

    D: dictionary
    np.array(dim, atoms)
    X: observation
    '''
    L = np.squeeze(L)
    Gammahat = np.squeeze(X - np.dot(D, A))
    Gamma = shrinkage(Gammahat + L / beta, 1 / beta)

    print Gammahat.shape
    print Gamma.shape
    print L.shape
    print A.shape

    G = -np.outer((L / beta + Gammahat - Gamma),A.T)
    Dhat = np.maximum(np.zeros(D.shape), D - tau * G)

    L = L + beta * (X - np.dot(Dhat, A) - Gamma)
    return Dhat, L

X = np.ones((5,1))
A = np.ones((3,1))
D = np.ones((5,3))

print OIADMM(D,A,X,X,1,1)