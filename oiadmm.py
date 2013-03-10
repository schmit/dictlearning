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

    #print 'NEW CALL'
    P = np.expand_dims(P, 1)
    X = np.zeros((A.shape[1], 1))
    #print X.shape
    #print A.shape
    E = P
    #print E.shape
    rho = np.zeros(P.shape)

    max_its = 50
    its = 0
    error = 1000
    tol = 1e-2
    while(its < max_its and error > tol):
        its = its+1
        AX = np.dot(A, X)
        rhophi = rho / phi

        E = shrinkage(np.squeeze(P - AX + rhophi), 1.0 / phi)
        E = np.expand_dims(E, 1)

        inner = AX + E - P - rhophi
        #print inner.shape
        G = np.dot(A.transpose(), inner)
        #print G.shape
        Xnew = np.maximum(X - kappa*G - lamb * kappa / phi, 0.0)
        error = np.linalg.norm(Xnew-X, 2)/(np.linalg.norm(Xnew, 2)+10**-6)
        X = Xnew
        #print X.shape
        #print A.shape
        rho = rho + gamma * phi * (P - np.dot(A, X) - E)
    #print np.squeeze(X)
    #print X.shape
    return np.squeeze(X)
