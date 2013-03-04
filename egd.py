import numpy as np

def gradL(B, w, x, p, l):
    bwmxt = (np.dot(B,w) - x).transpose()

    grad1 = np.dot(bwmxt, B).transpose()
    grad2 = np.log(w / p)
    return grad1 + l * grad2


def egd(D, x, l):
    y = np.expand_dims(x,axis=1)
    tol = 1e-12
    max_its = 100000
    alpha = 0.01
    
    B = np.append(D,-D,axis=1)
    w = np.ones((B.shape[1],1))
    p = np.ones((B.shape[1],1))

    grad = gradL(B,w,y,p,l)
    its = 0

    while(np.max(np.abs(grad)) > tol and its < max_its):
        #print L(B,y,w,l)
        its += 1
        w = w*np.exp(-alpha*grad)
        grad = gradL(B,w,y,p,l)

    alpha = w[0:len(w)/2.0] - w[len(w)/2.0:]

    #print np.sum(alpha > np.mean(alpha))

    return alpha

def L(d, x, w, l):
    return np.linalg.norm(np.dot(d,w)-x)





