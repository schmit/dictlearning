from sklearn import linear_model
import numpy as np
import numpy.random

clf = linear_model.Lasso(alpha = 0.1, fit_intercept = False)

X = np.random.randn(10,3)
beta = np.ones((3,1))
beta[2] = 0

y = np.dot(X, beta) + 0.1 * np.random.randn(10,1)

clf.fit(X,y)

print clf.coef_