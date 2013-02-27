import numpy as np
from math import exp, sqrt

def linearkernel(X, Y, sigma=1):
	'''
	Linear kernel for matrices
	rows: observations (every row a different obs)
	cols: features
	'''
	return np.dot(Y, X.transpose()) 

def gaussiankernel(X, Y, sigma=1):
	'''
	Gaussian kernel for matrices
	computes the distance from observations in Y to observations in X
	'''
	dimx = X.shape[0]
	dimy = Y.shape[0]

	K = np.zeros((dimy,dimx))
	for i in range(dimy):
		for j in range(dimx):
			k = exp(-np.dot(X[j]-Y[i], X[j]-Y[i])/(2*sigma**2))
			K[i,j] = k
	return K

def laplacekernel(X, Y, sigma=1):
	'''
	Gaussian kernel for matrices
	computes the distance from observations in Y to observations in X
	'''
	dimx = X.shape[0]
	dimy = Y.shape[0]

	K = np.zeros((dimy,dimx))
	for i in range(dimy):
		for j in range(dimx):
			k = exp(-sqrt(np.dot(X[j]-Y[i], X[j]-Y[i]))/(sigma))
			K[i,j] = k
	return K

'''
# Test code
a = np.ones((3,2))
b = np.ones((5,2))

b[0][0] = 0
b[1][0] = -1
b[2][0] = 3
b[3][0] = 3
b[4][0] = -3

b[0][1] = -4
b[1][1] = -3
b[2][1] = 2
b[3][1] = 1
b[4][1] = -6

a[0][0] = 2
a[1][0] = 1
a[2][0] = 0


a[0][1] = 1
a[1][1] = 1
a[2][1] = 2

print 'a'
print a
print 'b'
print b

print 'kernelb'
print linearkernel(b,b)
print 'kernelba'
print linearkernel(b,a)
#print gaussiankernel(b,b,5)
#print gaussiankernel(b,a,5)
'''