import numpy as np
import numpy.linalg
import time
from sklearn import linear_model
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import utility
from math import sqrt
import scipy.io as sio
import egd
import oiadmm
from adapdict import *


class AdaDictL1(AdapDict):
    def __init__(self, dimension, accuracy, sparse_method, sparse_parameters):
        print "\t\t L1 Loss Dictionary"
        super(AdaDictL1, self).__init__(dimension, accuracy, sparse_method, sparse_parameters)
        self._L = self._D

	# Set name of dictionary loss function
	self._Dloss = "L1"

    def train(self, x, beta=0.5):
        '''
        Train the dictionary on a single observation
        '''
        # compute coefficients
        alpha = self.sparsecode(x)
        # accuracy of fit:
        recon_err = (np.linalg.norm(x - np.dot(self._D, alpha)) \
                /(np.linalg.norm(x)+10**-6))
        start_err_bonus = max(0,(DICT_SLOWDOWN - self._natoms + 0.0) / DICT_SLOWDOWN)
        recon_err -= start_err_bonus/10
        # If the coding is not good enough, add the x to dictionary
        if recon_err > self._acc and self._natoms < DICT_MAX_ATOMS:
            self.appendtoD(x)
        # else update the dictionary
        else:
            # IOADMM update
            tau = 1.0/(2*max(np.dot(alpha,alpha),1e-6))
            self._D, self._L = oiadmm.OIADMM(self._D, alpha, x, self._L, beta, tau)
            # normalize columns
            for j in range(self._natoms):
                self._D[:,j] = self._D[:,j] / max(10**-6, np.linalg.norm(self._D[:,j], 1))
        # remove near duplications from D
        if self._natoms > DICT_MAX_ATOMS - 1:
            # only remove when there are enough atoms
            self.removeduplicatesD()

        # increase counter of number of samples trained
        self._ntrained += 1

    def appendtoD(self, x):
        col = x.reshape((x.shape[0], 1)) / (np.linalg.norm(x) + 10**-8)
        self._D = np.append(self._D, col, axis=1)
        self._natoms += 1


'''
train = "./matlab/X_test.mat"
test = "./matlab/y_test.mat"
# load data
X = sio.loadmat(train)
y = sio.loadmat(test)
X = X['X'].T
y = y['y'].T
dim = X.shape[1]

ad = AdaDictL1(dim, 0.5, 'lasso', 0.1)

print ad

ad.batchtrain(X)
ad.batchreconstruction(y, 'ytest_n')

print ad
ad.dimagesave((5,2), 'test')
''' 
