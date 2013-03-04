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
        super(AdaDictL1, self).__init__(dimension, accuracy, sparse_method, sparse_parameters)
        self._L = self._D

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
        if self._natoms > 0:
            # only remove when there are enough atoms
            self.removeduplicatesD()

        # increase counter of number of samples trained
        self._ntrained += 1

    def appendtoD(self, x):
        col = x.reshape((x.shape[0], 1)) / (np.linalg.norm(x) + 10**-8)
        self._D = np.append(self._D, col, axis=1)
        self._natoms += 1

    def removeduplicatesD(self):
        '''
        Possibly merge columns if they are too similar
        '''
        sim = abs(np.dot(self._D.T, self._D)) - np.identity(self._natoms)

        most_correlated = np.unravel_index(np.argmax(sim), sim.shape)
        if (sim[most_correlated] > DICT_MAX_CORR) and (self._natoms-1 not in most_correlated):
            # index where we will merge
            merger = most_correlated[0]
            # index that will be removed
            remover = most_correlated[1]

            # Merge in D
            self._D[:, merger] = 0.5 * self._D[:, merger] \
                + 0.5 * self._D[:, remover]
            self._D = np.delete(self._D, remover, 1)
            self._natoms -= 1

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
