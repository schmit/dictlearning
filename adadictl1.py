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
from adapdict import AdapDict

# Constants regarding dictionary update
# Move later
DICT_UPD_TOLERANCE = 10**-3
DICT_UPD_MAX_ITR = 50
# Dictionary tolerance doesnt seem to have much influence
#  on accuraccy nor computation time
# Dictionary max itr seems to be never reached, but a nice fail safe

# When to merge atoms when they are too similar
DICT_MAX_ATOMS = 100
DICT_MAX_CORR = 0.98

# Accepts lower precision for the first observations
# Select number here
DICT_SLOWDOWN = 100


class AdaDictL2(AdapDict):
    def __init__(self, dimension, accuracy, sparse_method, sparse_parameters):
        super(AdaDictL2, self).__init__(dimension, accuracy, sparse_method, sparse_parameters)

        # Dictionary matrix, start with all ones
        self.__D = sqrt(1.0/self.__dim) * np.ones((self.__dim, 1))
        # A and B as defined on page 25 of Mairal
        self.__A = np.zeros((self.__natoms, self.__natoms))
        self.__B = np.zeros((self.__dim, self.__natoms))

    def train(self, x):
        '''
        Train the dictionary on a single observation
        '''
        # compute coefficients
        alpha = self.sparsecode(x)
        # accuracy of fit:
        recon_err = (np.linalg.norm(x - np.dot(self.__D, alpha)) \
                /(np.linalg.norm(x)+10**-6))
        start_err_bonus = max(0,(DICT_SLOWDOWN - self.__natoms + 0.0) / DICT_SLOWDOWN)
        recon_err -= start_err_bonus/10
        # If the coding is not good enough, add the x to dictionary
        if recon_err > self.__acc and self.__natoms < DICT_MAX_ATOMS:
            self.appendtoD(x)
        # else update the dictionary
        else:
            ## IMPLEMENT UPDATE
            pass

        # remove near duplications from D
        if self.__natoms > 0:
            # only remove when there are enough atoms
            self.removeduplicatesD()

        # increase counter of number of samples trained
        self.__ntrained += 1

    def appendtoD(self, x):
        col = x.reshape((x.shape[0], 1)) / (np.linalg.norm(x) + 10**-8)
        self.__D = np.append(self.__D, col, axis=1)
        self.__natoms += 1

    def removeduplicatesD(self):
        '''
        Possibly merge columns if they are too similar
        '''
        sim = abs(np.dot(self.__D.T, self.__D)) - np.identity(self.__natoms)

        most_correlated = np.unravel_index(np.argmax(sim), sim.shape)
        if (sim[most_correlated] > DICT_MAX_CORR) and (self.__natoms-1 not in most_correlated):
            # index where we will merge
            merger = most_correlated[0]
            # index that will be removed
            remover = most_correlated[1]

            # Merge in D
            self.__D[:, merger] = 0.5 * self.__D[:, merger] \
                + 0.5 * self.__D[:, remover]
            self.__D = np.delete(self.__D, remover, 1)
            self.__natoms -= 1


train = "./matlab/X_test.mat"
test = "./matlab/y_test.mat"
# load data
X = sio.loadmat(train)
y = sio.loadmat(test)
X = X['X'].T
y = y['y'].T
dim = X.shape[1]

ad = AdaDictL2(dim, 0.5, 'kl', 0.1)

print ad

ad.batchtrain(X)
ad.batchreconstruction(y, 'ytest_n')

print ad
ad.dimagesave((5,2), 'test')

