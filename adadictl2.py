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

        # A and B as defined on page 25 of Mairal
        self._A = np.zeros((self._natoms, self._natoms))
        self._B = np.zeros((self._dim, self._natoms))

    def train(self, x):
        '''
        Train the dictionary on a single observation
        '''
        # compute coefficients
        alpha = self.sparsecode(x)

        # accuracy of fit:
        recon_err = (np.linalg.norm(x - np.dot(self._D, alpha)) \
                /(np.linalg.norm(x)+10**-6))
        start_err_bonus = max(0,(DICT_SLOWDOWN - self._natoms + 0.0) / DICT_SLOWDOWN)
        #recon_err -= start_err_bonus/10

        # If the coding is not good enough, add the x to dictionary
        if recon_err > self._acc and self._natoms < DICT_MAX_ATOMS:
            self.appendtoD(x)
        # else update the dictionary
        else:
            # update matrices
            self.updateAB(x, alpha)
            # update dictionary
            self.updateD(DICT_UPD_TOLERANCE, DICT_UPD_MAX_ITR)

        # remove near duplications from D
        if self._natoms > 0:
            # only remove when there are enough atoms
            self.removeduplicatesD()

        # increase counter of number of samples trained
        self._ntrained += 1

    def updateAB(self, x, alpha):
        '''
        Subroutine of train
        Update matrices A and B
        '''
        # line 5, 6 of Mairal p25
        self._A = self._A + np.outer(alpha, alpha)
        self._B = self._B + np.outer(x, alpha)

    def updateD(self, tol, max_itr):
        '''
        Subroutine of train
        Update the dictionary

        Input:
        tolerance for norm change in dictionary update
        max number of iterations to converge
        '''
        # Algorithm 2 of Mairal p25
        conv = False
        itr = 0
        norm_D_old = np.linalg.norm(self._D)
        while not conv:
            # update the columns of D
            for j in range(self._natoms):
                # only update columns that are used
                if self._A[j, j] != 0:
                    u = (self._B[:, j] - np.dot(self._D, self._A[:, j])) \
                        / self._A[j, j] + self._D[:, j]
                    self._D[:, j] = u / max(np.linalg.norm(u), 10**-6)

            # check the new norm for D
            norm_D_new = np.linalg.norm(self._D)
            norm_diff = abs(norm_D_new - norm_D_old)
            norm_D_old = norm_D_new
            if norm_diff < tol:
                conv = True
                # print the number of iterations till convergence
                # print 'converged after %d iterations' %(itr)
                # print 'difference in norm %0.2e' %(norm_diff)

            # check number of iterations
            itr += 1
            if itr > max_itr:
                conv = True
                # print 'not converged after %d iterations' %(max_itr)
                # print 'current norm difference: %0.2e' %(norm_diff)

    def appendtoD(self, x):
        col = x.reshape((x.shape[0], 1)) / (np.linalg.norm(x) + 10**-8)
        self._D = np.append(self._D, col, axis=1)
        # increase size of A
        # column
        self._A = np.append(self._A, np.zeros((self._natoms, 1)), axis=1)
        # row
        self._A = np.append(self._A, np.zeros((1, self._natoms+1)), axis=0)
        # add zero column to B
        self._B = np.append(self._B, np.zeros((self._dim, 1)), axis=1)
        # increase atom count
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

            # Merge in A
            newA = self._A

            # update the merger row and column
            newA[merger, :] = 0.5 * self._A[merger, :] \
                + 0.5 * self._A[remover, :]
            newA[:, merger] = 0.5 * self._A[:, merger] \
                + 0.5 * self._A[:, remover]
            # remove the remover row and column
            newA = np.delete(newA, remover, 0)
            newA = np.delete(newA, remover, 1)

            self._A = newA

            # Merge in B
            self._B[:, merger] = 0.5 * self._B[:, merger] \
                + 0.5 * self._B[:, remover]
            self._B = np.delete(self._B, remover, 1)

            self._natoms -= 1


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

