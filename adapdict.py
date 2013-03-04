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


class AdapDict:
    def __init__(self, dimension, accuracy, sparse_method, sparse_parameters):
        sparse_methods = ['linreg', 'lars', 'lassolars', 'lasso', 'kl']
        # check the method is implemented
        assert(sparse_method in sparse_methods)

        assert(accuracy < 1)
        assert(accuracy > 0)

        # Dictionary dimensions
        self.__dim = dimension
        self.__natoms = 1
        self.__acc = accuracy
        # initialization method for dictionary

        # Sparse coding method
        self.__sparse_par = sparse_parameters
        self.__sparse_name = sparse_method
        self.initcodingmethod(sparse_method, sparse_parameters)

        # Dictionary matrix, start with all ones
        self.__D = sqrt(1.0/self.__dim) * np.ones((self.__dim, 1))
        # A and B as defined on page 25 of Mairal
        self.__A = np.zeros((self.__natoms, self.__natoms))
        self.__B = np.zeros((self.__dim, self.__natoms))

        # Extras
        self.__ntrained = 0
        print "\t\t\tAdaptive Dictionary Initialized"


    def __repr__(self):
        string = "\t===Sparse dictionary model===\n"
        string += "Dimension: %d\n" % (self.__dim)
        string += "Number of atoms: %d\n" % (self.__natoms)
        return string

    def initcodingmethod(self, sparse_method, sparse_parameters):
        '''
        sparse_method for fitting the coefficients

        For dictionaries with large number of atoms, lars is fastest.
        larslasso performs very slow with large number of atoms.

        Also note that lars takes in the number of non-zero coefficients,
        while lasso and larslasso take in the regularization parameter lambda
        '''

        if sparse_method == 'linreg':
            def fn(D, x, par):
                return np.linalg.lstsq(D, x)[0]
        elif sparse_method == 'lars':
            def fn(D, x, par):
                clf = linear_model.Lars(fit_intercept=False, fit_path=True,
                    n_nonzero_coefs = par)
                clf.fit(D, x)
                return clf.coef_
        elif sparse_method == 'lassolars':
            def fn(D, x, par):
                clf = linear_model.LassoLars(alpha=sparse_parameters,
                    fit_intercept = False, fit_path=False, 
                    normalize=False)
                clf.fit(D, x)
                return clf.coef_[0]
        elif sparse_method == 'lasso':
            def fn(D, x, par):
                clf = linear_model.Lasso(alpha=sparse_parameters,
                    fit_intercept=False)
                clf.fit(D, x)
                return clf.coef_
        elif sparse_method == "kl":
            def fn(D, x, par):
                alpha = egd.egd(D,x,par)
                return alpha
        self.sparse_method_fn = fn

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
            # update matrices
            self.updateAB(x, alpha)
            # update dictionary
            self.updateD(DICT_UPD_TOLERANCE, DICT_UPD_MAX_ITR)

        # remove near duplications from D
        if self.__natoms > 0:
            # only remove when there are enough atoms
            self.removeduplicatesD()

        # increase counter of number of samples trained
        self.__ntrained += 1

    def batchtrain(self, X):
        '''
        Train a batch of training examples
        '''
        print "=== Batch training ==="
        time_start = time.time()
        xrange = X.shape[0]
        for j in range(xrange):
            if (j + 1) % (xrange / 10) == 0:
                print "Iteration %d" % (j + 1)
                print "Number of atoms: %d" % self.__natoms
            self.train(X[j, :])
        print "Trained sample in %0.2f seconds" % (time.time() - time_start)

    def reconstruction(self, x):
        '''
        code a sample in dictionary, return reconstruction error (2-norm)
        '''
        alpha = self.sparsecode(x)
        # reconstruction
        recon = np.squeeze(np.dot(self.__D, alpha))
        recon_err = np.linalg.norm(x - recon) / np.linalg.norm(x)
        return alpha, recon_err

    def batchreconstruction(self, X, dataname):
        '''
        code a batch of samples, compute average reconstruction error (2-norm)
        '''

        print "=== Reconstruction ==="
        time_start = time.time()

        n = X.shape[0]
        # sum of non-zero coefficients
        sum_nnz = 0.0
        # sum of errors
        sum_err = 0.0

        # the encodings
        alphas = np.zeros((X.shape[0], self.__natoms))

        # loop over the batch
        for j in range(n):
            recon = self.reconstruction(X[j, :])
            # extract encoding
            alpha_j = np.squeeze(recon[0])
            # add encoding to matrix

            alphas[j, :] = alpha_j
            sum_nnz += np.linalg.norm(alpha_j, 0)
            sum_err += recon[1]
            # print iteration number
            if (j + 1) % 10000 == 0:
                print "Iteration %d" % (j + 1)
        avg_nnz = sum_nnz / n
        avg_err = sum_err / n
        print "Average fraction of nonzero coefficients: %f" % \
            (avg_nnz / self.__natoms)
        print "Average relative l2 error on reconstruction set is: %0.2f" % \
            (avg_err)

        # save encoding to matlab file for later use
        folder = "./encodings/%s/%s/" % (dataname, self.__sparse_name)
        filename = "%r_%r" % (self.__natoms, int(self.__sparse_par * 100000))
        utility.savematrix(alphas, folder, filename)
        # timing
        print "Reconstructed sample in %0.2f seconds" % \
            (time.time() - time_start)

        return alphas

    def sparsecode(self, x):
        '''
        Subroutine of train
        Find a sparse coding of x in terms of the columns of the dictionary
        '''
        a = self.sparse_method_fn(self.__D, x, self.__sparse_par)
        return a

    def updateAB(self, x, alpha):
        '''
        Subroutine of train
        Update matrices A and B
        '''
        # line 5, 6 of Mairal p25
        self.__A = self.__A + np.outer(alpha, alpha)
        self.__B = self.__B + np.outer(x, alpha)

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
        norm_D_old = np.linalg.norm(self.__D)
        while not conv:
            # update the columns of D
            for j in range(self.__natoms):
                # only update columns that are used
                if self.__A[j, j] != 0:
                    u = (self.__B[:, j] - np.dot(self.__D, self.__A[:, j])) \
                        / self.__A[j, j] + self.__D[:, j]
                    self.__D[:, j] = u / max(np.linalg.norm(u), 10**-6)

            # check the new norm for D
            norm_D_new = np.linalg.norm(self.__D)
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
        self.__D = np.append(self.__D, col, axis=1)
        # increase size of A
        # column
        self.__A = np.append(self.__A, np.zeros((self.__natoms, 1)), axis=1)
        # row
        self.__A = np.append(self.__A, np.zeros((1, self.__natoms+1)), axis=0)
        # add zero column to B
        self.__B = np.append(self.__B, np.zeros((self.__dim, 1)), axis=1)
        # increase atom count
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

            # Merge in A
            newA = self.__A

            # update the merger row and column
            newA[merger, :] = 0.5 * self.__A[merger, :] \
                + 0.5 * self.__A[remover, :]
            newA[:, merger] = 0.5 * self.__A[:, merger] \
                + 0.5 * self.__A[:, remover]
            # remove the remover row and column
            newA = np.delete(newA, remover, 0)
            newA = np.delete(newA, remover, 1)

            self.__A = newA

            # Merge in B
            self.__B[:, merger] = 0.5 * self.__B[:, merger] \
                + 0.5 * self.__B[:, remover]
            self.__B = np.delete(self.__B, remover, 1)

            self.__natoms -= 1

    def dimagesave(self, dimensions, imname):
        '''
        Saves all the atoms of the dictionary as images

        dimensions: dimensions of images
        '''
        print "=== Save figures ==="

        # Make directory if it does not exist
        folder = "./images/%s/%s/%d_%d/" % (imname, self.__sparse_name,
            self.__natoms, int(self.__sparse_par * 100000))
        d = os.path.dirname(folder)
        if not os.path.exists(d):
            os.makedirs(d)

        for i in range(self.__natoms):
            img = np.reshape(self.__D[:, i], dimensions, order='F')
            imgplot = plt.imshow(img)
            filename = "%d" % (i)
            plt.savefig(folder + filename + '.png')

    def getA(self):
        return self.__A

    def getB(self):
        return self.__B

    def getD(self):
        return self.__D

    def setD(self, matrix):
        self.__D = matrix

    def getnatoms(self):
        return self.__natoms


train = "./matlab/X_test.mat"
test = "./matlab/y_test.mat"
# load data
X = sio.loadmat(train)
y = sio.loadmat(test)
X = X['X'].T
y = y['y'].T
dim = X.shape[1]

ad = AdapDict(dim, 0.5, 'lasso', 0.5)

print ad

ad.batchtrain(X)
ad.batchreconstruction(y, 'ytest_n')

print ad
ad.dimagesave((5,2), 'test')

