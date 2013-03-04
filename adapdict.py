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


class AdapDict(object):
    def __init__(self, dimension, accuracy, sparse_method, sparse_parameters):
        sparse_methods = ['linreg', 'lars', 'lassolars', 'lasso', 'kl']
        # check the method is implemented
        assert(sparse_method in sparse_methods)

        assert(accuracy < 1)
        assert(accuracy > 0)

        # Dictionary dimensions
        self._dim = dimension
        self._natoms = 1
        self._acc = accuracy
        # initialization method for dictionary

        # Sparse coding method
        self._sparse_par = sparse_parameters
        self._sparse_name = sparse_method
        self.initcodingmethod(sparse_method, sparse_parameters)

        # Dictionary matrix, start with all ones
        self._D = sqrt(1.0/self._dim) * np.ones((self._dim, 1))

        # Extras
        self._ntrained = 0
        print "\t\t\tAdaptive Dictionary Initialized"

    def __repr__(self):
        string = "\t===Sparse dictionary model===\n"
        string += "Dimension: %d\n" % (self._dim)
        string += "Number of atoms: %d\n" % (self._natoms)
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
        print "Error: train is not implemented"

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
                print "Number of atoms: %d" % self._natoms
            self.train(X[j, :])
        print "Trained sample in %0.2f seconds" % (time.time() - time_start)

    def reconstruction(self, x):
        '''
        code a sample in dictionary, return reconstruction error (2-norm)
        '''
        alpha = self.sparsecode(x)
        # reconstruction
        recon = np.squeeze(np.dot(self._D, alpha))
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
        alphas = np.zeros((X.shape[0], self._natoms))

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
            (avg_nnz / self._natoms)
        print "Average relative l2 error on reconstruction set is: %0.2f" % \
            (avg_err)

        # save encoding to matlab file for later use
        folder = "./encodings/%s/%s/" % (dataname, self._sparse_name)
        filename = "%r_%r" % (self._natoms, int(self._sparse_par * 100000))
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
        a = self.sparse_method_fn(self._D, x, self._sparse_par)
        return a


    def appendtoD(self, x):
        col = x.reshape((x.shape[0], 1)) / (np.linalg.norm(x) + 10**-8)
        self._D = np.append(self._D, col, axis=1)
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

            self._natoms -= 1

    def dimagesave(self, dimensions, imname):
        '''
        Saves all the atoms of the dictionary as images

        dimensions: dimensions of images
        '''
        print "=== Save figures ==="

        # Make directory if it does not exist
        folder = "./images/%s/%s/%d_%d/" % (imname, self._sparse_name,
            self._natoms, int(self._sparse_par * 100000))
        d = os.path.dirname(folder)
        if not os.path.exists(d):
            os.makedirs(d)

        for i in range(self._natoms):
            img = np.reshape(self._D[:, i], dimensions, order='F')
            imgplot = plt.imshow(img)
            filename = "%d" % (i)
            plt.savefig(folder + filename + '.png')

    def getD(self):
        return self._D

    def setD(self, matrix):
        self._D = matrix

    def getnatoms(self):
        return self._natoms
