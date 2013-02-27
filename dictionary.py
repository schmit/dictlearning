import numpy as np
import numpy.linalg
import numpy.random
import scipy.io as sio
import time
from sklearn import linear_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import utility

# Constants regarding dictionary update
# Move later
DICT_UPD_TOLERANCE = 10 ** -3
DICT_UPD_MAX_ITR = 50
# Dictionary tolerance doesnt seem to have much influence on accuraccy nor computation time
# Dictionary max itr seems to be never reached, but a nice fail safe


class Dictionary:
    def __init__(self, dimension, nr_atoms, method, method_parameters, init):
        methods = ['linreg', 'lars', 'lasso']
        inits = ['randn', 'randu', 'gabor']
        # check the method is implemented
        assert(method in methods)
        # check the initilization is implemented
        assert(init in inits)

        # Dictionary dimensions
        self.__dim = dimension
        self.__natoms = nr_atoms
        # initialization method for dictionary
        self.__dinit = init

        # Dictionary matrix
        self.initdictionary(init)
        # A and B as defined on page 25 of Mairal
        self.__A = np.zeros((nr_atoms, nr_atoms))
        self.__B = np.zeros((dimension, nr_atoms))

        # Sparse coding method
        self.__method_par = method_parameters
        self.initcodingmethod(method, method_parameters)

        # Extras
        self.__ntrained = 0

    def __repr__(self):
        string = "\t===Sparse dictionary model===\n"
        string += "Dimension: %d\n" % (self.__dim)
        string += "Number of atoms: %d\n" % (self.__natoms)
        return string

    def initdictionary(self, init):
        '''
        Initialize the dictionary
        Several initializations could be used:
        - first few sample points
        - DCT
        - Wavelets
        (see Mairal p.29)

        Note the dimension of the dictionary should be (dim,nr_atoms)
        '''

        # for now: return random matrix
        if init == 'randn':
            D = np.random.randn(self.__dim, self.__natoms)
            self.__D = D
        elif init == 'randu':
            D = np.random.rand(self.__dim, self.__natoms)
            self.__D = D
        elif init == 'gabor':
            n = max(self.__dim, self.__natoms)
            atoms = utility.gabor2DFunction(int(np.ceil(n/4)),4)
            D = np.zeros((self.__dim, self.__natoms))
        
            atoms = atoms[:self.__natoms]
            print len(atoms)
            for idx, atom in enumerate(atoms):
                D[:, idx] = np.transpose(atom) / np.linalg.norm(atom)
            self.__D = D


    def initcodingmethod(self, method, method_parameters):
        if method == 'linreg':
            def fn(D, x, par):
                return np.linalg.lstsq(D, x)[0]
        elif method == 'lars':
            def fn(D, x, par):
                clf = linear_model.LassoLars(alpha = method_parameters, fit_intercept = False, fit_path = False, \
                    normalize=False)
                clf.fit(D, x)
                return clf.coef_[0]
        elif method == 'lasso':
            def fn(D, x, par):
                clf = linear_model.Lasso(alpha = method_parameters, fit_intercept = False)
                clf.fit(D, x)
                return clf.coef_
        self.method_fn = fn

    def train(self, x):
        '''
        Train the dictionary on a single observation
        '''
        # compute coefficients
        alpha = self.sparsecode(x)

        # update matrices
        self.updateAB(x, alpha)
        # update dictionary
        self.updateD(DICT_UPD_TOLERANCE, DICT_UPD_MAX_ITR)

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
            self.train(X[j, :])
        print "Trained sample in %0.2f seconds" % (time.time() - time_start)

    def reconstruction(self, x):
        '''
        code a sample in dictionary, return reconstruction error (2-norm)
        '''
        alpha = self.sparsecode(x)
        # reconstruction
        recon = np.dot(self.__D, alpha)
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
        sum_err = 0

        # the encodings
        alphas = np.zeros((X.shape[0], self.__natoms))

        # loop over the batch
        for j in range(n):
            recon = self.reconstruction(X[j, :])
            # extract encoding
            alpha_j = recon[0]
            # add encoding to matrix
            alphas[j, :] = alpha_j
            sum_nnz += np.linalg.norm(alpha_j, 0)
            sum_err += recon[1]
            # print iteration number
            if (j + 1) % 10000 == 0:
                print "Iteration %d" % (j + 1)
        avg_nnz = sum_nnz / n
        avg_err = sum_err / n
        print "Average fraction of nonzero coefficients: %f" % (avg_nnz / self.__natoms)
        print "Average relative l2 error on reconstruction set is: %0.2f" % (avg_err)

        # save encoding to matlab file for later use
        folder = "./encodings/%s/" % (self.__dinit)
        filename = "%r_%r" % (self.__natoms, int(self.__method_par * 100000))
        utility.savematrix(alphas, folder, filename)
        # timing
        print "Reconstructed sample in %0.2f seconds" % (time.time() - time_start)

        return alphas

    def sparsecode(self, x):
        '''
        Subroutine of train
        Find a sparse coding of x in terms of the columns of the dictionary
        '''
        x = x - np.mean(x)
        alpha = self.method_fn(self.__D, x, self.__method_par)
        return alpha

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
                    u = (self.__B[:, j] - np.dot(self.__D, self.__A[:, j])) / self.__A[j, j] + self.__D[:, j]
                    self.__D[:, j] = u / max(np.linalg.norm(u), 1)

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

    def getA(self):
        return self.__A

    def getB(self):
        return self.__B

    def getD(self):
        return self.__D

    def setD(self, matrix):
        self.__D = matrix

    def dimagesave(self, dimensions, imname):
        '''
        Saves all the atoms of the dictionary as images

        dimensions: dimensions of images
        '''
        print "=== Save figures ==="

        # Make directory if it does not exist
        folder = "./images/%s/%s_%d_%d/" % (imname, self.__dinit, self.__natoms, int(self.__method_par * 100000))
        d = os.path.dirname(folder)
        if not os.path.exists(d):
            os.makedirs(d)

        for i in range(self.__natoms):
            img = np.reshape(self.__D[:, i], dimensions, order='F')
            imgplot = plt.imshow(img)
            filename = "%d" % (i)
            plt.savefig(folder + filename + '.png')

'''
train = "./matlab/X_test.mat"
test = "./matlab/y_test.mat"
# load data
X = sio.loadmat(train)
y = sio.loadmat(test)
X = X['X'].T
y = y['y'].T
dim = X.shape[1]

lassodn = Dictionary(dim,12, 'lasso', 0.5, 'randn')


print lassodn
lassodn.batchtrain(X)
lassodn.batchreconstruction(y, 'ytest_n')

lassodn.dimagesave((5,2), 'test')
'''