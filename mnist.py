import numpy as np
import scipy.io as sio
import dictionary
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from multiOGD import *
from kernels import *
import sys
import argparse
import utility

sys.stdout = utility.Logger()
print 'Starting run of MNIST.py'

parser = argparse.ArgumentParser(description='MNIST: Encode sparse dictionary and fit model')
parser.add_argument('dict_fit', help="model for fitting dictionary (linreg, lasso, lars)")
parser.add_argument('dict_init', help='initialization of dictionary')
parser.add_argument('dict_atoms', help='nr of atoms in dictionary')
parser.add_argument('dict_reg', help='regularization in sparse encoding')
parser.add_argument('mod_reg', help='regularization svm fit')

params = parser.parse_args(sys.argv[1:])

DICT_FIT = params.dict_fit
DICT_INIT = params.dict_init
DICT_ATOMS = int(params.dict_atoms)
DICT_REG = float(params.dict_reg)
MOD_REG = float(params.mod_reg)

print params


def showimage(x):
    img = np.reshape(x, (28, 28), order = 'F')
    imgplot = plt.imshow(img)
    plt.show()

mnist_train = sio.loadmat('./data/mnist/MNIST_train.mat')
mnist_test = sio.loadmat('./data/mnist/MNIST_test.mat')

X_train = mnist_train['X'][0][0][2].transpose()
y_train = mnist_train['y']
X_test = mnist_test['Xtest'].transpose()
y_test = mnist_test['ytest']
dim = X_train.shape[1]

## Dictionary
lasso_d = dictionary.Dictionary(dim, DICT_ATOMS, DICT_FIT, DICT_REG, DICT_INIT)

lasso_d.batchtrain(X_train)

# Save dictionary atoms as images
lasso_d.dimagesave((28, 28), 'mnist')

# Find reconstructions
alphas_train = lasso_d.batchreconstruction(X_train, 'mnist_train')
alphas_test = lasso_d.batchreconstruction(X_test, 'mnist_test')

## Classification
ogd_m = multiOGD(10, DICT_ATOMS, MOD_REG)
ogd_m.train(alphas_train, y_train)
ogd_m.predict(alphas_test, y_test)

print 'Run of MNIST.py is complete!'
'''
Atoms: 200
Reg: 0.05  too much
'''
