import numpy as np
import scipy.io as sio
import adadictl1
import adadictl2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from multiOGD import *
from kernels import *
import sys
import argparse
import utility

sys.stdout = utility.Logger()
print 'Starting run of MNIST.py'

parser = argparse.ArgumentParser(description=
        'MNIST: Encode sparse dictionary and fit model')
parser.add_argument('dict_fit',
        help="model for fitting dictionary (linreg, lasso, lars)")
parser.add_argument('dict_acc',
        help='desired accuracy')
parser.add_argument('dict_reg',
        help='regularization in sparse encoding')
parser.add_argument('mod_reg',
        help='regularization svm fit')

params = parser.parse_args(sys.argv[1:])

DICT_FIT = params.dict_fit
DICT_ACC = float(params.dict_acc)
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
modl1 = adadictl1.AdaDictL1(dim, DICT_ACC, DICT_FIT, DICT_REG)

n_obs = 2000

# Train model
modl1.batchtrain(X_train[range(n_obs)])

# Find reconstructions
alphas_train = modl1.batchreconstruction(X_train[range(n_obs)],
    'mnist_train')
alphas_test = modl1.batchreconstruction(X_test[range(n_obs)],
    'mnist_test')

print modl1

## Classification
ogd_m = multiOGD(10, modl1.getnatoms(), MOD_REG)
ogd_m.train(alphas_train, y_train[range(n_obs)])
ogd_m.predict(alphas_test, y_test[range(n_obs)])

# Save dictionary atoms as images
modl1.dimagesave((28, 28), 'mnist')

print 'Run of MNIST.py is complete!'

'''
Atoms: 200
Reg: 0.05  too much
'''

