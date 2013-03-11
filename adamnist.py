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
'''
parser = argparse.ArgumentParser(description=
        'MNIST: Encode sparse dictionary and fit model')
parser.add_argument('n_obs',
        help='number of observations')
parser.add_argument('dict_fit',
        help="model for fitting dictionary (linreg, lasso, lars)")
parser.add_argument('dict_loss',
        help='loss function: l1 or l2')
parser.add_argument('dict_reg',
        help='regularization in sparse encoding')


params = parser.parse_args(sys.argv[1:])

N_OBS = int(params.n_obs)
DICT_FIT = params.dict_fit
DICT_LOSS = params.dict_loss
DICT_REG = float(params.dict_reg)
'''

DICT_ACC = 0.5


def showimage(x):
    img = np.reshape(x, (28, 28), order = 'F')
    imgplot = plt.imshow(img)
    plt.show()

def run(n_obs, loss, reg, amount):
    print 'Starting run of MNIST.py'

    mnist_train = sio.loadmat('./data/mnist/MNIST_train.mat')
    mnist_test = sio.loadmat('./data/mnist/MNIST_test.mat')

    X_train = mnist_train['X'][0][0][2].transpose()
    y_train = mnist_train['y']
    X_test = mnist_test['Xtest'].transpose()
    y_test = mnist_test['ytest']
    dim = X_train.shape[1]

    ## Dictionary
    if loss == "l1":
        dictmod = adadictl1.AdaDictL1(dim, DICT_ACC, reg, amount)
    else:
        dictmod = adadictl2.AdaDictL2(dim, DICT_ACC, reg, amount)


    # Train model
    dictmod.batchtrain(X_train[range(n_obs)])

    # Find reconstructions
    al1_train = dictmod.batchreconstruction(X_train[range(n_obs)],
        'mnist_train')
    al1_test = dictmod.batchreconstruction(X_test,
        'mnist_test')

    ## Classification
    ogd = multiOGD(10, dictmod.getnatoms(), 0.001)
    ogd.train(al1_train , y_train[range(n_obs)])
    ogd.train(al1_train , y_train[range(n_obs)])
    ogd.seteta(0.0005)
    ogd.train(al1_train , y_train[range(n_obs)])
    ogd.train(al1_train , y_train[range(n_obs)])
    ogd.seteta(0.0001)
    ogd.train(al1_train , y_train[range(n_obs)])
    ogd.train(al1_train , y_train[range(n_obs)])
    ogd.seteta(0.00001)
    ogd.train(al1_train , y_train[range(n_obs)])
    ogd.train(al1_train , y_train[range(n_obs)])
    ogd.train(al1_train , y_train[range(n_obs)])
    ogd.seteta(0.000001)
    ogd.train(al1_train , y_train[range(n_obs)])
    ogd.train(al1_train , y_train[range(n_obs)])
    ogd.train(al1_train , y_train[range(n_obs)])
    ogd.train(al1_train , y_train[range(n_obs)])
    ogd.train(al1_train , y_train[range(n_obs)])

    ogd.predict(al1_test, y_test)
    imagenm = "mnist_%d" % n_obs
    dictmod.dimagesave((28,28), imagenm)

    print 'Run of MNIST.py is complete!\n\n'

'''
Atoms: 200
Reg: 0.05  too much
'''


N_OBS = 10000

print "\tL1 \tLASSO \t0.00005"
run(N_OBS, 'l1', 'lasso', 0.00005)
print "\tL2 \tLASSO \t0.0005"
run(N_OBS, 'l2', 'lasso', 0.0005)

print "\tL1 \tKL \t1.0"
run(N_OBS, 'l1', 'kl', 1.0)
print "\tL2 \tKL \t1.0"
run(N_OBS, 'l2', 'kl', 1.0)

print "\tL1 \tNone"
run(N_OBS, 'l2', 'linreg', 0.00001)
print "\tL2 \tNone"
run(N_OBS, 'l1', 'linreg', 0.00001)

N_OBS = 60000

print "\tL1 \tLASSO \t0.00005"
run(N_OBS, 'l1', 'lasso', 0.00005)
print "\tL2 \tLASSO \t0.0005"
run(N_OBS, 'l2', 'lasso', 0.0005)

print "\tL1 \tKL \t1.0"
run(N_OBS, 'l1', 'kl', 1.0)
print "\tL2 \tKL \t1.0"
run(N_OBS, 'l2', 'kl', 1.0)

print "\tL1 \tNone"
run(N_OBS, 'l2', 'linreg', 0.00001)
print "\tL2 \tNone"
run(N_OBS, 'l1', 'linreg', 0.00001)
