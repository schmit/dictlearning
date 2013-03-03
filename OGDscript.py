import numpy as np
import scipy.io as sio
import math
from multiOGD import *
from kernels import *

print 'Loading data'

mnist_train = sio.loadmat('./data/mnist/MNIST_train.mat')
mnist_test = sio.loadmat('./data/mnist/MNIST_test.mat')
y_train = mnist_train['y']
y_test = mnist_test['ytest']

enc_train_file = 'encodings/mnist_train_s/lasso/100_100.mat'
enc_test_file = 'encodings/mnist_test_s/lasso/100_100.mat'
enc_train = sio.loadmat(enc_train_file)
enc_test = sio.loadmat(enc_test_file)

A_train = enc_train['alphas'].todense()
A_test = enc_test['alphas'].todense()

X_train = mnist_train['X'][0][0][2].transpose()
X_test = mnist_test['Xtest'].transpose()

print A_train.shape
print X_train.shape

print 'Loading complete'

print '\t\t\t====== Dictionary ======'

# train model
dictmod = multiOGD(10, 100, 0.001)
dictmod.train(A_train, y_train)
# predict
dictmod.predict(A_test, y_test)

print '\t\t\t===== No dictionary ======'

# train model
model = multiOGD(10, 784, 0.001)
model.train(X_train, y_train)
# predict
model.predict(X_test, y_test)
