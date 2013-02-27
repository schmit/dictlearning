import numpy as np
import scipy.io as sio
import math
from multiOGD import *
from kernels import *

# No kernel

print '\t\t\t===== No kernel ======'
# load data
print 'Loading data'
mnist_train = sio.loadmat('./data/mnist/MNIST_train.mat')
mnist_test = sio.loadmat('./data/mnist/MNIST_test.mat')
X_train = mnist_train['X'][0][0][2].transpose()
y_train = mnist_train['y']
X_test = mnist_test['Xtest'].transpose()
y_test = mnist_test['ytest']
del mnist_train
del mnist_test
print 'Loading complete'

# train model
model = multiOGD(10, 784, 0.001)
model.train(X_train, y_train)
# predict
model.predict(X_test, y_test)


# Kernel method
print '\t\t\t===== Kernel ====='

# load data
print 'Loading data'
mnist_small_train = sio.loadmat('./data/mnist/MNIST_train2k.mat')
mnist_small_test = sio.loadmat('./data/mnist/MNIST_test2k.mat')
X_small_train = mnist_small_train['X'].transpose()
y_small_train = mnist_small_train['y'].transpose()-1
X_small_test = mnist_small_test['Xtest'].transpose()
y_small_test = mnist_small_test['ytest'].transpose()-1
del mnist_small_train
del mnist_small_test
print 'Loading complete'

# linear kernel
print "\t\t===Training with LINEAR kernel==="
lin_kernelmod = multiOGD(10, 2000, 0.01)
lin_kernelmod.kerneltrain(X_small_train, y_small_train, linearkernel)
lin_kernelmod.kernelpredict(X_small_train, y_small_train,  linearkernel)
lin_kernelmod.kernelpredict(X_small_test, y_small_test, linearkernel)

# gaussian kernel
print "\t\t===Training with GAUSSIAN kernel==="
gau_kernelmod = multiOGD(10,2000,0.01)
gau_kernelmod.kerneltrain(X_small_train, y_small_train, gaussiankernel,1.0)
gau_kernelmod.kernelpredict(X_small_train, y_small_train,  gaussiankernel,1.0)
gau_kernelmod.kernelpredict(X_small_test, y_small_test, gaussiankernel,1.0)

# sigma = 1.2 : 0.870000
# sigma = 1.1 : 0.873000
# sigma = 1.1 : 0.874000 eta = 0.01
# sigma = 1.1 :  0.879000 eta = 500
# : : 0.882500 eta = 1000

# sigma = 1 : 0.8735
# sigma = 0.7 : 0.868
# sigma = 0.5 : 0.869
# sigma = 0.2 : 0.857

# laplace kernel
print "\t\t===Training with LAPLACE kernel==="
lap_kernelmod = multiOGD(10,2000,200.0)
lap_kernelmod.kerneltrain(X_small_train, y_small_train, laplacekernel,1.5)
lap_kernelmod.kernelpredict(X_small_train, y_small_train,  laplacekernel,1.5)
lap_kernelmod.kernelpredict(X_small_test, y_small_test, laplacekernel,1.5)

# sigma = 1: 0.832
# eta ~ 200 gets good results


