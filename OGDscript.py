import numpy as np
import scipy.io as sio
import math
from multiOGD import *
from kernels import *
import numpy.linalg

from sklearn import svm

print 'Loading data'

mnist_train = sio.loadmat('./data/mnist/MNIST_train.mat')
mnist_test = sio.loadmat('./data/mnist/MNIST_test.mat')
y_train = mnist_train['y']
y_test = mnist_test['ytest']
'''
enc_train_file = 'encodings/mnist_train/L2/lasso/200_50.mat'
enc_test_file = 'encodings/mnist_test/L2/lasso/200_50.mat'
enc_train = sio.loadmat(enc_train_file)
enc_test = sio.loadmat(enc_test_file)

A_train = enc_train['alphas'].toarray()
A_test = enc_test['alphas'].toarray()

print A_train.shape
'''

X_train = mnist_train['X'][0][0][2].transpose()
X_test = mnist_test['Xtest'].transpose()

n_obs = 60000

X_train = X_train[range(n_obs)]
y_train = y_train[range(n_obs)]

print 'Loading complete'

print '\t\t\t====== Dictionary ======'

# train model
dictmod = multiOGD(10, 784, 0.001)
dictmod.train(X_train, y_train)
dictmod.seteta(0.0005)
dictmod.train(X_train, y_train)
dictmod.seteta(0.0001)
dictmod.train(X_train, y_train)
dictmod.seteta(0.00001)
dictmod.train(X_train, y_train)
dictmod.train(X_train, y_train)
dictmod.seteta(0.000001)
dictmod.train(X_train, y_train)
dictmod.train(X_train, y_train)




# predict
dictmod.predict(X_train, y_train)
dictmod.predict(X_test, y_test)

print '\t\t\t===== No dictionary ======'
'''
# train model
model = multiOGD(10, 784, 0.001)
model.train(X_train, y_train)
# predict
model.predict(X_test, y_test)
'''
'''
print '\t\t\t=== SVM ==='
A_clf = svm.LinearSVC(penalty='l2', loss='l1', dual=True)
X_clf = svm.LinearSVC(penalty='l2', loss='l1', dual=True)

A_clf.fit(A_train, y_train)
print (np.linalg.norm(A_clf.predict(A_test)==y_test, 1)+0.0)/10000

X_clf.fit(X_train, y_train)
print (np.linalg.norm(X_clf.predict(X_test)==y_test, 1)+0.0)/10000
'''
