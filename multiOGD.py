import numpy as np
from numpy.linalg import cholesky
from scipy.linalg import solve_triangular

class multiOGD:
    def __init__(self, n_classes, n_features, eta):
        if isinstance(n_classes, int) and isinstance(n_features, int) and isinstance(eta, float):
            self.__n_classes = n_classes
            self.__n_features = n_features
            self.__eta = eta
            self.__n_trained = 0
            # hinge loss
            self.__training_hl = 0
            # zero one loss
            self.__training_01l = 0

            # parameters stored as a numpy array: rows are features
            # columns are classes
            self.__params = np.zeros((n_classes,n_features))
        else:
            raise ValueError("incorrect parameters")

    def __repr__(self):
        string = "\t===Multiclass OGD model===\n"
        string += "Number of classes: %d\n" % (self.__n_classes)
        string += "Number of features: %d\n" % (self.__n_features)
        string += "Number of trained examples: %d\n" %(self.__n_trained)
        string += "Parameters:\n %r\n" %(self.__params)
        if self.__n_trained > 0:
            string += "Avg hinge loss: %f \tAvg zero-one loss: %f\n" %(self.__training_hl/self.__n_trained, float(self.__training_01l)/self.__n_trained)
        return string

    def train(self, xtrain, ytrain):
        print "\t ===Training model===="
        for i in range(len(ytrain)):
            self.train_sample(xtrain[i], int(ytrain[i]))

            # output to console
            if (i+1) % 20000 == 0:
                print "trained %d samples" % (i+1)
                self.printlosses()

    def kerneltrain(self, X, y, kernel, sigma=1):
        # create kernel matrix
        print '... form kernel ...'
        K = kernel(X, X, sigma) + np.identity(X.shape[0])*0.001
        print '... cholesky decomposition ...'
        L = cholesky(K)
        self.__L = L
        self.__X = X

        # train samples
        self.train(L, y)

    def predict(self, xtest, ytest):
        print "\t ===Predicting test set==="
        nr_correct = 0
        total_nr = ytest.size
        labels = [0 for i in range(self.n_classes())]
        # how often a label is shown in total
        tot_labels = [0 for i in range(self.n_classes())]

        for i in range(total_nr):
            pred = self.predict_sample(xtest[i])
            assert(isinstance(pred,int))
            nr_correct += (pred==int(ytest[i]))
            labels[int(ytest[i])] += (pred == int(ytest[i]))
            tot_labels[int(ytest[i])] += 1

        print "Accuracy: %f" % (float(nr_correct)/total_nr)
        print [round(float(labels[i])/tot_labels[i],2) for i in range(self.n_classes())]

    def kernelpredict(self, Xnew, ynew, kernel, sigma=1):
        print "\t ===Forming test set==="
        # form kernel vector for each observation
        print '... forming kernel'
        K_pred = kernel(self.__X, Xnew, sigma)

        # solve a linear system for each 
        print '... solving triangular system'
        Xhat = solve_triangular(self.__L,K_pred.transpose(), lower=True)        
        self.predict(Xhat.transpose(), ynew)

    def train_sample(self, X, y):
        '''
        Trains the model on example (X,y)
        input: X: np.array((n_features, 1)): features
               y: int: label (valid: 0 ... n_classes-1)
        no output
        '''
        if y not in range(self.__n_classes):
            raise ValueError("Class label wrong")
        self.__n_trained += 1
        hingelosses = self.calclosses(X,y)
        # maximum loss
        amhinge = hingelosses.argmax()
        # update weights (only if needed)
        assert(isinstance(y,int))
        if y != amhinge:
            self.__params[y] += self.__eta * X.transpose()
            self.__params[amhinge] -= self.__eta * X.transpose()

    def predict_sample(self, x):
        '''
        Predicts y based on features x
        input: x: np.array((n_features, 1)): features
        output: int: predicted label
        '''
        scores = np.dot(self.__params, x)
        return scores.argmax()

    def calclosses(self, X, y):
        # calculate w_y dot x_y
        wydotxy = np.dot(self.__params[y], X)
        # calculate w dot x for all w and x
        wdotx = np.dot(self.__params, X)
        # calculate losses for all labels
        losses = wdotx - wydotxy * np.ones((1,self.__n_classes)) + 1

        # update zero-one loss
        assert(isinstance(y,int))
        self.__training_01l += (y != losses[0].argmax())
        # subtract 1 from true label
        losses[0][y] -= 1

        # update hinge loss
        self.__training_hl += losses.max()

        # return losses
        return losses

    def printlosses(self):
        if self.__n_trained > 0:
            print "avg hinge loss: %f \t avg 0-1 loss: %f" % (self.__training_hl/self.__n_trained, float(self.__training_01l)/self.__n_trained)

    def n_classes(self):
        return self.__n_classes

    def seteta(self, eta):
        self.__eta = eta


'''
# Test code
model = multiOGD(3,4,0.1)
X = np.array([0, 3, 2, 0])
y = 1
model.train(X,y)
X = np.array([2, 0, 0, 1])
y = 2
model.train(X,y)
X = np.array([3, 0, 0, 1])
y = 2
model.train(X,y)

print model
'''
