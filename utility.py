import os
from time import gmtime, strftime
import sys
import scipy.io as sio
from scipy.sparse import csr_matrix
import numpy as np
from numpy import cos as cos
from numpy import sin as sin
import matplotlib.pyplot as plt

pi = 3.141592654


def savematrix(matrix, folder, filename):
    sparsematrix = csr_matrix(matrix)
    d = os.path.dirname(folder)
    if not os.path.exists(d):
        os.makedirs(d)
    sio.savemat(folder + filename + '.mat', {"alphas": sparsematrix})


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = ''
        self.logfile = open("HISTORY.LOG", "a")

    def write(self, message):
        self.terminal.write(message)
        if message != '\n':
            ct = strftime("%Y-%m-%d %H:%M:%S", gmtime())
            self.log += ('%s: %s' % (ct, message))
        else:
            self.log += message

    def __del__(self):
        self.log += '\n\n'
        self.logfile.write(self.log)
        self.logfile.close()


def gabor2DFunction(frequency, rotation, gamma=1, etha=1, m=28, n=28):
        '''
        Implementation of normalized 2-D Gabor filter function
        defined by Kyrki V, 2002
        '''
        fmax = 1/14.0
        a = 0.5
        flist = [(a ** k) * fmax for k in xrange(frequency)]
        thetalist = [pi * angle / rotation for angle in xrange(rotation)]

        set = [(f, theta)  for f in flist for theta in thetalist] 
        xy_index = np.transpose(np.nonzero(np.ones((m,n)))) # Get x,y index of image
        
        # Image coordinates are centered to origin
        x = xy_index[:, 0]
        x = x - (max(x) + min(x))/2
        y = xy_index[:, 1]
        y = y - (max(y)+min(y))/2
                
        # Get Gabor function for all frequencies and rotations
        # then superpositioning for all rotations
        gb = []
        for f, theta in set:        
            x_new = x * cos(theta) + y * sin(theta)
            x_new = np.ceil(x_new)         
            
            y_new = -1 * x * sin(theta) + y * cos(theta)
            y_new = np.ceil(y_new)        
            
            N = float(f ** 2) / float(pi * gamma * etha)
            term2d = -1 * f ** 2 * (x_new ** 2 / gamma ** 2 + y_new ** 2 / etha ** 2)
            gb.append(N * np.exp(term2d) * cos((2 * pi * f * x_new)))

        return [np.reshape(np.reshape(f, (m, n)), (m * n, 1), order='F') for f in gb]

def test():
    g = gabor2DFunction(2,4)
    for i in xrange(2*4):
        plt.imshow(np.reshape(g[i],(28,28),order='F'))
        plt.show()
