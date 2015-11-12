from NeuralNet_batch import *
import pdb
from numpy import *
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib.cm as cm

# X is data matrix (each row is a data point)
# Y is desired output (1 or -1)
# scoreFn is a function of a data point
# values is a list of values to plot


def plot(LRpredictor, X, y, grid_size, filename):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size),
                         indexing='ij')
    flatten = lambda m: np.array(m).reshape(-1,)

    result = []
    for (i, j) in itertools.product(range(grid_size), range(grid_size)):
        point = np.array([xx[i, j], yy[i, j]]).reshape(1, 2)
        result.append(LRpredictor(point))
    #print 'resutl',result
    Z = np.array(result).reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                 cmap=cm.Paired,
                 extend='both',
                 alpha=0.8)
    plt.scatter(flatten(X[:, 0]), flatten(X[:, 1]),
                c=flatten(y))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.savefig(filename)
#    plt.show()

def demo():
	filename = "../hw3_resources/toy_multiclass_2_train.csv"
	T = scipy.io.loadmat(filename)['toy_data']
	X = T[:,0:2]
	Y = T[:,2]
	ListDic = {}
	ListDic[1] = [1,0,0]
	ListDic[2] = [0,1,0]
	ListDic[3] = [0,0,1]
	Data = []
	for i in range(X.shape[0]):
		y = Y[i]
		fart = list((X[i,:].tolist(),ListDic[y]))
		Data.append(fart)
		
	filename = "../hw3_resources/toy_multiclass_2_validate.csv"
	T = scipy.io.loadmat(filename)['toy_data']
	X = T[:,0:2]
	Y = T[:,2]
	Datav = []
	for i in range(X.shape[0]):
		y = Y[i]
		fart = list((X[i,:].tolist(),ListDic[y]))
		Datav.append(fart)
		
	filename = "../hw3_resources/toy_multiclass_2_test.csv"
	T = scipy.io.loadmat(filename)['toy_data']
	X = T[:,0:2]
	Y = T[:,2]
	Datat = []
	for i in range(X.shape[0]):
		y = Y[i]
		fart = list((X[i,:].tolist(),ListDic[y]))
		Datat.append(fart)
	
	# make sure the data looks right
	iter = 500
	LR = 1
	HLs = [2,5,10,50]
	LDs = [0,1e-8,1e-6,1e-4,1e-2,1]
	for i,hidlay in enumerate(HLs):
		for j,Lambda in enumerate(LDs):
			fig, ax = ppl.subplots()
			outfile = 'Decision_boundry_'+str(hidlay)+'layer'+'_'+str(Lambda)
			for repeat in range(1):
				NN = MLP_NeuralNetwork(2, hidlay, 3, iterations = iter, learning_rate = LR, Lambda = Lambda)
				[train_err, val_err, test_err, I, LOSS] = NN.train(outfile, Data, Datav, Datat)
				plot(NN.predict, X, Y, grid_size=300, filename='test'+str(hidlay)+str(Lambda)+'.pdf')
				print str(hidlay), str(Lambda), train_err, val_err, test_err
				

if __name__ == '__main__':
    demo()