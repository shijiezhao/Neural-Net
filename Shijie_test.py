from NeuralNet_Shijie import *

def demo():
	filename = "../hw3_resources/mnist_train.csv"
	T = scipy.io.loadmat(filename)['tr']
	X = T[:,0:784]
	Y = T[:,784]
	ListDic = {}
	ListDic[1] = [1,0,0,0,0,0]
	ListDic[2] = [0,1,0,0,0,0]
	ListDic[3] = [0,0,1,0,0,0]
	ListDic[4] = [0,0,0,1,0,0]
	ListDic[5] = [0,0,0,0,1,0]
	ListDic[6] = [0,0,0,0,0,1]
	Data = []
	for i in range(X.shape[0]):
		y = Y[i]
		fart = list((X[i,:].tolist(),ListDic[y]))
		Data.append(fart)
		
	filename = "../hw3_resources/mnist_validate.csv"
	T = scipy.io.loadmat(filename)['va']
	X = T[:,0:784]
	Y = T[:,784]
	Datav = []
	for i in range(X.shape[0]):
		y = Y[i]
		fart = list((X[i,:].tolist(),ListDic[y]))
		Datav.append(fart)
		
	filename = "../hw3_resources/mnist_test.csv"
	T = scipy.io.loadmat(filename)['te']
	X = T[:,0:784]
	Y = T[:,784]
	Datat = []
	for i in range(X.shape[0]):
		y = Y[i]
		fart = list((X[i,:].tolist(),ListDic[y]))
		Datat.append(fart)
	
	# make sure the data looks right
	hidlay = 5
	iter = 500
	LR = MM = 0.5
	Lambda = 1e-5
	outfile = 'test1'
	HLs = [10]
	LDs = [0,1e-6]
	for i,hidlay in enumerate(HLs):
		for j,Lambda in enumerate(LDs):
			outfile = 'SGD_outfile_HL__'+str(i)+'_'+str(j)
			NN = MLP_NeuralNetwork(784, hidlay, 6, iterations = iter, learning_rate = LR, Lambda = Lambda)
			NN.train(outfile, Data, Datav, Datat)

if __name__ == '__main__':
    demo()