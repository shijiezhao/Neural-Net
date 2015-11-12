from NeuralNet_batch import *

def demo():
	filename = "mnist_train.csv"
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
		
	filename = "mnist_validate.csv"
	T = scipy.io.loadmat(filename)['va']
	X = T[:,0:784]
	Y = T[:,784]
	Datav = []
	for i in range(X.shape[0]):
		y = Y[i]
		fart = list((X[i,:].tolist(),ListDic[y]))
		Datav.append(fart)
		
	filename = "mnist_test.csv"
	T = scipy.io.loadmat(filename)['te']
	X = T[:,0:784]
	Y = T[:,784]
	Datat = []
	for i in range(X.shape[0]):
		y = Y[i]
		fart = list((X[i,:].tolist(),ListDic[y]))
		Datat.append(fart)
	
	# make sure the data looks right
	iter = 100
	LR = MM = 1
	Lambda = 1e-5
# 	outfile1 = 'Model_selection.txt'
# 	g = open(outfile1,'w')
	HLs = [2,5,10,20,100]
	LDs = [0,1e-6,1e-4,1e-2]
	for i,hidlay in enumerate(HLs):
		outfile1 = 'Model_Sel_'+str(hidlay)+'layer'+'.txt'
		g = open(outfile1,'w')
		for j,Lambda in enumerate(LDs):
			fig, ax = ppl.subplots()
			outfile = 'Outfile_'+str(hidlay)+'layer'+'_'+str(Lambda)+'.txt'
# 			f.write('Hidden units: ' +'\t'+ str(hidlay) + '\t' + 'lambda: ' + '\t' + str(Lambda) + '\n')
			VE=[]
			TE=[]
			for repeat in range(1):
				NN = MLP_NeuralNetwork(784, hidlay, 6, iterations = iter, learning_rate = LR, Lambda = Lambda)
				[train_err, val_err, test_err, I, LOSS] = NN.train(outfile, Data, Datav, Datat)
# 				ppl.plot(I,LOSS)
				print str(hidlay), str(Lambda), train_err, val_err, test_err
# 				f.write(str(val_err)+'\t'+str(test_err)+'\n')
				VE.append(val_err)
				TE.append(test_err)
# 			fig.savefig(outfile+'.pdf')
# 			g.write('Hidden units: ' +'\t'+ str(hidlay) + '\t' + 'lambda: ' + '\t' + str(Lambda) + ':\t' + str(np.mean(VE))+'\t'+str(np.std(VE))+'\t'+str(np.mean(TE))+'\t'+str(np.std(TE))+'\n')

if __name__ == '__main__':
    demo()