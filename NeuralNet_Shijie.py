"""
Implement Neural Net myself... Don't trust others' code...
@Author: Shijie Zhao; @Cambridge; @Nov-08-2015
@Math background: MIT M.L. 6.867 lecture 13
"""
import math
import random
import numpy as np
import scipy.io
import prettyplotlib as ppl
import matplotlib.pyplot as plt
np.seterr(all = 'ignore')

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# derivative of sigmoid
def dsigmoid(x):
    return np.exp(-x) / ((1 + np.exp(-x)) ** (2))

class MLP_NeuralNetwork(object):

	## 1. Initialize the data structure
	def __init__(self, input, hidden, output, iterations, learning_rate, Lambda):
		## 1.1. initialize parameters
		self.iterations = iterations
		self.learning_rate = learning_rate
		self.Lambda = Lambda
		
		## 1.2. initialize arrays parameters
		self.input = input + 1 # add 1 for bias node
		self.hidden = hidden
		self.output = output
		
		## 1.3. initialize the middle structures
		self.ax = np.ones((self.input,1))
		self.a1 = np.ones((self.hidden,1))
		self.az = np.ones((self.hidden,1))
		self.a2 = np.ones((self.output,1))
		self.ao = np.ones((self.output,1))
		
		## 1.4. initialize weight matrices
		input_range = 1.0 / self.input ** (2)
		output_range = 1.0 / self.hidden ** (2)
		self.w1 = np.random.normal(loc = 0, scale = input_range, size = (self.hidden, self.input))
		self.w2 = np.random.normal(loc = 0, scale = output_range, size = (self.output, self.hidden))
		
	## 2. FeedForward propagation, and return the result
	def feedForward(self, inputs):
		## 2.1. re-initiate inputs
		self.ax[:-1,:] = np.array(inputs).reshape((self.input-1,1))
		
		## 2.2. hidden layer propagation
		self.a1 = np.dot(self.w1, self.ax)
		self.az = sigmoid(self.a1)
		
		## 2.3. output layer propagation
		self.a2 = np.dot(self.w2, self.az)
		self.ao = sigmoid(self.a2)
		return self.ao
		
	## 3. Back propagation, and return error
	def backPropagate(self, targets):
		## 3.1. Calculate dw2 matrix
		targets = np.array([targets])
		targets = targets.transpose()   ## to make the targets vector in a k*1 dimension
		delta2 = self.ao - targets
		dw2 = np.outer(delta2, self.az)
		
		## 3.2. Calculate dw1 matrix 
		dsa1 = dsigmoid(self.a1[:,0])
		didsa1 = np.diag(dsa1)
		temp = np.dot(didsa1, self.w2.transpose())
		delta1 = np.dot(temp, delta2)
		dw1 = np.outer(delta1, self.ax)
		
		## 3.3. Regularization
		dw2 = dw2 + 2*self.Lambda*self.w2
		dw1 = dw1 + 2*self.Lambda*self.w1
		
		## 3.4. Update matrix value
		self.w2 = self.w2 - self.learning_rate * dw2
		self.w1 = self.w1 - self.learning_rate * dw1
		
		## 3.5. Return loss function
		l1 = -np.dot(targets.transpose(), np.log(self.ao)) - np.dot((1-targets.transpose()), np.log(1-self.ao))
		reg = self.Lambda * (np.sum(self.w1**2) + np.sum(self.w2**2))
		loss = l1[0][0] + reg
		return loss
		
	## 4. Train the model	
	def train(self, outfile, train, validate, test):
		## 4.1. iteration
		for i in range(self.iterations):
			## 4.1.1. Shuffle data as for Stochastic Gradient Descent
			random.shuffle(train)
			## 4.1.2. For each data, do G.D.
			error = 0
			for tr in train:
				inputs = tr[0]
				targets = tr[1]
				self.feedForward(inputs)
				error += self.backPropagate(targets)
			error = error/len(train)
			print i, error
	
	## 5. Make prediction
	def predict(self, X):
		predictions = []
		for p in X:
			predictions.append(self.feedForward(p))
		return predictions
        
        
        