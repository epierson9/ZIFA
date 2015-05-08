import ZIFA
import numpy as np
from pylab import *
import random
from copy import deepcopy

def generateFactorAnalysisData(params):
	"""
	generates zero-inflated data. 
	Checked. 
	"""
	n = params['n']
	d = params['d']
	k = params['k']
	sigma = params['sigma']
	mu = params['mu']
	decay_coef = params['decay_coef']
	range_from_value = .1
	Z = np.random.multivariate_normal(mean = np.zeros([k,]), cov = np.eye(k), size = n).transpose()
	A = np.random.random([d, k]) - .5
	mu = np.array([(np.random.uniform() * range_from_value * 2 + (1 - range_from_value)) * mu for i in range(d)])
	sigmas = np.array([(np.random.uniform() * range_from_value * 2 + (1 - range_from_value)) * sigma for i in range(d)])
	noise = np.zeros([d, n])
	for j in range(d):
		noise[j, :] = mu[j] + np.random.normal(loc = 0, scale = sigmas[j], size = n)
	X = (np.dot(A, Z) + noise).transpose()
	Y = deepcopy(X)
	
	for i in range(n):
		for j in range(d):
			if np.random.random() < np.exp(-decay_coef * (X[i][j] ** 2)):
				Y[i][j] = 0
			
	assert((not np.isnan(X).any()) and (not np.isnan(Y).any()) and (not np.isnan(Z).any()))
	
	return X, Y, Z.transpose(), {'A':A, 'mus':mu, 'sigmas':sigmas, 'decay_coef':decay_coef}

def testAlgorithm():
	d = 10
	k = 2
	random.seed(30)
	np.random.seed(32)
	n = 200
	sigma = np.random.random() * .2 + .1
	mu = np.random.random() * .5 + 3
	decay_coef = np.random.random() * .2
	params = {'d': d, 'k': k, 'n': n, 'sigma': sigma, 'mu':mu, 'decay_coef':decay_coef}
	X, Y, Z, true_params = generateFactorAnalysisData(params)
	EZ, params = ZIFA.fitModel(Y, k)

if __name__ == '__main__':
	testAlgorithm()