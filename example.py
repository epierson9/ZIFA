import ZIFA, block_ZIFA
import numpy as np
from pylab import *
import random
from copy import deepcopy
from sklearn.decomposition import FactorAnalysis

def generateSimulatedDimensionalityReductionData(n_clusters, n, d, k, sigma, decay_coef):
	"""
	generates data with multiple clusters. 
	Checked. 
	"""
	mu = 3
	range_from_value = .1
	
	if n_clusters == 1:
		Z = np.random.multivariate_normal(mean = np.zeros([k,]), cov = np.eye(k), size = n).transpose()
		cluster_ids =  np.ones([n,])
	else:
		Z = np.zeros([k, n])
		cluster_ids = np.array([random.choice(range(n_clusters)) for i in range(n)])
		for id in list(set(cluster_ids)):
			idxs = cluster_ids == id
			cluster_mu = (np.random.random([k,]) - .5) * 5
			Z[:, idxs] = np.random.multivariate_normal(mean = cluster_mu, cov = .05 * np.eye(k), size = idxs.sum()).transpose()
			
	A = np.random.random([d, k]) - .5
	mu = np.array([(np.random.uniform() * range_from_value * 2 + (1 - range_from_value)) * mu for i in range(d)])
	sigmas = np.array([(np.random.uniform() * range_from_value * 2 + (1 - range_from_value)) * sigma for i in range(d)])
	noise = np.zeros([d, n])
	for j in range(d):
		noise[j, :] = mu[j] + np.random.normal(loc = 0, scale = sigmas[j], size = n)
	X = (np.dot(A, Z) + noise).transpose()
	Y = deepcopy(X)
	Y[Y < 0] = 0
	rand_matrix = np.random.random(Y.shape)
	
	cutoff = np.exp(-decay_coef * (Y ** 2))
	zero_mask = rand_matrix < cutoff
	Y[zero_mask] = 0
	print 'Fraction of zeros: %2.3f; decay coef: %2.3f' % ((Y == 0).mean(), decay_coef)

	return X, Y, Z.transpose(), cluster_ids

def testAlgorithm():
	random.seed(30)
	np.random.seed(32)
	n = 200
	d = 20
	k = 2
	sigma = .3
	n_clusters = 3
	decay_coef = .1
	X, Y, Z, ids = generateSimulatedDimensionalityReductionData(n_clusters, n, d, k, sigma, decay_coef)

	Zhat, params = ZIFA.fitModel(Y, k)
	colors = ['red', 'blue', 'green']
	cluster_ids = sorted(list(set(ids)))
	model = FactorAnalysis(n_components = k)
	factor_analysis_Zhat = model.fit_transform(Y)
	figure(figsize = [15, 5])
	subplot(131)
	for id in cluster_ids:
		scatter(Z[ids == id, 0], Z[ids == id, 1], color = colors[id - 1], s = 4)
		title('True Latent Positions\nFraction of Zeros %2.3f' % (Y == 0).mean())
		xlim([-4, 4])
		ylim([-4, 4])
	subplot(132)
	for id in cluster_ids:
		scatter(Zhat[ids == id, 0], Zhat[ids == id, 1], color = colors[id - 1], s = 4)
		xlim([-4, 4])
		ylim([-4, 4])
		title('ZIFA Estimated Latent Positions')
		#title(titles[method])
	subplot(133)
	for id in cluster_ids:
		scatter(factor_analysis_Zhat[ids == id, 0], factor_analysis_Zhat[ids == id, 1], color = colors[id - 1], s = 4)
		xlim([-4, 4])
		ylim([-4, 4])
		title('Factor Analysis Estimated Latent Positions')
	
	
	show()
	

if __name__ == '__main__':
	testAlgorithm()