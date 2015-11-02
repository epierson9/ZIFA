import numpy as np
from scipy.optimize import curve_fit, minimize
from sklearn.decomposition import FactorAnalysis
from copy import deepcopy
from collections import Counter
import random
from scipy.stats import multivariate_normal
import warnings

"""
Zero-inflated factor analysis (ZIFA). Performs dimensionality reduction on zero-inflated data. 
Created by Emma Pierson and Christopher Yau. 

Sample usage:
Z, model_params = fitModel(Y, k)

where Y is the observed zero-inflated data, k is the desired number of latent dimensions, and Z is the low-dimensional projection. 

See example.py for a full example.
"""
def mult_diag(d, mtx, left=True):
    """
    Multiply a full matrix by a diagonal matrix.
    This function should always be faster than dot.
    Input:
    d -- 1D (N,) array (contains the diagonal elements)
    mtx -- 2D (N,N) array
    Returns:
    mult_diag(d, mts, left=True) == dot(diag(d), mtx)
    mult_diag(d, mts, left=False) == dot(mtx, diag(d))
    """
    if left:
        return (d*mtx.T).T
    else:
        return d*mtx
def checkNoNans(matrix_list):
	"""
	Returns false if any of the matrices are nans or infinite. 
	Checked. 
	"""
	for i, M in enumerate(matrix_list):
		if np.any(np.isnan(np.array(M))) or np.any(np.isinf(np.array(M))):
			raise Exception('Matrix index %i in list has a NaN or infinite element' % i)
			
			
def invertFast(A, d):
	"""
	given an array A of shape d x k and a d x 1 vector d, computes (A * A.T + diag(d)) ^{-1}
	Checked.
	"""
	assert(A.shape[0] == d.shape[0])
	assert(d.shape[1] == 1)
	k = A.shape[1]
	A = np.array(A)
	d_vec = np.array(d)
	d_inv = np.array(1/d_vec[:, 0])
	inv_d_squared = np.dot(np.atleast_2d(d_inv).T, np.atleast_2d(d_inv))	
	M = np.diag(d_inv) -  inv_d_squared * np.dot(np.dot(A, np.linalg.inv(np.eye(k, k) + np.dot(A.T, mult_diag(d_inv, A)))), A.T)
	return M
def Estep(Y, A, mus, sigmas, decay_coef):
	"""
	estimates the requisite latent expectations in the E-step. 
	Checked. 
	Input: 
	Y: observed data. 
	A, mus, sigmas, decay_coef: parameters. 
	Returns: 
	EZ, EZZT, EX, EXZ, EX2: latent expectations. 
	"""
	assert(len(Y[0]) == len(A))
	N, D = Y.shape
	D, K = A.shape
	assert((sigmas.shape[0] == D) and (sigmas.shape[1] == 1))
	assert((mus.shape[0] == D) and (mus.shape[1] == 1))
	EX = np.zeros([N, D])
	EXZ = np.zeros([N, D, K])#this is a 3D tensor. 
	EX2 = np.zeros([N, D])
	EZ = np.zeros([N, K])
	EZZT = np.zeros([N, K, K])
	for i in range(N):
		#compute P(Z, X_0 | Y) following three step formula. 
		#1. compute P(Z, X_0)
		Y_i = Y[i, :]
		Y_is_zero = np.abs(Y_i) < 1e-6
		zero_indices = np.array([True for a in range(K)] + [np.abs(Y_i[j]) < 1e-6 for j in range(D)])
		dim = K + D#this is dimension of matrix
		
		#2. compute P(Z, X_0 | Y_+)
		mu_c, sigma_c, augmentedA_0, augmentedA_plus, augmented_D, sigma_22_inv = calcConditionalDistribution(A, mus, sigmas, np.array([np.abs(Y_i[j]) < 1e-6 for j in range(D)]), Y_i[~Y_is_zero])
		
		#3. compute P(Z, X_0 | Y_+, Y_0)
		dim = len(sigma_c)
		matrixToInvert = computeMatrixInLastStep(A, np.abs(Y[i, :]) < 1e-6, sigmas, K, sigma_c, decay_coef, sigma_22_inv)
		if (Y_is_zero).sum() < D:
			magical_matrix = 2 * decay_coef * (mult_diag(augmented_D, matrixToInvert) + augmentedA_0 * (np.eye(K)-augmentedA_plus.T * sigma_22_inv * augmentedA_plus) * (augmentedA_0.T * matrixToInvert))
		else:
			magical_matrix = 2 * decay_coef * (mult_diag(augmented_D, matrixToInvert) + augmentedA_0 * (augmentedA_0.T * matrixToInvert))
		magical_matrix[:, :K] = 0
		sigma_11 = augmentedA_0 * augmentedA_0.T + np.diag(augmented_D)
		if (Y_is_zero).sum() < D:
			sigma_xz = np.array(sigma_c - mult_diag(augmented_D, np.array(magical_matrix), left = False) -  (magical_matrix *augmentedA_0) * ((np.eye(K) - augmentedA_plus.T * sigma_22_inv * augmentedA_plus) * augmentedA_0.T))
		else:
			sigma_xz = np.array(sigma_c - mult_diag(augmented_D, np.array(magical_matrix), left = False) -  (magical_matrix *augmentedA_0) * augmentedA_0.T)
		mu_xz = np.array(np.matrix(np.eye(dim) - magical_matrix) * np.matrix(mu_c))
		EZ[i, :] = mu_xz[:K, 0]
		EX[i, Y_is_zero] = mu_xz[K:, 0]
		EX2[i, Y_is_zero] = mu_xz[K:, 0] ** 2 + np.diag(sigma_xz[K:, K:])
		EZZT[i, :, :] = np.dot(np.atleast_2d(mu_xz[:K, :]), np.atleast_2d(mu_xz[:K,:].transpose()))  + sigma_xz[:K, :K]
		EXZ[i, Y_is_zero, :] =  np.dot(mu_xz[K:], mu_xz[:K].transpose()) + sigma_xz[K:, :K]	
	return EZ, EZZT, EX, EXZ, EX2
def applyWoodburyIdentity(A_inv, B_inv, C):
	"""
	uses Woodbury identity to compute inverse of (A + C*B*C.T)
	"""
	A_inv = np.matrix(A_inv)
	B_inv = np.matrix(B_inv)
	C = np.matrix(C)
	
	A_inv_C = (A_inv * C)
	M = A_inv - A_inv_C  * np.linalg.inv(B_inv + C.T * A_inv_C) * A_inv_C.T
	return M
def computeMatrixInLastStep(A, zero_indices, sigmas, K, sigma_c, decay_coef, sigma_22_inv):
	"""
	Optimized matrix method for the E-step. 
	This computes (1 + 2*decay_coef * I_x * sigma_c) ^ {-1}
	zero_indices should have length D. 
	"""
	A_0 = np.matrix(A[zero_indices, :])
	A_plus = np.matrix(A[~zero_indices, :])
	sigmas_0 = sigmas[zero_indices]
	sigmas_plus = sigmas[~zero_indices]
	E_xx = sigma_c[K:, :][:, K:]
	E_xz = sigma_c[K:, :][:, :K]
	E_00_prime_inv = np.matrix(invertFast(A_0, sigmas_0 ** 2 + 1 / (2. * decay_coef)))
	E_plusplus_inv = sigma_22_inv
	E_0plus = A_0*A_plus.T;
	if (E_plusplus_inv.shape[0] == 0) or (E_plusplus_inv.shape[1] == 0):
		inv_matrix = (1/(2. * decay_coef)) * E_00_prime_inv
	elif (A_0.shape[0] < A_0.shape[1]):
		E_plusplus = A_plus*A_plus.T + np.diag(sigmas_plus[:, 0] ** 2)
		inv_matrix = np.linalg.inv(2. * decay_coef * (np.linalg.inv(E_00_prime_inv) - E_0plus * E_plusplus_inv * E_0plus.T))
	else:
		b_inv = np.linalg.inv((np.matrix(A_0).T * E_00_prime_inv) * np.matrix(A_0))
		innermost_inverse = applyWoodburyIdentity(-E_plusplus_inv, b_inv, np.matrix(A_plus))
		inv_matrix = (1 / (2. * decay_coef)) * (E_00_prime_inv - (E_00_prime_inv * A_0)*(A_plus.T * innermost_inverse * A_plus)*(A_0.T * E_00_prime_inv))
	dim = len(sigma_c)
	M = np.zeros([dim, dim])
	M[:K, :K] = np.eye(K)
	M[K:, :K] = -2 * decay_coef * np.dot(inv_matrix, E_xz)
	M[K:, K:] = inv_matrix
	return np.array(M)
	
def Mstep(Y, EZ, EZZT, EX, EXZ, EX2, oldA, old_mus, old_sigmas, old_decay_coef, singleSigma = False):
	"""
	estimates parameters given the expectations computed in the E-step. 
	Input: 
	Y: observed values.
	EZ, EZZT, EX, EXZ, EX2: expectations of latent variables computed in E-step. 
	oldA, old_mus, old_sigmas, old_decay_coef: old parameters. 
	singleSigma: only estimates one sigma as opposed to sigma_j for all j. 
	Returns: 
	A, mus, sigmas, decay_coef: new values of parameters. 
	"""
	assert(len(Y) == len(EZ))
	N, D = Y.shape
	N, K = EZ.shape
	
	#First estimate A and mu
	A = np.zeros([D, K])
	mus = np.zeros([D,1])
	sigmas = np.zeros([D,1])
	Y_is_zero = np.abs(Y) < 1e-6
	#make B, which is the same for all J. 
	B = np.eye(K + 1)
	for k1 in range(K):
		for k2 in range(K):
			B[k1][k2] = sum(EZZT[:, k1, k2])
		B[K, :K] = EZ.sum(axis = 0)
		B[:K, K] = EZ.sum(axis = 0)
	B[K, K] = N
	tiled_EZ = np.tile(np.resize(EZ, [N, 1, K]), [1, D, 1])
	tiled_Y = np.tile(np.resize(Y, [N, D, 1]), [1, 1, K])
	tiled_Y_is_zero = np.tile(np.resize(Y_is_zero, [N, D, 1]), [1, 1, K])
	c = np.zeros([K + 1, D])
	c[K, :] += (Y_is_zero * EX + (1 - Y_is_zero) * Y).sum(axis = 0)
	c[:K, :] = (tiled_Y_is_zero * EXZ + (1 - tiled_Y_is_zero) * tiled_Y * tiled_EZ).sum(axis = 0).transpose()
	solution = np.dot(np.linalg.inv(B), c)
	A = solution[:K, :].transpose()
	mus = np.atleast_2d(solution[K, :]).transpose()

	#then optimize sigma
	EXM = np.zeros([N, D])#have to figure these out  after updating mu. 
	EM = np.zeros([N, D])
	EM2 = np.zeros([N, D])
	
	tiled_mus = np.tile(mus.transpose(), [N, 1])
	tiled_A = np.tile(np.resize(A, [1, D, K]), [N, 1, 1])
	
	EXM = (tiled_A * EXZ).sum(axis = 2) + tiled_mus * EX
	test_sum = (tiled_A * tiled_EZ).sum(axis = 2)
	A_product = np.tile(np.reshape(A, [1, D, K]), [K, 1, 1]) * (np.tile(np.reshape(A, [1, D, K]), [K, 1, 1]).T)
	for i in range(N):
		EM[i, :] = (np.dot(A, EZ[i, :].transpose()) + mus.transpose())#this should be correct
		EZZT_tiled = np.tile(np.reshape(EZZT[i, :, :], [K, 1, K]), [1, D, 1])
		ezzt_sum = (EZZT_tiled * A_product).sum(axis = 2).sum(axis = 0)
		EM2[i, :] = ezzt_sum + 2 * test_sum[i, :]*tiled_mus[i, :] + tiled_mus[i, :] ** 2
	sigmas = (Y_is_zero * (EX2 - 2*EXM + EM2) + (1 - Y_is_zero) * (Y**2 - 2*Y*EM + EM2)).sum(axis = 0)
	sigmas = np.atleast_2d(np.sqrt(sigmas / N)).transpose()
	if singleSigma:
		sigmas = np.mean(sigmas) * np.ones(sigmas.shape)
	decay_coef = minimize(lambda x:decayCoefObjectiveFn(x, Y, EX2), old_decay_coef, jac = True, bounds = [[1e-8, np.inf]])
	decay_coef = decay_coef.x[0]
	return A, mus, sigmas, decay_coef

	
def calcConditionalDistribution(A, mus, sigmas, zero_indices, observed_values):
	"""
	Computes the distribution of X and Z conditional on the NONZERO values of Y. Matrix computations optimized. 
	Input: 
	A, mus, sigmas are parameters. 
	zero_indices has length D and zero_indices[j] is zero iff Y[j] is zero. 
	observed_values are the values of Y at the nonzero indices
	Output: various matrices used in the rest of the E-step. 
	
	"""
	D, K = A.shape
	dim = D + K
	#mu_x and sigma_x here are the PRIOR DISTRIBUTIONS over [Z, X]
	mu_x = np.zeros([dim, 1])
	mu_x[K:dim, :] = mus
	augmentedA = np.matrix(np.zeros([dim, K]))
	augmentedA[:K, :] = np.eye(K)
	augmentedA[K:, :] = A
	mu_x = np.atleast_2d(mu_x)
	observed_values = np.atleast_2d(observed_values)
	
	if len(observed_values) == 1:
		observed_values = observed_values.transpose()
	mu_diff = np.matrix(np.atleast_2d(observed_values - mus[~zero_indices]))
	assert(mu_diff.shape[0] == len(observed_values) and mu_diff.shape[1] == 1)
	assert(len(zero_indices) == D)
	assert(zero_indices.sum() == D - len(observed_values))
	augmented_zero_indices = np.array([True for a in range(K)] + list(zero_indices))
	#1 denotes the zero entries, 2 denotes the nonzero entries
	augmentedA_0 = augmentedA[augmented_zero_indices, :]
	augmentedA_plus = augmentedA[~augmented_zero_indices, :]
	sigma_11 = augmentedA_0 * augmentedA_0.T
	sigma_11[K:, K:] = sigma_11[K:, K:] +  np.diag(sigmas[zero_indices][:, 0] ** 2)
	augmented_D = np.array([0 for i in range(K)] + list(sigmas[zero_indices][:, 0] ** 2))
	
	
	if len(observed_values) == 0:
		sigma_x = augmentedA * augmentedA.T
		sigma_x[K:, K:] = sigma_x[K:, K:] + np.diag(sigmas[zero_indices][:, 0] ** 2)
		return  mu_x, sigma_x, augmentedA_0, augmentedA_plus, augmented_D, np.array([[]])
	sigma_22_inv = np.matrix(invertFast(A[~zero_indices, :], sigmas[~zero_indices] ** 2))
	mu_0 = mu_x[augmented_zero_indices, :] + augmentedA_0 * (augmentedA_plus.T * (sigma_22_inv * mu_diff))
	sigma_0 = sigma_11 - augmentedA_0 * (augmentedA_plus.T * sigma_22_inv * augmentedA_plus) * augmentedA_0.T
	assert((mu_0.shape[0] == zero_indices.sum() + K) and (mu_0.shape[1] == 1))
	assert(sigma_0.shape[0] == zero_indices.sum() + K and sigma_0.shape[1] == zero_indices.sum() + K)
	return np.array(mu_0), np.array(sigma_0), augmentedA_0, augmentedA_plus, augmented_D, sigma_22_inv

def decayCoefObjectiveFn(x, Y, EX2): 
	"""
	Computes the objective function for terms involving lambda in the M-step. 
	Checked. 
	Input: 
	x: value of lambda
	Y: the matrix of observed values
	EX2: the matrix of values of EX2 estimated in the E-step. 
	Returns: 
	obj: value of objective function
	grad: gradient
	"""
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		y_squared = Y ** 2
		Y_is_zero = np.abs(Y) < 1e-6
		exp_Y_squared = np.exp(-x * y_squared)
		log_exp_Y = np.nan_to_num(np.log(1 - exp_Y_squared))
		exp_ratio = np.nan_to_num(exp_Y_squared / (1 - exp_Y_squared))
		obj = sum(sum(Y_is_zero * (-EX2*x) + (1 - Y_is_zero) * log_exp_Y))
		grad = sum(sum(Y_is_zero * (-EX2) + (1 - Y_is_zero) * y_squared * exp_ratio))
		if type(obj) is np.float64:
			obj = -np.array([obj])
		if type(grad) is np.float64:
			grad = -np.array([grad])
		return obj, grad
def exp_decay(x, decay_coef):
	"""
	squared exponential decay function.
	"""	
	return np.exp(-decay_coef*(x**2))
def initializeParams(Y, K, singleSigma = False, makePlot = False):
	"""
	initializes parameters using a standard factor analysis model (on imputed data) + exponential curve fitting. 
	Checked. 
	Input: 
	Y: data matrix, n_samples x n_genes
	K: number of latent components
	singleSigma: uses only a single sigma as opposed to a different sigma for every gene 
	makePlot: makes a mu - p_0 plot and shows the decaying exponential fit. 
	Returns: 
	A, mus, sigmas, decay_coef: initialized model parameters. 
	"""
	
	N, D = Y.shape
	model = FactorAnalysis(n_components = K)
	zeroedY = deepcopy(Y)
	mus = np.zeros([D, 1])
	for j in range(D):
		non_zero_idxs = np.abs(Y[:, j]) > 1e-6
		mus[j] = zeroedY[:, j].mean()
		zeroedY[:, j] = zeroedY[:, j] - mus[j]
	model.fit(zeroedY)
	A = model.components_.transpose()
	sigmas = np.atleast_2d(np.sqrt(model.noise_variance_)).transpose()
	if singleSigma:
		sigmas = np.mean(sigmas) * np.ones(sigmas.shape)
	#now fit decay coefficient
	means = []
	ps = []
	for j in range(D):
		non_zero_idxs = np.abs(Y[:, j]) > 1e-6
		means.append(Y[non_zero_idxs, j].mean())
		ps.append(1 - non_zero_idxs.mean())
	decay_coef, pcov = curve_fit(exp_decay, means, ps, p0 = .05)
	decay_coef = decay_coef[0]
	mse = np.mean(np.abs(ps - np.exp(-decay_coef * (np.array(means) ** 2))))
	if (mse > 0) and makePlot:
		figure()
		scatter(means, ps)
		plot(np.arange(min(means), max(means), .1), np.exp(-decay_coef * (np.arange(min(means), max(means), .1) ** 2)))
		title('Decay Coef is %2.3f; MSE is %2.3f' % (decay_coef, mse))
		show()
	return A, mus, sigmas, decay_coef
def testInputData(Y):
	if (Y - np.array(Y, dtype = 'int32')).sum() < 1e-6:
		raise Exception('Your input matrix is entirely integers. It is possible but unlikely that this is correct: ZIFA takes as input LOG read counts, not read counts.')
	Y_is_zero = np.abs(Y) < 1e-6
	if (Y_is_zero).sum() == 0:
		raise Exception('Your input matrix contains no zeros. This is possible but highly unlikely in scRNA-seq data. ZIFA takes as input log read counts.')
	if (Y < 0).sum() > 0:
		raise Exception('Your input matrix contains negative values. ZIFA takes as input log read counts and should not contain negative values.')
	zero_fracs = Y_is_zero.mean(axis = 0)
	column_is_all_zero = zero_fracs == 1.
	if column_is_all_zero.sum() > 0:
		raise Exception("Your Y matrix has columns which are entirely zero; please filter out these columns and rerun the algorithm.")

def fitModel(Y, K, singleSigma = False):
	"""
	fits the model to data.
	Input: 
	Y: data matrix, n_samples x n_genes
	K: number of latent components
	singleSigma: if True, fit only a single variance parameter (zero-inflated PPCA) rather than a different one for every gene (zero-inflated factor analysis). 
	Returns: 
	EZ: the estimated positions in the latent space, n_samples x K
	params: a dictionary of model parameters. Throughout, we refer to lambda as "decay_coef". 
	"""
	N, D = Y.shape
	if D > 2000:
		print 'Warning: this dataset has a large number of genes. If ZIFA takes too long to run, try using block_ZIFA.py instead'
	testInputData(Y)
	print 'Running zero-inflated factor analysis with N = %i, D = %i, K = %i' % (N, D, K)
	#initialize the parameters
	np.random.seed(23)
	A, mus, sigmas, decay_coef = initializeParams(Y, K, singleSigma = singleSigma)
	checkNoNans([A, mus, sigmas, decay_coef])
	max_iter = 100
	param_change_thresh = 1e-2
	n_iter = 0
	while n_iter < max_iter:
		EZ, EZZT, EX, EXZ, EX2 = Estep(Y, A, mus, sigmas, decay_coef)
		new_A, new_mus, new_sigmas, new_decay_coef = Mstep(Y, EZ, EZZT, EX, EXZ, EX2, A, mus, sigmas, decay_coef, singleSigma = singleSigma)
		checkNoNans([EZ, EZZT, EX, EXZ, EX2, new_A, new_mus, new_sigmas, new_decay_coef])
		paramsNotChanging = True
		max_param_change = 0
		for new, old in [[new_A, A], [new_mus, mus], [new_sigmas, sigmas], [new_decay_coef, decay_coef]]:
			rel_param_change = np.mean(np.abs(new - old)) / np.mean(np.abs(new))
			if rel_param_change > max_param_change:
				max_param_change = rel_param_change
			if rel_param_change > param_change_thresh:
				paramsNotChanging = False
				break
		A = new_A
		mus = new_mus
		sigmas = new_sigmas
		decay_coef = new_decay_coef
		if paramsNotChanging:
			print 'Param change below threshold %2.3e after %i iterations' % (param_change_thresh, n_iter)
			break
		if n_iter >= max_iter:
			print 'Maximum number of iterations reached; terminating loop'
		n_iter += 1	
	EZ, EZZT, EX, EXZ, EX2 = Estep(Y, A, mus, sigmas, decay_coef)
	params = {'A':A, 'mus':mus, 'sigmas':sigmas, 'decay_coef':decay_coef}
	return EZ, params



