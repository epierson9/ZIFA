from __future__ import print_function
import numpy as np
import random
from copy import deepcopy

from .ZIFA import checkNoNans, Estep, Mstep, initializeParams

"""

Block zero-inflated factor analysis (ZIFA). Performs dimensionality reduction on zero-inflated data.
Faster than ZIFA: uses block subsampling to increase efficiency.
Created by Emma Pierson and Christopher Yau.

Sample usage:

Z, model_params = fitModel(Y, k)

or

Z, model_params = fitModel(Y, k, n_blocks = 5)

where Y is the observed zero-inflated data (n_samples by n_genes), k is the desired number of latent dimensions, and Z is the low-dimensional projection
and n_blocks is the number of blocks to divide genes into. By default, the number of blocks is set to n_genes / 500 (yielding a block size of approximately 500).

Runtime is roughly linear in the number of samples and the number of genes, and quadratic in the block size.
Decreasing the block size may decrease runtime but will also produce less reliable results.

Runtime for block ZIFA on the full single-cell dataset from Pollen et al, 2014 (~250 samples, ~20,000 genes) is approximately 15 minutes on a quadcore Mac Pro.

"""


def generateIndices(n_blocks, N, D):
	"""
	generates indices for block matrix computation.
	Checked.
	Input:
	n_blocks: number of blocks to use.
	N: number of samples.
	D: number of genes.
	Output:
	y_indices_to_use[i][j] is the indices of block j in sample i.
	"""
	y_indices_to_use = []
	idxs = list(range(D))
	n_in_block = int(1. * D / n_blocks)

	for i in range(N):
		partition = []
		random.shuffle(idxs)
		n_added = 0

		for block in range(n_blocks):
			start = n_in_block * block
			end = start + n_in_block

			if block < n_blocks - 1:
				idxs_in_block = idxs[start:end]
			else:
				idxs_in_block = idxs[start:]

			partition.append(sorted(idxs_in_block))
			n_added += len(idxs_in_block)

		y_indices_to_use.append(partition)

		if i == 0:
			print('Block sizes', [len(a) for a in partition])

		assert(n_added == D)

	return y_indices_to_use


def combineMatrices(y_indices, all_EZs, all_EZZTs, all_EXs, all_EXZs, all_EX2s):
	"""
	Combines the expectations computed for each block into a single expectation.
	Checked.
	Input:
	y_indices[j] is the indices for block j.
	all_EZs[j] is the expected value of Z as computed using the indices in block j.
	and similarly for other expectations.
	Returns:
	the matrices of expectations required for the M-step.
	combined_EZ, combined_EZZT, combined_EX, combined_EXZ, combined_EX2
	Note that all these computations are for a single sample.
	"""
	n_blocks = len(all_EZs)
	D = sum([len(a) for a in y_indices])
	K = all_EZs[0].shape[1]

	combined_EX = np.zeros([D])
	combined_EXZ = np.zeros([D, K])
	combined_EX2 = np.zeros([D])
	combined_EZ = np.zeros([K])
	combined_EZZT = np.zeros([K, K])

	assert(len(all_EZs) == len(all_EZZTs) == len(all_EXs) == len(all_EXZs) == len(all_EX2s) == len(y_indices))

	for block in range(n_blocks):
		block_indices = y_indices[block]
		combined_EX[block_indices] = all_EXs[block]
		combined_EX2[block_indices] = all_EX2s[block]
		combined_EZ = combined_EZ + all_EZs[block]
		combined_EZZT = combined_EZZT + all_EZZTs[block]
		combined_EXZ[block_indices, :] = all_EXZs[block][0, :, :]

	combined_EZ = combined_EZ / n_blocks
	combined_EZZT = combined_EZZT / n_blocks

	return combined_EZ, combined_EZZT, combined_EX, combined_EXZ, combined_EX2


def testInputData(Y):
	if (Y - np.array(Y, dtype='int32')).sum() < 1e-6:
		raise Exception('Your input matrix is entirely integers. It is possible but unlikely that this is correct: ZIFA takes as input LOG read counts, not read counts.')

	Y_is_zero = np.abs(Y) < 1e-6
	if (Y_is_zero).sum() == 0:
		raise Exception('Your input matrix contains no zeros. This is possible but highly unlikely in scRNA-seq data. ZIFA takes as input log read counts.')

	if (Y < 0).sum() > 0:
		raise Exception('Your input matrix contains negative values. ZIFA takes as input log read counts and should not contain negative values.')

	zero_fracs = Y_is_zero.mean(axis=0)
	column_is_all_zero = zero_fracs == 1.

	if column_is_all_zero.sum() > 0:
		print("Warning: Your Y matrix has %i columns which are entirely zero; filtering these out before continuing." % (column_is_all_zero.sum()))
		Y = Y[:, ~column_is_all_zero]
	elif (zero_fracs > .9).sum() > 0:
		print('Warning: your Y matrix contains genes which are frequently zero. If the algorithm fails to converge, try filtering out genes which are zero more than 80 - 90% of the time, or using standard ZIFA.')

	return Y


def runEMAlgorithm(Y, K, singleSigma=False, n_blocks=None):
	Y = testInputData(Y)
	N, D = Y.shape

	if n_blocks is None:
		n_blocks = int(max(1, D / 500))
		print('Number of blocks has been set to %i' % n_blocks)

	print('Running block zero-inflated factor analysis with N = %i, D = %i, K = %i, n_blocks = %i' % (N, D, K, n_blocks))

	# Generate blocks.
	y_indices_to_use = generateIndices(n_blocks, N, D)

	# Initialize the parameters
	np.random.seed(23)
	A, mus, sigmas, decay_coef = initializeParams(Y, K, singleSigma=singleSigma)
	checkNoNans([A, mus, sigmas, decay_coef])

	max_iter = 20
	param_change_thresh = 1e-2
	n_iter = 0

	EZ = np.zeros([N, K])
	EZZT = np.zeros([N, K, K])
	EX = np.zeros([N, D])
	EXZ = np.zeros([N, D, K])
	EX2 = np.zeros([N, D])

	while n_iter < max_iter:
		for i in range(N):
			block_EZs = []
			block_EZZTs = []
			block_EXs = []
			block_EXZs = []
			block_EX2s = []

			for block in range(n_blocks):
				y_idxs = y_indices_to_use[i][block]
				Y_to_use = Y[i, y_idxs]
				A_to_use = A[y_idxs, :]
				mus_to_use = mus[y_idxs]
				sigmas_to_use = sigmas[y_idxs]

				block_EZ, block_EZZT, block_EX, block_EXZ, block_EX2 = Estep(np.array([Y_to_use]), A_to_use, mus_to_use, sigmas_to_use, decay_coef)
				block_EZs.append(block_EZ)
				block_EZZTs.append(block_EZZT)
				block_EXs.append(block_EX)
				block_EXZs.append(block_EXZ)
				block_EX2s.append(block_EX2)

			EZ[i], EZZT[i], EX[i], EXZ[i], EX2[i] = combineMatrices(y_indices_to_use[i], block_EZs, block_EZZTs, block_EXs, block_EXZs, block_EX2s)

		new_A, new_mus, new_sigmas, new_decay_coef = Mstep(Y, EZ, EZZT, EX, EXZ, EX2, A, mus, sigmas, decay_coef, singleSigma=singleSigma)

		try:
			checkNoNans([EZ, EZZT, EX, EXZ, EX2, new_A, new_mus, new_sigmas, new_decay_coef])
		except:
			print("Error: algorithm failed to converge. Usual solutions to this problem: filtering out genes which are zero more than 80 - 90% of the time, or using standard ZIFA. Automatically retrying ZIFA when filtering out genes.")
			return None

		paramsNotChanging = True
		max_param_change = 0
		for new, old in [[new_mus, mus], [new_A, A], [new_sigmas, sigmas], [new_decay_coef, decay_coef]]:
			rel_param_change = np.mean(np.abs(new - old)) / np.mean(np.abs(new))
			if rel_param_change > max_param_change:
				max_param_change = rel_param_change
			if rel_param_change > param_change_thresh:
				paramsNotChanging = False

		A = new_A
		mus = new_mus
		sigmas = new_sigmas
		decay_coef = new_decay_coef

		if paramsNotChanging:
			print('Param change below threshold %2.3e after %i iterations' % (param_change_thresh, n_iter))
			break

		if n_iter >= max_iter:
			print('Maximum number of iterations reached; terminating loop')

		n_iter += 1

	params = {'A': A, 'mus': mus, 'sigmas': sigmas, 'decay_coef': decay_coef, 'X': EX}

	return EZ, params


def fitModel(Y, K, singleSigma=False, n_blocks=None, p0_thresh=.95):
	"""
	fits the model to data.
	Input:
	Y: data matrix, n_samples x n_genes
	K: number of latent components
	singleSigma: if True, fit only a single variance parameter (zero-inflated PPCA) rather than a different one for every gene (zero-inflated factor analysis).
	p0_thresh: filters out genes that are zero in more than this proportion of samples.
	Returns:
	EZ: the estimated positions in the latent space, n_samples x K
	params: a dictionary of model parameters. Throughout, we refer to lambda as "decay_coef".
	"""

	Y = deepcopy(Y)
	assert(p0_thresh >= 0 and p0_thresh <= 1)

	print('Filtering out all genes which are zero in more than %2.1f%% of samples. To change this, change p0_thresh.' % (p0_thresh * 100))

	Y = Y[:, (np.abs(Y) < 1e-6).mean(axis=0) <= p0_thresh]
	results = runEMAlgorithm(Y, K, singleSigma=singleSigma, n_blocks=n_blocks)

	while results is None:
		Y_is_zero = np.abs(Y) < 1e-6
		max_zero_frac = Y_is_zero.mean(axis=0).max()
		new_max_zero_frac = max_zero_frac * .95
		print('Previously, maximum fraction of zeros for a gene was %2.3f; now lowering that to %2.3f and rerunning ZIFA' % (max_zero_frac, new_max_zero_frac))

		Y = Y[:, Y_is_zero.mean(axis=0) < new_max_zero_frac]
		print('After filtering out genes with too many zeros, %i samples and %i genes' % Y.shape)

		results = runEMAlgorithm(Y, K, singleSigma=singleSigma, n_blocks=n_blocks)

	return results


if __name__ == '__main__':

	Y = np.array([[0, 0, 0], [0, 0, .2], [0, 0.1, 0]])
	fitModel(Y, 2)
