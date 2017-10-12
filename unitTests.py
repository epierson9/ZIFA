from __future__ import print_function
from ZIFA import ZIFA, block_ZIFA
import numpy as np
import random
from copy import deepcopy
from sklearn.decomposition import FactorAnalysis
from example import generateSimulatedDimensionalityReductionData
from scipy.stats import pearsonr


def unitTests():
    """
    Just test ZIFA and block ZIFA under a variety of conditions to make sure projected dimensions don't change. 
    """
    random.seed(35)
    np.random.seed(32)

    n = 200
    d = 20
    k = 2
    sigma = .3
    n_clusters = 3
    decay_coef = .1

    X, Y, Z, ids = generateSimulatedDimensionalityReductionData(n_clusters, n, d, k, sigma, decay_coef)
    Zhat, params = ZIFA.fitModel(Y, k)
    assert np.allclose(Zhat[-1, :], [ 1.50067515, 0.04742477])
    assert np.allclose(params['A'][0, :], [ 0.66884415, -0.17173555])
    assert np.allclose(params['decay_coef'], 0.10458794970222711)
    assert np.allclose(params['sigmas'][0], 0.30219903)

    Zhat, params = block_ZIFA.fitModel(Y, k) 
    assert np.allclose(Zhat[-1, :], [1.49712162, 0.05823952]) # this is slightly different (though highly correlated) because ZIFA runs one extra half-step of EM
    assert np.allclose(params['A'][0, :], [ 0.66884415, -0.17173555])
    assert np.allclose(params['decay_coef'], 0.10458794970222711)
    assert np.allclose(params['sigmas'][0], 0.30219903)

    Zhat, params = block_ZIFA.fitModel(Y, k, n_blocks = 3)
    assert np.allclose(Zhat[-1, :], [  9.84455438e-01, 4.50924335e-02])

    n = 50
    d = 60
    k = 3
    sigma = .3
    n_clusters = 3
    decay_coef = .1

    X, Y, Z, ids = generateSimulatedDimensionalityReductionData(n_clusters, n, d, k, sigma, decay_coef)
    Zhat, params = block_ZIFA.fitModel(Y, k, n_blocks = 3)
    assert np.allclose(Zhat[-1, :], [-1.69609638,-0.5475882, 0.08008015])

    X, Y, Z, ids = generateSimulatedDimensionalityReductionData(n_clusters, n, d, k, sigma, decay_coef)
    Zhat, params = ZIFA.fitModel(Y, k)
    print(Zhat[-1, :])
    assert np.allclose(Zhat[-1, :], [-0.63075905, -0.77361427, -0.11544281])

    
    print('Tests passed!')

if __name__ == '__main__':
    unitTests()