from __future__ import print_function
from ZIFA import ZIFA, block_ZIFA
import numpy as np
import random
from copy import deepcopy
from sklearn.decomposition import FactorAnalysis
from example import generateSimulatedDimensionalityReductionData
from scipy.stats import pearsonr
import platform
import scipy
import sklearn

def unitTests():
    """
    Test ZIFA and block ZIFA under a variety of conditions to make sure projected dimensions and parameters don't change. 
    """

    print("\n\n\n****Running unit tests!\nIMPORTANT: These unit tests pass with:\n\
    Python version 2.7.10 (your version: %s)\n\
    numpy 1.13.1 (your version: %s)\n\
    scipy 0.18.1 (your version: %s)\n\
    sklearn 0.16.1 (your version: %s)" % (platform.python_version(), np.__version__, scipy.__version__, sklearn.__version__))
    print("Different versions of Python or those packages may yield slightly different results and fail to pass the asserts unless you increase the absolute_tolerance parameter, set below.")
    print("If your configuration yields significantly different results, please contact emmap1@cs.stanford.edu.\n\n")

    absolute_tolerance = 1e-8

    random.seed(35)
    np.random.seed(32)

    n = 200
    d = 20
    k = 2
    sigma = .3
    n_clusters = 3
    decay_coef = .1

    X, Y, Z, ids = generateSimulatedDimensionalityReductionData(n_clusters, n, d, k, sigma, decay_coef)
    old_Y = deepcopy(Y)
    Zhat, params = ZIFA.fitModel(Y, k)
    assert np.allclose(Y, old_Y)


    # for Z and A, we compare the absolute values of the parameters because some package versions appear to flip the sign (which is fine and will not affect results)
    assert np.allclose(np.abs(Zhat[-1, :]), np.abs([ 1.50067515, 0.04742477]), atol=absolute_tolerance)
    assert np.allclose(np.abs(params['A'][0, :]), np.abs([ 0.66884415, -0.17173555]), atol=absolute_tolerance)
    assert np.allclose(params['decay_coef'], 0.10458794970222711, atol=absolute_tolerance)
    assert np.allclose(params['sigmas'][0], 0.30219903, atol=absolute_tolerance)

    Zhat, params = block_ZIFA.fitModel(Y, k) 
    assert np.allclose(Y, old_Y)
    assert np.allclose(np.abs(Zhat[-1, :]), np.abs([1.49712162, 0.05823952]), atol=absolute_tolerance) # this is slightly different (though highly correlated) because ZIFA runs one extra half-step of EM
    assert np.allclose(np.abs(params['A'][0, :]), np.abs([ 0.66884415, -0.17173555]), atol=absolute_tolerance)
    assert np.allclose(params['decay_coef'], 0.10458794970222711, atol=absolute_tolerance)
    assert np.allclose(params['sigmas'][0], 0.30219903, atol=absolute_tolerance)

    Zhat, params = block_ZIFA.fitModel(Y, k, n_blocks = 3)
    assert np.allclose(Y, old_Y)
    assert np.allclose(np.abs(Zhat[-1, :]), np.abs([  9.84455438e-01, 4.50924335e-02]), atol=absolute_tolerance)

    n = 50
    d = 60
    k = 3
    sigma = .3
    n_clusters = 3
    decay_coef = .1

    X, Y, Z, ids = generateSimulatedDimensionalityReductionData(n_clusters, n, d, k, sigma, decay_coef)
    old_Y = deepcopy(Y)
    Zhat, params = block_ZIFA.fitModel(Y, k, n_blocks = 3)
    assert np.allclose(Y, old_Y)
    assert np.allclose(np.abs(Zhat[-1, :]), np.abs([-1.69609638,-0.5475882, 0.08008015]), atol=absolute_tolerance)

    X, Y, Z, ids = generateSimulatedDimensionalityReductionData(n_clusters, n, d, k, sigma, decay_coef)
    old_Y = deepcopy(Y)
    Zhat, params = ZIFA.fitModel(Y, k)
    print(Zhat[-1, :])
    assert np.allclose(np.abs(Zhat[-1, :]), np.abs([-0.63075905, -0.77361427, -0.11544281]), atol=absolute_tolerance)
    assert np.allclose(Y, old_Y)
    
    print('Tests passed with absolute tolerance %2.3e!' % absolute_tolerance)

if __name__ == '__main__':
    unitTests()