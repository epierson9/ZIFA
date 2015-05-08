# ZIFA
Zero-inflated dimensionality reduction algorithm for single-cell data

Algorithm code is contained in ZIFA.py. 

Sample usage: 

import ZIFA
Z, model_params = ZIFA.fitModel(Y, k)

where Y is the observed zero-inflated data, k is the desired number of latent dimensions, and Z is the low-dimensional projection. 

See example.py for a full example.

This code requires pylab, scipy, numpy, and scikits.learn for full functionality. 
 
