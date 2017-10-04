import csv
import scipy
import numpy as np
from ZIFA import ZIFA
from ZIFA import block_ZIFA
import pandas as pd

# This gives an example for how to read in a real data called input.table. 
# genes are columns, samples are rows, each number is separated by a space. 
# If you do not want to install pandas, you can also use np.loadtxt: https://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html

file = pd.read_csv('input.table', sep=' ')
table = np.array(file)
Z, model_params = block_ZIFA.fitModel(table, 5)
np.savetxt('output.ZIFA.table', Z, fmt='%.2f')
