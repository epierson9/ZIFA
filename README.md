# ZIFA
Zero-inflated dimensionality reduction algorithm for single-cell data. Created by Emma Pierson and Christopher Yau.

## Citation

 @article{pierson2015zifa,
  title={ZIFA: Dimensionality reduction for zero-inflated single-cell gene expression analysis},
  author={Pierson, Emma and Yau, Christopher},
  journal={Genome biology},
  volume={16},
  number={1},
  pages={1},
  year={2015},
  publisher={BioMed Central}
}

## Instructions

If you are using count data, we recommend taking the log (ie, Y = log2(1 + count_data)) prior to using ZIFA. 

Algorithm code is contained in ZIFA.py and block_ZIFA.py. For datasets with more than a few thousand genes, we recommend using block_ZIFA, which subsamples genes in blocks to increase efficiency; it should yield similar results to ZIFA. Runtime for block ZIFA on the full single-cell dataset from Pollen et al, 2014 (~250 samples, ~20,000 genes) is approximately 15 minutes on a quadcore Mac Pro. 

Runtime for block ZIFA is roughly linear in the number of samples and the number of genes, and quadratic in the block size. 
Decreasing the block size may decrease runtime but will also produce less reliable results. 

See example.py for a full example demonstrating superior performance over factor analysis. 

See read_in_real_data_example.py for a example demonstrating how to read in real data using pandas and run ZIFA on it. 

ZIFA requires pylab, scipy, numpy, and scikits.learn for full functionality. 

Please contact emmap1@cs.stanford.edu with any questions or comments. Prior to issuing pull requests, please confirm that your code passes the tests by running unitTests.py. The tests take about 30 seconds to run. 

##Installation

Download the code: `git clone https://github.com/epierson9/ZIFA`

Install the package: `cd ZIFA` then `python setup.py install`

##Sample usage

```python
from ZIFA import ZIFA
from ZIFA import block_ZIFA
```

To fit ZIFA:

```python
Z, model_params = ZIFA.fitModel(Y, k)
```

To fit with the block algorithm:

```python
Z, model_params = block_ZIFA.fitModel(Y, k)
```

or 

```python
Z, model_params = block_ZIFA.fitModel(Y, k, n_blocks = desired_n_blocks)
```

where Y is the observed zero-inflated data, k is the desired number of latent dimensions, and Z is the low-dimensional projection and desired_n_blocks is the number of blocks to divide genes into. By default, the number of blocks is set to n_genes / 500 (yielding a block size of approximately 500). 
 
